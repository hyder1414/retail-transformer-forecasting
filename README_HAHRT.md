# MSML 612 Project – Holiday-Aware Hierarchical Residual Transformer (HAHRT)

This repository contains our MSML 612 graduate project on **multi-store, multi-department retail demand forecasting** using a **Transformer-based neural network** on **Walmart** weekly sales data.

We work with the **Walmart Recruiting – Store Sales Forecasting** dataset (`train.csv`, `features.csv`, `stores.csv`) and design a **hybrid model**:

> **GBDT baseline + Holiday-Aware Hierarchical Residual Transformer (HAHRT)**

The goal is to forecast weekly sales with low **Weighted Mean Absolute Error (WMAE)**, where **holiday weeks are penalized 5× more** than non-holiday weeks.

---

## 1. Problem Overview

- **Task:** Time-series regression (continuous output), not classification  
- **Unit:** `(Store, Dept)` pair  
- **Input:** Previous `L` weeks of:
  - Sales history (lagged `Weekly_Sales`)
  - Calendar: `Date`, `week_of_year`, `month`, `year`, `IsHoliday`
  - External: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
  - Price/promo: `MarkDown1`–`MarkDown5`
  - Store metadata: `Type` (A/B/C), `Size`
- **Output:** Next-week `Weekly_Sales` for every `(Store, Dept)`  
- **Metric:** **WMAE** (Weighted MAE with 5× weight for holiday weeks)

We train a **global model** over all stores and departments, with a **date-based train/val split**.

---

## 2. Dataset

We use the original Walmart competition files:

- `data/raw/train.csv`  
  `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`
- `data/raw/features.csv`  
  `Store`, `Date`, `Temperature`, `Fuel_Price`, `MarkDown1`–`MarkDown5`, `CPI`, `Unemployment`, `IsHoliday`
- `data/raw/stores.csv`  
  `Store`, `Type` (A/B/C), `Size`

### Expected layout

```text
data/
  raw/
    train.csv
    features.csv
    stores.csv
  interim/
  processed/
reports/
src/
```

> `data/interim/` and model checkpoints are generated automatically and are git-ignored.

---

## 3. Model Architecture – HAHRT

We propose **HAHRT – Holiday-Aware Hierarchical Residual Transformer**, a 2-stage hybrid model.

### Stage 1 – GBDT Baseline (Tabular Model)

For each `(Store, Dept, Date)` we engineer:

- **Identifiers / static**
  - `StoreIdx`, `DeptIdx`, `StoreTypeIdx`, normalized `Size` (`SizeNorm`)
- **Calendar**
  - `week_of_year`, `month`, `year`, `year_index`, `IsHoliday`
- **External / macro**
  - `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
  - `Temp_Anomaly` = `Temperature` – average temperature for that `week_of_year`
- **Lagged & rolling sales**
  - `lag_1`, `lag_2`, `lag_7`, `lag_52`
  - `roll_mean_4`, `roll_mean_8`, `roll_mean_13`
- **Markdowns**
  - `MarkDown1`–`MarkDown5` (filled with 0 when missing)

We train a **GradientBoostingRegressor** (scikit-learn) and compute:

- `Baseline_Pred` = GBDT prediction of `Weekly_Sales`
- `Residual` = `Weekly_Sales - Baseline_Pred`

These are saved into:

```text
data/interim/hahrt_with_residuals.parquet
```

### Stage 2 – HAHRT (Transformer on Residuals)

We treat each `(Store, Dept)` as a time series and build sliding windows:

- **Input window**: e.g. `L = 24` weeks of history
- For each window ending at time `t`, we construct:
  - Continuous covariates (`x_cont`): `Residual` history, externals, lags, rolling stats, markdowns, etc.
  - Time features: `week_of_year`, `year_index`, `IsHoliday`
  - Static/hierarchical features: `StoreIdx`, `DeptIdx`, `StoreTypeIdx`, `SizeNorm`

**Transformer core:**

- Global **Transformer encoder** over `[L, D]` residual sequences
- **Static identity embeddings** (Store, Dept, Type, Size) used to:
  - build a static context vector per series
  - produce **FiLM-style adapters** (γ, β) that modulate layer activations:
    h_t <- h_t * (1 + γ_{store,dept}) + β_{store,dept}
- **Holiday awareness:**
  - `IsHoliday` embeddings
  - extra gating on holiday timesteps so the model can treat them differently

**Output head:**

- Predicts a **residual correction** r_hat_t
- Final forecast:
  y_hat_t^final = y_hat_t^GBDT + r_hat_t^Transformer

**Loss (metric-aligned):**

- We train on the **residuals** with **Weighted MAE**:
  - weight = 5 for holiday weeks, weight = 1 otherwise (matches competition WMAE)
- (Optional) a small MSE term can be added for stability.

---

## 4. Repo Structure (Core Files)

```text
src/
  models/
    hahrt_data.py       # Data prep, feature engineering, GBDT baseline, residuals, sequence datasets
    hahrt_model.py      # HAHRT Transformer model definition
    train_hahrt.py      # Orchestration + training loop for HAHRT (Stage 2) and baseline (Stage 1)
    train_baseline_lgbm.py  # Earlier LGBM-style baseline (tabular)
  eval/
    eval_gbdt_wmae.py   # Compute WMAE for GBDT baseline only
  features/
    build_windows.py    # Legacy sliding-window builder for earlier experiments
  config/
    ...                 # Optional configs from earlier baselines
```

Older transformer experiments are kept for comparison, but **HAHRT is the main final model**.

---

## 5. Environment Setup

From the project root:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scriptsctivate         # Windows (if needed)

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure these files exist before training:

```text
data/raw/train.csv
data/raw/features.csv
data/raw/stores.csv
```

---

## 6. Training HAHRT (GBDT + Transformer)

This command:

- Loads & merges Walmart CSVs
- Engineers calendar + lag features
- Trains the **GBDT baseline** and computes residuals
- Saves `data/interim/hahrt_with_residuals.parquet`
- Builds sequence datasets for `(Store, Dept)` time series
- Trains the **HAHRT Transformer** and tracks validation WMAE

```bash
python -m src.models.train_hahrt --epochs 10 --batch_size 64
```

Example log:

```text
Loading and preparing Walmart data...
Building HAHRT sequence datasets...
Instantiating HAHRT model...
Epoch 01 | Train Loss: 2389697.42 | Val WMAE (resid): 1574.78 | Val WMAE (final): 1574.78
  -> New best model saved to data/interim/hahrt_best_model.pt
...
```

- **`Val WMAE (final)`** = WMAE of final forecast (`Baseline_Pred + residual correction`)
- `data/interim/hahrt_best_model.pt` = best Transformer checkpoint by validation WMAE.

Hyperparameters (e.g. `input_window`, `d_model`, number of layers/heads, GBDT depth/trees) can be tuned in `hahrt_data.py` and `train_hahrt.py`.

---

## 7. Evaluating the GBDT Baseline Only

To see how much the Transformer helps over the tree baseline, run:

```bash
python -m src.eval.eval_gbdt_wmae
```

This script:

- Loads `data/interim/hahrt_with_residuals.parquet`
- Applies the same validation split as training (e.g. `Date >= 2012-01-01`)
- Computes **validation WMAE using only `Baseline_Pred`** from the GBDT

Example output:

```text
Validation WMAE (GBDT baseline only): 1653.26
```

You can then compare:

- GBDT baseline WMAE (e.g. 1653.26)
- HAHRT WMAE from `train_hahrt.py` logs (e.g. ~1575)

to quantify the improvement contributed by the Transformer stage.

---

## 8. How Teammates Can Contribute

Some natural collaboration areas:

- **Feature Engineering**  
  Edit `prepare_walmart_dataframe()` in `hahrt_data.py` to:
  - add new covariates (e.g., pooled department signals),
  - refine holiday indicators, or economic features,
  - experiment with different lag/rolling windows.

- **Model Architecture (Transformer)**  
  Edit `hahrt_model.py` / `train_hahrt.py` to:
  - change number of layers, heads, hidden size,
  - add TFT-inspired components (variable gating, static enrichment),
  - implement holiday-biased attention or other architectural tweaks.

- **Evaluation & Visualization**  
  Add scripts under `src/eval/` to:
  - compute holiday vs non-holiday WMAE,
  - plot True vs Baseline vs HAHRT for selected `(Store, Dept)` pairs,
  - generate figures/tables for the paper and slides.

- **Live Demo**  
  Implement a simple CLI or Streamlit app that:
  - lets you pick a store/department,
  - loads the best checkpoint,
  - shows history + GBDT baseline + HAHRT forecast and prints WMAE.

---

## 9. What to Show in the Presentation

For a 10–15 minute talk, we recommend:

1. **Problem & Data**
   - Explain Walmart weekly sales data, multi-store/multi-dept structure.
   - Emphasize holiday weighting in the WMAE metric.

2. **Architecture Diagram**
   - Stage 1: GBDT on tabular features → `Baseline_Pred`
   - Stage 2: HAHRT Transformer on residual sequences → residual correction
   - Show static embeddings, FiLM adapters, holiday-awareness.

3. **Metrics Table**
   - (Optional) Naive or seasonal-naive baseline
   - GBDT baseline WMAE
   - HAHRT WMAE

4. **Qualitative Plot**
   - True vs GBDT vs HAHRT around holidays (Thanksgiving, Christmas, etc.)

5. **Novelty / Significance**
   - Residual Transformer on top of a strong GBDT
   - Holiday-aware, metric-aligned loss (WMAE with 5× holiday weight)
   - Hierarchical FiLM adapters for store/dept-level customization

6. **Limitations / Future Work**
   - Compare against more sophisticated TS baselines (STL/ARIMA, TFT)
   - Use richer item-level data (e.g., M5) on a subset
   - Explore deeper TFT-style components (variable selection, static enrichment)

---

## 10. Reproducibility

- Any teammate with the raw Walmart CSVs and Python ≥ 3.10 can:
  1. Create a virtualenv
  2. Install `requirements.txt`
  3. Run `python -m src.models.train_hahrt`
  4. Run `python -m src.eval.eval_gbdt_wmae`
- Intermediate artifacts and checkpoints live under `data/interim/` and are **not** tracked in git.
- Random seeds can be set inside `train_hahrt.py` for more reproducible experiments.
