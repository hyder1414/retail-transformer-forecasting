# HAHRT – Holiday‑Aware Hierarchical Residual Transformer

This repo implements a full pipeline for **multi‑store, multi‑department Walmart sales forecasting**.

The **main model** is `HAHRT` – a **H**oliday‑**A**ware **H**ierarchical **R**esidual **T**ransformer.  
Other models (naïve baselines, XGBoost/GBDT, an older Transformer) are for **benchmarking / experiments only**.

---

## 1. Problem & dataset

We forecast `Weekly_Sales` for each `(Store, Dept)` over time using:

- Past sales history
- Calendar & holidays
- Weather (`Temperature`)
- Macro features (`CPI`, `Unemployment`)
- Markdown / promo signals (`MarkDown1–5`)
- Store‑level attributes (`Type`, `Size`)

The data is compatible with the Kaggle **“Walmart Recruiting – Store Sales Forecasting”** format and expects the following files:

- `data/raw/train.csv`
- `data/raw/features.csv`
- `data/raw/stores.csv`

(There is also `sampleSubmission.csv` and `test.csv` in `data/raw/`, but they are not required for training.)

---

## 2. Repo layout

```text
.
├── data
│   ├── raw/            # Original CSVs (train.csv, features.csv, stores.csv, etc.)
│   ├── interim/        # Joined data, baseline features, residuals, checkpoints
│   └── processed/      # Older windowed dataset for the simple Transformer baseline
├── experiments/        # (Optional) experiment artifacts / notebooks
├── notebooks/          # Jupyter notebooks
├── reports/
│   ├── figures/        # Plots
│   └── metrics/        # Metrics dumps
├── requirements.txt    # Python dependencies
└── src
    ├── config
    │   ├── paths.py              # PROJECT_ROOT, DATA_DIR, RAW_DIR, etc.
    │   └── data_config.py        # Legacy config for older Transformer data pipeline
    ├── features
    │   ├── build_dataset.py      # Build data/interim/joined.parquet (old pipeline)
    │   ├── build_baseline_features.py  # Feature engineering for XGBoost baseline
    │   ├── build_windows.py      # Legacy window builder for the simple Transformer
    │   └── dataset.py            # Dataset for the legacy Transformer
    ├── utils
    │   └── metrics.py            # WMAE and other metrics
    ├── eval
    │   └── eval_gbdt_wmae.py     # Eval script for GBDT/XGBoost baselines
    └── models
        ├── hahrt_model.py            # **Main HAHRT architecture**
        ├── hahrt_data.py             # Data prep, GBDT baseline, residuals, seq. datasets
        ├── train_hahrt.py            # **Main training script for HAHRT**
        ├── train_hahrt_rc_gbdt.py    # Variant with weaker GBDT (ablation)
        ├── train_baseline_lgbm.py    # XGBoost baseline on baseline_features.parquet
        ├── transformer_forecaster.py # Minimal Transformer forecaster (legacy)
        ├── train_transformer.py      # Training script for the legacy Transformer
        └── baseline_naive.py         # Naïve/seasonal naïve baselines
```

> **If you’re just here to use the main model, focus on:**
> - `src/models/hahrt_model.py`
> - `src/models/hahrt_data.py`
> - `src/models/train_hahrt.py`
> - `data/interim/hahrt_best_model*.pt` (already‑trained checkpoints, if present)

---

## 3. Environment setup

From the project root (same folder as `src/` and `data/`):

```bash
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate        # Windows (PowerShell / cmd)

pip install --upgrade pip
pip install -r requirements.txt
```

Make sure the raw CSVs are present:

```text
data/raw/train.csv
data/raw/features.csv
data/raw/stores.csv
```

---

## 4. HAHRT pipeline (main model)

### 4.1 High‑level idea

HAHRT does **residual forecasting** instead of directly predicting sales:

1. Train a **tabular GBDT baseline** on row‑level features  
   (`train_gbdt_and_compute_residuals` in `hahrt_data.py`).
2. Compute residuals:

   \[
   \text{Residual} = y_{\text{true}} - y_{\text{GBDT}}
   \]

3. Build sliding windows per time series (each `(Store, Dept)`) using `build_sequence_datasets`.
4. Train a **global Transformer** to predict the residual at the last step of each window.
5. Final prediction at inference time:

   \[
   \hat{y}_{\text{final}} = \hat{y}_{\text{GBDT}} + \hat{r}_{\text{HAHRT}}
   \]

**Loss / metric**

- Training loss: **Weighted MAE (WMAE)** on residuals  
  (holiday weeks get higher weight).
- Validation metrics (see `evaluate` in `train_hahrt.py`):
  - `wmae_resid` – WMAE on residuals
  - `wmae_final` – WMAE on final sales (`Baseline_Pred + Residual_Pred`)

### 4.2 Input features used by HAHRT

The sequence inputs are constructed in `hahrt_data.py`.

For each sliding window of length `L = input_window` (default 24):

#### Continuous sequence features (`x_cont`, shape `[B, L, D]`)

From `cont_feature_cols`:

- `Residual` – most recent residuals in the window
- `Temperature`
- `Fuel_Price`
- `CPI`
- `Unemployment`
- `Temp_Anomaly` – temperature minus the average for that week of year
- `MarkDown1–5`
- `lag_1`, `lag_2`, `lag_7`, `lag_52`
- `roll_mean_4`, `roll_mean_8`, `roll_mean_13`

These are all numeric and are filled with `0.0` where missing.

#### Time / holiday tokens (per timestep)

- `week_of_year` – ISO week of year
- `year_index` – integer index of the year
- `is_holiday` – `0/1` flag

#### Static / hierarchical features (per sequence)

- `store_idx` – encoded store ID
- `dept_idx` – encoded department ID
- `store_type_idx` – encoded store type
- `size_norm` – normalized store size

#### Targets (for the last timestep in the window)

In `HAHRTSequenceDataset.__getitem__`:

- `target_resid` – residual for the last timestep
- `target_weight` – WMAE weight (5 for holiday weeks, 1 otherwise)
- `baseline_pred_target` – GBDT prediction at that timestep
- `target_y` – true `Weekly_Sales`

---

## 5. HAHRT architecture

Defined in `src/models/hahrt_model.py`.

### 5.1 Configuration (`HAHRTConfig`)

```python
@dataclass
class HAHRTConfig:
    input_dim: int
    num_stores: int
    num_depts: int
    num_store_types: int
    max_week_of_year: int
    max_year_index: int

    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    holiday_bias_strength: float = 1.5  # bias toward holiday timesteps
```

The `meta` dict returned by `build_sequence_datasets` fills in the non‑default fields.

### 5.2 Components

**1. Continuous feature projection**

```python
self.input_proj = nn.Linear(cfg.input_dim, d_model)
```

All continuous inputs (`x_cont`) are projected into the model dimension.

**2. Calendar & holiday embeddings**

The model learns embeddings for:

- `week_of_year` → `week_embed`
- `year_index` → `year_embed`
- `is_holiday` (0/1) → `holiday_embed`

These are added to the projected inputs:

```python
x = self.input_proj(x_cont)
x = x + week_emb + year_emb + hol_emb
```

This acts as a learned **seasonal + holiday‑aware positional encoding**.

**3. Static hierarchical embeddings + FiLM**

Static per‑series features are embedded:

```python
store_e = self.store_embed(store_idx)
dept_e  = self.dept_embed(dept_idx)
type_e  = self.store_type_embed(store_type_idx)
size_e  = self.size_mlp(size_norm)
```

They are concatenated and passed through a small MLP to produce **FiLM parameters** (`gamma`, `beta`) of size `d_model`:

```python
static_vec = torch.cat([store_e, dept_e, type_e, size_e], dim=-1)
film_params = self.static_to_film(static_vec)  # [B, 2 * d_model]
gamma, beta = torch.chunk(film_params, 2, dim=-1)
```

Every Transformer layer uses these `(gamma, beta)` to modulate the hidden states, effectively giving each `(Store, Dept)` its own adaptation of the global model:

```python
x = x * (1 + gamma[:, None, :]) + beta[:, None, :]
```

This is the **hierarchical** part of HAHRT.

**4. Holiday‑Aware multi‑head self‑attention**

`HolidayAwareSelfAttention` modifies standard scaled dot‑product attention:

- Compute standard scores `QK^T / sqrt(d_head)`
- Add a learned bias on attention scores for timesteps where `is_holiday == 1`
  (controlled by `holiday_bias_strength`)

This pushes the model to pay more attention to historical holiday weeks when forming predictions.

**5. Transformer encoder stack**

`HolidayAwareTransformerEncoderLayer`:

- Holiday‑aware multi‑head self‑attention
- Position‑wise feed‑forward network with ReLU
- LayerNorm and dropout
- FiLM modulation using `(gamma, beta)` at each layer

The model stacks `cfg.num_layers` of these layers in `self.layers`.

**6. Output head**

After the encoder stack:

- Take the last timestep representation `last_hidden = x[:, -1, :]`
- Predict a scalar residual correction:

```python
self.output_head = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(),
    nn.Linear(d_model, 1),
)
```

The forward pass returns a tensor of shape `[B]` with the residual correction.

---

## 6. Training HAHRT from scratch

From the project root (after environment setup and with raw CSVs in `data/raw/`):

```bash
python -m src.models.train_hahrt   --val_start_date 2012-01-01   --input_window 24   --batch_size 64   --epochs 30   --lr 3e-4   --d_model 128   --n_heads 4   --num_layers 3   --d_ff 256   --dropout 0.1   --holiday_bias_strength 1.5
```

You can omit flags to use the defaults defined in `train_hahrt.py`.

The script:

1. Calls `prepare_walmart_dataframe()` to:
   - Load and merge `train.csv`, `features.csv`, `stores.csv`
   - Add calendar features (`week_of_year`, `month`, `year_index`, etc.)
   - Compute `Temp_Anomaly`
   - Build lag and rolling mean features
   - Encode store/department indices and store type
2. Calls `train_gbdt_and_compute_residuals()` to:
   - Train a `GradientBoostingRegressor` baseline on `feature_cols`
   - Add `Baseline_Pred` and `Residual` columns to the DataFrame
3. Calls `build_sequence_datasets()` to:
   - Construct sliding windows per `(Store, Dept)`
   - Split into train/validation based on `val_start_date`
   - Return `HAHRTSequenceDataset` objects and `meta` with dimensions
4. Instantiates `HAHRTModel(cfg)` and trains it using WMAE on residuals.
5. Saves the best model (lowest validation `wmae_final`) to:

   ```text
   data/interim/hahrt_best_model.pt
   ```

During training you’ll see logs like:

```text
Epoch 01 | Train WMAE (resid):  X.XX | Val WMAE (resid):  Y.YY | Val WMAE (final):  Z.ZZ
  -> New best model saved to data/interim/hahrt_best_model.pt
```

---

## 7. Using existing HAHRT checkpoints (no retraining)

If a teammate has already run `train_hahrt.py`, you should see one or more files in `data/interim/`:

- `hahrt_best_model.pt` – main checkpoint with strong GBDT baseline
- `hahrt_best_model_rc_gbdt.pt` – variant trained with reduced‑capacity GBDT
- `hahrt_best_model_weaker_gbdt.pt` – other ablations (if present)

To **reuse** a checkpoint and evaluate it:

```python
import torch
from torch.utils.data import DataLoader

from src.models.hahrt_model import HAHRTModel, HAHRTConfig
from src.models.hahrt_data import (
    prepare_walmart_dataframe,
    train_gbdt_and_compute_residuals,
    build_sequence_datasets,
    DATA_DIR,
)
from src.models.train_hahrt import evaluate, move_batch_to_device

VAL_START = "2012-01-01"
INPUT_WINDOW = 24
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Rebuild dataframe and residuals (must match training setup)
df = prepare_walmart_dataframe()
df = train_gbdt_and_compute_residuals(df, val_start_date=VAL_START)

# 2) Build datasets/loaders
train_ds, val_ds, meta = build_sequence_datasets(
    df, input_window=INPUT_WINDOW, val_start_date=VAL_START
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 3) Config and model
cfg = HAHRTConfig(
    input_dim=meta["input_dim"],
    num_stores=meta["num_stores"],
    num_depts=meta["num_depts"],
    num_store_types=meta["num_store_types"],
    max_week_of_year=meta["max_week_of_year"],
    max_year_index=meta["max_year_index"],
    d_model=128,
    n_heads=4,
    num_layers=3,
    d_ff=256,
    dropout=0.1,
    holiday_bias_strength=1.5,
)

model = HAHRTModel(cfg).to(device)

# 4) Load checkpoint
ckpt_path = DATA_DIR / "interim" / "hahrt_best_model.pt"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)

# 5) Evaluate
metrics = evaluate(model, val_loader, device)
print(metrics)  # {'wmae_resid': ..., 'wmae_final': ...}
```

You can adapt this to generate predictions for custom date ranges or specific stores/departments.

---

## 8. Other models / scripts (optional, baselines)

### 8.1 XGBoost (GBDT) baseline

Files:

- `src/features/build_dataset.py`
- `src/features/build_baseline_features.py`
- `src/models/train_baseline_lgbm.py`
- `src/eval/eval_gbdt_wmae.py`

Steps:

```bash
# 1) Build joined.parquet from raw CSVs
python -m src.features.build_dataset

# 2) Build baseline_features.parquet with lags/rolling means
python -m src.features.build_baseline_features

# 3) Train XGBoost baseline
python -m src.models.train_baseline_lgbm

# 4) Evaluate WMAE of the GBDT/XGBoost baseline
python -m src.eval.eval_gbdt_wmae
```

This baseline is useful to compare against HAHRT.

### 8.2 Simple Transformer forecaster (legacy baseline)

Files:

- `src/features/dataset.py`
- `data/processed/windows_L52_H4.npz`
- `src/models/transformer_forecaster.py`
- `src/models/train_transformer.py`

This is a simpler model that:

- Uses prebuilt windows of length 52 to predict the next 4 weeks.
- Uses sinusoidal positional encoding.
- Does **not** use residual modeling, holiday bias, or hierarchical FiLM.

Run:

```bash
python -m src.models.train_transformer
```

### 8.3 Naïve baselines

File:

- `src/models/baseline_naive.py`

Run:

```bash
python -m src.models.baseline_naive
```

These give last‑value / seasonal naïve baselines for quick sanity checks.

---

## 9. Quick guide for teammates

**If you only want to run the already‑trained model:**

1. Make sure you have the raw CSVs in `data/raw/`.
2. Activate the virtual environment and `pip install -r requirements.txt`.
3. Use the Python snippet in section 7 to load `hahrt_best_model.pt` and evaluate.

**If you want to retrain HAHRT from scratch:**

```bash
python -m src.models.train_hahrt
```

(optionally add CLI flags to control `epochs`, `input_window`, etc.)

**If you want baselines for comparison:**

1. Run `build_dataset.py` and `build_baseline_features.py`.
2. Run `train_baseline_lgbm.py` and `eval_gbdt_wmae.py`.
3. Optionally run `train_transformer.py` or `baseline_naive.py` to compare different model classes.

---

## 10. Summary of novelty

Compared to a standard Transformer or pure GBDT, this project combines:

- **Residual modeling**: a strong GBDT baseline plus a sequence model for residuals.
- **Hierarchical conditioning**: store/department/type/size embeddings used via **FiLM modulation** in every Transformer layer.
- **Holiday‑aware attention**: explicit attention bias toward holiday timesteps so the model can reuse historical holiday patterns more effectively.
- **Rich covariates**: temperature anomaly, markdowns, macro indicators, and lag/rolling features fed as continuous inputs.

Together, these design choices make HAHRT a **global, holiday‑aware, hierarchy‑conditioned residual forecaster** suitable for realistic retail demand forecasting at scale.
