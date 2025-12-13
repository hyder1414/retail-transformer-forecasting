# HAHRT – Holiday‑Aware Hierarchical Residual Transformer

Retail sales forecasting for every `(Store, Dept)` at Walmart scale. The main model is **HAHRT**, a Transformer that learns residuals on top of a strong GBDT baseline and is conditioned on holidays + hierarchy (store, department, type, size).

This README is written so a TA/professor can re-run the project end-to-end.

---

## 1) What you need

- Python 3.9+ with `venv`
- Raw data (Kaggle “Walmart Recruiting – Store Sales Forecasting”):
  - `data/raw/train.csv`
  - `data/raw/features.csv`
  - `data/raw/stores.csv`
- Dataset link: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting  
  (download via Kaggle CLI: `kaggle competitions download -c walmart-recruiting-store-sales-forecasting` and unzip into `data/raw/`)
- Optional: cached artifacts if provided with the repo
  - `data/interim/hahrt_best_model.pt` (HAHRT checkpoint)
  - `data/interim/hahrt_with_residuals.parquet` (GBDT residual cache)
  - `reports/metrics/hahrt_train_history.json` (training curves)

---

## 2) Quick start (evaluation only)

1) Create and activate a virtualenv, then install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Ensure the raw CSVs are in `data/raw/`.

3) If `data/interim/hahrt_best_model.pt` exists, run:
```bash
python - <<'PY'
import torch
from torch.utils.data import DataLoader
from src.models.hahrt_model import HAHRTModel, HAHRTConfig
from src.models.hahrt_data import prepare_walmart_dataframe, train_gbdt_and_compute_residuals, build_sequence_datasets
from src.models.train_hahrt import evaluate, move_batch_to_device

VAL_START = "2012-01-01"
INPUT_WINDOW = 24
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = prepare_walmart_dataframe()
df = train_gbdt_and_compute_residuals(df, val_start_date=VAL_START)
_, val_ds, meta = build_sequence_datasets(df, input_window=INPUT_WINDOW, val_start_date=VAL_START)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

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
    dropout=0.15,
    holiday_bias_strength=1.5,
    use_film=True,
    use_local_conv=True,
)
model = HAHRTModel(cfg).to(DEVICE)
state = torch.load("data/interim/hahrt_best_model.pt", map_location=DEVICE)
model.load_state_dict(state)

metrics = evaluate(model, val_loader, DEVICE)
print(metrics)
PY
```
This rebuilds residuals, loads the checkpoint, and prints WMAE/MAE metrics on the validation split.

---

## 3) Training HAHRT from scratch

1) Install deps and place `data/raw/*.csv` as above.  
2) Train:
```bash
python -m src.models.train_hahrt \
  --val_start_date 2012-01-01 \
  --input_window 24 \
  --batch_size 128 \
  --epochs 20 \
  --lr 3e-4
```
Flags are optional; defaults live in `src/models/train_hahrt.py`.

What the script does:
- Merges raw CSVs and engineers calendar/weather/markdown/lag/rolling features.
- Trains a scikit-learn GBDT baseline and writes residuals to `data/interim/hahrt_with_residuals.parquet`.
- Builds sliding-window sequence datasets and trains the Transformer on residuals with holiday-weighted MAE.
- Saves the best checkpoint to `data/interim/hahrt_best_model.pt` and logs metrics/history to `reports/metrics/`.

---

## 4) Notebook (`notebooks/hahrt_report.ipynb`)

Purpose: show benchmarks, plots, and interactive store/department slices.

- Set `USE_CHECKPOINT = True` to load `data/interim/hahrt_best_model.pt` (default).
- Run all cells; if `reports/metrics/hahrt_train_history.json` exists, training curves will render; otherwise the notebook will say history is missing.
- The dropdown cell lets you pick a store/dept and compare Actual vs GBDT vs HAHRT.
- Extra plots summarize gains by store/department, store type, seasonality, holidays, and error distributions.

If you want to train inside the notebook, set `USE_CHECKPOINT = False` and add a training loop (otherwise it only evaluates).

---

## 5) Repository map

```text
data/
  raw/        # train.csv, features.csv, stores.csv (required)
  interim/    # residual caches, checkpoints (auto-created)
  processed/  # legacy windows for the simple Transformer baseline
notebooks/
  hahrt_report.ipynb   # main report + visuals
reports/
  metrics/    # JSON metrics and training history
src/
  models/
    hahrt_data.py      # data prep, GBDT baseline, seq datasets
    hahrt_model.py     # HAHRT architecture (holiday-aware attention + FiLM)
    train_hahrt.py     # main training script
    train_baseline_lgbm.py, baseline_naive.py, transformer_forecaster.py (baselines)
  features/            # legacy feature builders for baselines
  utils/metrics.py     # WMAE and helpers
```

---

## 6) Repro checklist (TA-friendly)

- [ ] `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] Raw CSVs in `data/raw/`
- [ ] (Optional) drop in provided checkpoint to `data/interim/hahrt_best_model.pt`
- [ ] `python -m src.models.train_hahrt` (skip if using the checkpoint)
- [ ] Run `notebooks/hahrt_report.ipynb` top to bottom to view metrics/plots

---

## 7) Notes and troubleshooting

- If you get a JSON error loading history, delete `reports/metrics/hahrt_train_history.json` and re-run training; the notebook now handles missing/invalid history gracefully.
- CPU-only is fine; training is slower but works. The scripts auto-pick `cuda` when available.
- GBDT training uses scikit-learn (no LightGBM dependency) to avoid OS-specific issues.
- If you modify `VAL_START_DATE` or `INPUT_WINDOW`, rebuild residuals by re-running `train_hahrt.py` so the cache matches the config.
