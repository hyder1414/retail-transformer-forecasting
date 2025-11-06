# 🧠 Global Transformer for Retail Demand Forecasting  
**Calibrated Multi-Horizon Predictions with Price & Promotion Effects**

---

### 📍 Overview
This project implements a **global Transformer-based time-series forecaster** for retail demand (Walmart-like M5 dataset).  
It learns shared representations across tens of thousands of item series and incorporates exogenous factors such as price, promotions, holidays, and weather.

---

### 🧰 Tech Stack
| Area | Tools |
|------|-------|
| Data Processing | Polars • DuckDB • PyArrow |
| Modeling | PyTorch 2.2 • PyTorch Lightning 2.5 |
| Baselines | LightGBM • XGBoost • ARIMA/ETS (statsmodels) |
| Experiment Tracking | MLflow |
| Environment | Python 3.11 (venv) |

---

### 🚀 Quickstart

```bash
# 1️⃣  Clone this repo
git clone https://github.com/<your-username>/retail-transformer-forecasting.git
cd retail-transformer-forecasting

# 2️⃣  Create environment
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3️⃣  Place data
#   Expected structure:
#   data/raw/{calendar.csv, sales_train_validation.csv, sell_prices.csv, ...}

# 4️⃣  (Optional) Run the sanity check notebook
#   notebooks/00_sanity_check.ipynb

# 5️⃣  Work in branches
git checkout -b feature/<your-name>



### Project Structure
data/
 ├── raw/            # Original CSVs (M5, holidays, weather)
 ├── interim/        # Lightly processed or sampled
 └── processed/      # Final model-ready tables
src/
 ├── config/         # YAML or JSON experiment configs
 ├── features/       # Feature generation scripts
 ├── models/         # Baselines & Transformer models
 ├── eval/           # Metrics & calibration utilities
 └── utils/          # Helpers & logging
notebooks/           # EDA & sanity notebooks
reports/             # Figures & metrics for write-up
experiments/         # MLflow or ClearML runs



### Team
Bat-Amgalan Enkhtaivan

Haider Khan

Nigar Aliyeva

Saanvi Joginipally

Usha Vuchidi