# Equipment Predictive Maintenance & Remaining Useful Life

A reproducible predictive-maintenance project for estimating **remaining useful life (RUL)** from multivariate equipment sensor data.

## Why this project
Heavy-equipment fleets generate continuous telemetry such as temperature, vibration, pressure, current, and efficiency. This project demonstrates an end-to-end ML workflow for converting that telemetry into an RUL estimate that can support maintenance planning.

## Features
- Synthetic degradation data generator for equipment units
- Rolling statistics and degradation-trend feature engineering
- Machine-level train/test split to avoid leakage across the same asset
- Random Forest regression baseline
- MAE, RMSE, and R² evaluation
- Feature-importance export
- Loader for NASA C-MAPSS turbofan degradation data

## Tech
Python, pandas, NumPy, scikit-learn, time-series feature engineering, Random Forest, statistical modeling.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/simulate_data.py
PYTHONPATH=src python src/train.py
```

Metrics are written to `outputs/metrics.json`.

## NASA C-MAPSS
Download the public C-MAPSS dataset separately and place a `train_FD001.txt` file under `data/`.

```bash
PYTHONPATH=src python src/load_cmapss.py
```

## Resume summary
Built an end-to-end predictive-maintenance pipeline for multivariate sensor data to estimate remaining useful life (RUL), supporting synthetic degradation data and NASA C-MAPSS equipment data. Engineered rolling sensor statistics and degradation trends and evaluated Random Forest regression with machine-level train/test separation using MAE, RMSE, and R².
