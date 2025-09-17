# AI-based Intrusion Detection System (IDS)

This project provides a packet-level IDS using classic ML (Random Forest, XGBoost) and unsupervised anomaly detection (IsolationForest, Autoencoder). It supports benchmark datasets: NSL-KDD, CIC-IDS2017, UNSW-NB15.

## Setup

1. Create a Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place datasets under `data/`:
- `data/NSL-KDD/` → `KDDTrain+.csv` and `KDDTest+.csv` (or `.txt`)
- `data/CIC-IDS2017/` → multiple CSVs
- `data/UNSW-NB15/` → training/testing CSVs

## Training (Supervised)
```bash
python scripts/train_supervised.py --dataset NSL-KDD
```
Artifacts: models under `artifacts/models/`, reports under `artifacts/reports/`.

## Training (Unsupervised)
```bash
python scripts/train_unsupervised.py --dataset NSL-KDD
```
Saves IsolationForest and (optionally) Autoencoder models.

## Realtime Simulation
1. Train a model first, then run:
```bash
python scripts/run_realtime.py --model artifacts/models/rf_model.joblib --dataset NSL-KDD
```
Logs alerts to `artifacts/logs/ids.log`.

## Dashboard
```bash
streamlit run app/dashboard.py
```
Shows recent alerts and counts over time.

## Explainability
Use SHAP and LIME from `src/ids/explain/explain.py` to explain predictions.

## Example Run
```bash
python scripts/example_run.py
```
Trains a small model (if missing) and streams a short demo.

## Notes
- Preprocessing performs missing value handling, normalization, and categorical encoding inside pipelines for supervised models.
- For unsupervised, a simple `get_dummies` encoding is applied.
- You can switch datasets using `--dataset` flag: `NSL-KDD`, `CIC-IDS2017`, `UNSW-NB15`.
