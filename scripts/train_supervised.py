from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ids.config import DEFAULT_DATASET, MODELS_DIR, REPORTS_DIR
from src.ids.data.datasets import load_dataset
from src.ids.data.preprocess import basic_clean, get_feature_target, split_train_val_test
from src.ids.models.supervised import train_random_forest, train_xgboost
from src.ids.utils.metrics import compute_classification_metrics, compute_confusion


def main(dataset: str, out_dir: Path):
	print(f"Loading dataset: {dataset}")
	df, target_col = load_dataset(dataset)
	df = basic_clean(df)
	X, y = get_feature_target(df, target_col)
	X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

	print("Training RandomForest...")
	rf_model, rf_info = train_random_forest(X_train, y_train, X_val, y_val)

	print("Evaluating RandomForest...")
	y_pred = rf_model.predict(X_test)
	try:
		y_proba = rf_model.predict_proba(X_test)
	except Exception:
		y_proba = None
	rf_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
	cm = compute_confusion(y_test, y_pred).tolist()

	out_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(rf_model, out_dir / "rf_model.joblib")
	(Path(REPORTS_DIR) / "rf_metrics.json").write_text(json.dumps({"metrics": rf_metrics, "grid": rf_info}, indent=2))

	# Try XGBoost if installed
	try:
		print("Training XGBoost...")
		xgb_model, xgb_info = train_xgboost(X_train, y_train, X_val, y_val)
		y_pred = xgb_model.predict(X_test)
		try:
			y_proba = xgb_model.predict_proba(X_test)
		except Exception:
			y_proba = None
		xgb_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
		joblib.dump(xgb_model, out_dir / "xgb_model.joblib")
		(Path(REPORTS_DIR) / "xgb_metrics.json").write_text(json.dumps({"metrics": xgb_metrics, "grid": xgb_info}, indent=2))
	except Exception as e:
		print(f"Skipping XGBoost: {e}")

	print("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
	parser.add_argument("--out_dir", type=Path, default=MODELS_DIR)
	args = parser.parse_args()
	main(args.dataset, args.out_dir)
