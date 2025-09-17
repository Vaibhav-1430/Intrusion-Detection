from __future__ import annotations

import sys
from pathlib import Path
import itertools

import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ids.config import MODELS_DIR, DEFAULT_DATASET
from src.ids.data.datasets import load_dataset
from src.ids.data.preprocess import basic_clean, get_feature_target, split_train_val_test
from src.ids.models.supervised import train_random_forest
from src.ids.realtime.detector import RealTimeDetector


def ensure_model(model_path: Path, dataset: str):
	if model_path.exists():
		return joblib.load(model_path)
	print("Training quick RandomForest model for demo...")
	df, target_col = load_dataset(dataset)
	df = basic_clean(df)
	X, y = get_feature_target(df, target_col)
	X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
	model, _ = train_random_forest(X_train, y_train, X_val, y_val, param_grid={"clf__n_estimators": [150], "clf__max_depth": [None]})
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, model_path)
	return model


def main():
	print("Starting IDS example run...")
	model_path = MODELS_DIR / "rf_model.joblib"
	print(f"Model path: {model_path}")
	
	estimator = ensure_model(model_path, DEFAULT_DATASET)
	print("Model loaded/created successfully")

	# Take a small sample for streaming demo
	print("Loading dataset...")
	df, target_col = load_dataset(DEFAULT_DATASET)
	print(f"Dataset shape: {df.shape}, target column: {target_col}")
	
	df = basic_clean(df)
	X, _ = get_feature_target(df, target_col)
	print(f"Features shape: {X.shape}")
	
	X_sample = X.sample(n=min(1024, len(X)), random_state=42)
	print(f"Sample shape: {X_sample.shape}")

	detector = RealTimeDetector(estimator)
	batch_size = 128
	print(f"Starting realtime simulation with batch size {batch_size}...")
	
	for i in range(0, len(X_sample), batch_size):
		batch = X_sample.iloc[i:i+batch_size]
		alerts = detector.process_batch(batch)
		print(f"Batch {i//batch_size}: {int((alerts['pred']==1).sum())} alerts")
	
	print("Example run completed!")


if __name__ == "__main__":
	main()
