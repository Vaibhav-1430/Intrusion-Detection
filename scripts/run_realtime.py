from __future__ import annotations

import argparse
import _posixsubprocess

import sys
from pathlib import Path

import joblib  

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ids.config import MODELS_DIR, DEFAULT_DATASET
from src.ids.data.datasets import load_dataset
from src.ids.data.preprocess import basic_clean, get_feature_target, extract_packet_like_features
from src.ids.realtime.simulator import stream_packets
from src.ids.realtime.detector import RealTimeDetector


def main(model_path: Path, dataset: str):
	print(f"Loading model: {model_path}")
	estimator = joblib.load(model_path)

	print(f"Loading dataset: {dataset}")
	df, target_col = load_dataset(dataset)
	df = basic_clean(df)
	df = extract_packet_like_features(df)
	X, _ = get_feature_target(df, target_col)

	detector = RealTimeDetector(estimator)
	for batch in stream_packets(X, batch_size=128, interval_sec=0.25):
		alerts = detector.process_batch(batch)
		num_alerts = int((alerts["pred"] == 1).sum())
		print(f"Processed {len(batch)} packets | Alerts: {num_alerts}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=Path, default=MODELS_DIR / "rf_model.joblib")
	parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
	args = parser.parse_args()
	main(args.model, args.dataset)
