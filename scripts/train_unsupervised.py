from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ids.config import DEFAULT_DATASET, MODELS_DIR, REPORTS_DIR
from src.ids.data.datasets import load_dataset
from src.ids.data.preprocess import basic_clean, get_feature_target
from src.ids.models.unsupervised import train_isolation_forest, train_autoencoder
from src.ids.utils.visualize import plot_anomaly_scores


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
	return pd.get_dummies(df, drop_first=False)


def main(dataset: str, out_dir: Path):
	print(f"Loading dataset: {dataset}")
	df, target_col = load_dataset(dataset)
	df = basic_clean(df)
	X, _ = get_feature_target(df, target_col)

	X_enc = one_hot_encode(X)

	print("Training IsolationForest (unsupervised)...")
	iso_pipe = train_isolation_forest(X_enc)
	out_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(iso_pipe, out_dir / "iso_model.joblib")

	try:
		print("Scoring anomaly scores for visualization...")
		scores = -iso_pipe.decision_function(X_enc)
		fig = plot_anomaly_scores(scores)
		(REPORTS_DIR / "figs").mkdir(parents=True, exist_ok=True)
		fig.savefig(REPORTS_DIR / "figs" / "iso_scores.png", dpi=150)
	except Exception:
		pass

	# Try Autoencoder
	try:
		print("Training Autoencoder (unsupervised)...")
		ae_pipe, info = train_autoencoder(X_enc, encoding_dim=32, epochs=5)
		joblib.dump(ae_pipe, out_dir / "ae_model.joblib")
		(Path(REPORTS_DIR) / "ae_history.json").write_text(json.dumps(info, indent=2))
	except Exception as e:
		print(f"Skipping Autoencoder: {e}")

	print("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
	parser.add_argument("--out_dir", type=Path, default=MODELS_DIR)
	args = parser.parse_args()
	main(args.dataset, args.out_dir)
