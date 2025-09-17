from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from ..config import DATA_DIR, TARGET_COLUMN_MAP


def load_dataset(name: str, path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
	"""Load a supported IDS dataset and return (dataframe, target_column_name).

	Args:
		name: One of {"NSL-KDD", "CIC-IDS2017", "UNSW-NB15"}.
		path: Optional explicit path to the dataset directory or file.

	Returns:
		(df, target_col)
	"""
	name = name.upper().replace("_", "-")
	if name not in {"NSL-KDD", "CIC-IDS2017", "UNSW-NB15"}:
		raise ValueError(f"Unsupported dataset: {name}")

	if path is None:
		path = DATA_DIR / name

	if name == "NSL-KDD":
		return _load_nsl_kdd(path), TARGET_COLUMN_MAP["NSL-KDD"]
	elif name == "CIC-IDS2017":
		return _load_cic_ids2017(path), TARGET_COLUMN_MAP["CIC-IDS2017"]
	else:
		return _load_unsw_nb15(path), TARGET_COLUMN_MAP["UNSW-NB15"]


def _load_nsl_kdd(path: Path) -> pd.DataFrame:
	"""Load NSL-KDD combined as a single DataFrame.
	Expected files: KDDTrain+.txt, KDDTest+.txt (CSV-like)."""
	path.mkdir(parents=True, exist_ok=True)
	train = _smart_read(path / "KDDTrain+.csv") if (path / "KDDTrain+.csv").exists() else _smart_read(path / "KDDTrain+.txt")
	test = _smart_read(path / "KDDTest+.csv") if (path / "KDDTest+.csv").exists() else _smart_read(path / "KDDTest+.txt")
	if train is None and test is None:
		raise FileNotFoundError(
			"NSL-KDD files not found. Place KDDTrain+/KDDTest+ under data/NSL-KDD."
		)
	frames = [df for df in [train, test] if df is not None]
	return pd.concat(frames, ignore_index=True)


def _load_cic_ids2017(path: Path) -> pd.DataFrame:
	"""Load CIC-IDS2017 aggregated CSVs (e.g., Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv)."""
	path.mkdir(parents=True, exist_ok=True)
	csvs = list(path.glob("*.csv"))
	if not csvs:
		raise FileNotFoundError("CIC-IDS2017 CSVs not found in data/CIC-IDS2017")
	dfs = [pd.read_csv(p, low_memory=False) for p in csvs]
	return pd.concat(dfs, ignore_index=True)


def _load_unsw_nb15(path: Path) -> pd.DataFrame:
	"""Load UNSW-NB15 CSVs (e.g., UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv)."""
	path.mkdir(parents=True, exist_ok=True)
	csvs = list(path.glob("*.csv"))
	if not csvs:
		raise FileNotFoundError("UNSW-NB15 CSVs not found in data/UNSW-NB15")
	dfs = [pd.read_csv(p, low_memory=False) for p in csvs]
	return pd.concat(dfs, ignore_index=True)


def _smart_read(p: Path) -> Optional[pd.DataFrame]:
	if not p.exists():
		return None
	if p.suffix.lower() in {".csv", ".txt"}:
		try:
			return pd.read_csv(p)
		except Exception:
			return pd.read_csv(p, header=None)
	elif p.suffix.lower() in {".parquet"}:
		return pd.read_parquet(p)
	else:
		raise ValueError(f"Unsupported file format: {p}")
