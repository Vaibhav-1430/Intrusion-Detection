from __future__ import annotations

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Ensure directories exist at import time
for _p in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
	_p.mkdir(parents=True, exist_ok=True)

# Dataset options
DEFAULT_DATASET: str = "NSL-KDD"  # or "CIC-IDS2017", "UNSW-NB15"
TARGET_COLUMN_MAP = {
	"NSL-KDD": "label",
	"CIC-IDS2017": "Label",
	"UNSW-NB15": "label",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Realtime simulation
STREAM_BATCH_SIZE = 128
STREAM_INTERVAL_SEC = 0.25

# Logging
LOG_FILE = LOGS_DIR / "ids.log"
