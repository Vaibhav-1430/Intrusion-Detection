from __future__ import annotations

import time
from typing import Iterator, Dict

import pandas as pd

from ..config import STREAM_BATCH_SIZE, STREAM_INTERVAL_SEC


def stream_packets(df: pd.DataFrame, batch_size: int = STREAM_BATCH_SIZE, interval_sec: float = STREAM_INTERVAL_SEC) -> Iterator[pd.DataFrame]:
	"""Yield successive batches of packet-like rows to simulate realtime traffic."""
	start = 0
	while start < len(df):
		end = min(start + batch_size, len(df))
		yield df.iloc[start:end]
		start = end
		time.sleep(interval_sec)
