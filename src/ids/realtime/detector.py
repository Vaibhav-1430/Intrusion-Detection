from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import LOG_FILE


class RealTimeDetector:
	def __init__(self, estimator, alert_threshold: float = 0.5, log_file: Path = LOG_FILE):
		self.estimator = estimator
		self.alert_threshold = alert_threshold
		self.log_file = Path(log_file)
		self.log_file.parent.mkdir(parents=True, exist_ok=True)

	def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
		proba = None
		try:
			proba = self.estimator.predict_proba(batch)
			y_pred = (proba[:, 1] >= self.alert_threshold).astype(int)
		except Exception:
			# Fallback for estimators without predict_proba
			y_pred = self.estimator.predict(batch)

		alerts = batch.copy()
		alerts["pred"] = y_pred
		if proba is not None:
			alerts["score"] = proba[:, 1]
		else:
			alerts["score"] = -1.0
		flagged = alerts[alerts["pred"] == 1]
		if not flagged.empty:
			self._log_alerts(flagged)
		return alerts

	def _log_alerts(self, df: pd.DataFrame):
		with self.log_file.open("a", encoding="utf-8") as f:
			for _, row in df.iterrows():
				timestamp = datetime.utcnow().isoformat()
				f.write(f"{timestamp}\tALERT\tpred={row['pred']}\tscore={row['score']}\n")
