from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
	if y_proba is not None:
		try:
			auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba)
			metrics["roc_auc"] = auc
		except Exception:
			pass
	return metrics


def compute_confusion(y_true, y_pred) -> np.ndarray:
	return confusion_matrix(y_true, y_pred)
