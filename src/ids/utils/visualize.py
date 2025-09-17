from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def plot_distribution(values, title: str, bins: int = 50):
	plt.figure(figsize=(6, 4))
	plt.hist(values, bins=bins, color="#4e79a7", alpha=0.8)
	plt.title(title)
	plt.tight_layout()
	return plt.gcf()


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[list] = None, title: str = "Confusion Matrix"):
	plt.figure(figsize=(5, 4))
	plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()
	classes = class_names or ["Normal", "Attack"]
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cm.max() / 2.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.tight_layout()
	return plt.gcf()


def plot_roc_curve(estimator, X_test, y_test, title: str = "ROC Curve"):
	fig, ax = plt.subplots(figsize=(5, 4))
	RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
	ax.set_title(title)
	fig.tight_layout()
	return fig


def plot_anomaly_scores(scores, title: str = "Anomaly Scores"):
	return plot_distribution(scores, title=title, bins=60)
