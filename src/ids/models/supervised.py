from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

try:
	from xgboost import XGBClassifier  # type: ignore
	_has_xgb = True
except Exception:
	_has_xgb = False

from ..data.preprocess import build_preprocessor


def train_random_forest(X_train, y_train, X_val, y_val, param_grid: Dict[str, Any] | None = None) -> Tuple[Pipeline, Dict[str, Any]]:
	preprocessor, _, _ = build_preprocessor(X_train)
	clf = RandomForestClassifier(random_state=42, n_jobs=-1)
	pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

	if param_grid is None:
		param_grid = {
			"clf__n_estimators": [200, 400],
			"clf__max_depth": [None, 20],
			"clf__min_samples_split": [2, 5],
		}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
	grid.fit(X_train, y_train)

	y_pred = grid.best_estimator_.predict(X_val)
	report = classification_report(y_val, y_pred, output_dict=True)

	proba = _safe_predict_proba(grid.best_estimator_, X_val)
	auc = roc_auc_score(y_val, proba[:, 1], multi_class="ovr") if proba.shape[1] > 1 else np.nan
	return grid.best_estimator_, {"report": report, "roc_auc": auc, "best_params": grid.best_params_}


def train_xgboost(X_train, y_train, X_val, y_val, param_grid: Dict[str, Any] | None = None) -> Tuple[Pipeline, Dict[str, Any]]:
	if not _has_xgb:
		raise RuntimeError("xgboost not available")

	preprocessor, _, _ = build_preprocessor(X_train)
	clf = XGBClassifier(
		eval_metric="logloss",
		tree_method="hist",
		random_state=42,
	)
	pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

	if param_grid is None:
		param_grid = {
			"clf__n_estimators": [200, 400],
			"clf__max_depth": [6, 10],
			"clf__learning_rate": [0.05, 0.1],
		}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
	grid.fit(X_train, y_train)

	y_pred = grid.best_estimator_.predict(X_val)
	report = classification_report(y_val, y_pred, output_dict=True)

	proba = _safe_predict_proba(grid.best_estimator_, X_val)
	auc = roc_auc_score(y_val, proba[:, 1], multi_class="ovr") if proba.shape[1] > 1 else np.nan
	return grid.best_estimator_, {"report": report, "roc_auc": auc, "best_params": grid.best_params_}


def _safe_predict_proba(model: Pipeline, X):
	try:
		return model.predict_proba(X)
	except Exception:
		# Fallback for models without predict_proba in some settings
		pred = model.predict(X)
		classes = np.unique(pred)
		proba = np.zeros((len(pred), len(classes)))
		for i, c in enumerate(classes):
			proba[:, i] = (pred == c).astype(float)
		return proba
