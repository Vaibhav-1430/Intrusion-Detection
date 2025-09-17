from __future__ import annotations

from typing import Optional

import numpy as np

try:
	import shap
	_has_shap = True
except Exception:
	_has_shap = False

try:
	from lime.lime_tabular import LimeTabularExplainer
	_has_lime = True
except Exception:
	_has_lime = False


def shap_explain_pipeline(pipeline, X_sample, max_display: int = 10):
	if not _has_shap:
		raise RuntimeError("SHAP not available")
	# Try to get inner model
	model = getattr(pipeline, "named_steps", {}).get("clf", pipeline)
	explainer = None
	try:
		explainer = shap.TreeExplainer(model)
	except Exception:
		explainer = shap.Explainer(model.predict, X_sample)
	shap_values = explainer(X_sample)
	return shap_values


def lime_explain_instance(pipeline, X_train, feature_names, class_names, row_idx: int = 0):
	if not _has_lime:
		raise RuntimeError("LIME not available")
	explainer = LimeTabularExplainer(
		X_train,
		feature_names=feature_names,
		class_names=class_names,
		discretize_continuous=True,
		mode="classification",
	)
	instance = X_train[row_idx]
	exp = explainer.explain_instance(instance, pipeline.predict_proba, num_features=10)
	return exp
