from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
	import tensorflow as tf
	from tensorflow import keras
	_has_tf = True
except Exception:
	_has_tf = False


def train_isolation_forest(X_train) -> Pipeline:
	scaler = StandardScaler(with_mean=False)
	iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
	pipe = Pipeline([("scaler", scaler), ("iso", iso)])
	pipe.fit(X_train)
	return pipe


def train_autoencoder(X_train, encoding_dim: int = 32, epochs: int = 10, batch_size: int = 256) -> Tuple[Pipeline, Dict]:
	if not _has_tf:
		raise RuntimeError("TensorFlow not available for Autoencoder")

	# Ensure dense input
	X_train_dense = X_train.values if hasattr(X_train, "values") else X_train

	input_dim = X_train_dense.shape[1]
	inputs = keras.Input(shape=(input_dim,))
	h = keras.layers.Dense(encoding_dim, activation="relu")(inputs)
	z = keras.layers.Dense(input_dim, activation="linear")(h)
	model = keras.Model(inputs, z)
	model.compile(optimizer="adam", loss="mse")
	history = model.fit(X_train_dense, X_train_dense, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

	class AutoencoderWrapper:
		def __init__(self, model):
			self.model = model
		def fit(self, X):
			return self
		def decision_function(self, X):
			Xd = X.values if hasattr(X, "values") else X
			recon = self.model.predict(Xd, verbose=0)
			return np.mean((Xd - recon) ** 2, axis=1)
		def predict(self, X):
			# Outlier = -1, Inlier = 1
			scores = self.decision_function(X)
			threshold = np.percentile(scores, 95)
			return np.where(scores > threshold, -1, 1)

	pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("ae", AutoencoderWrapper(model))])
	pipe.fit(X_train_dense)
	return pipe, {"history": history.history}
