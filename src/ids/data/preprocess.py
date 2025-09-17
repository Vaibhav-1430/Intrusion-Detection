from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
	# Drop duplicate rows and strip whitespace in column names
	df = df.drop_duplicates().copy()
	df.columns = [str(c).strip() for c in df.columns]
	# Replace inf/nan
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
	return df


def get_feature_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
	if target_col not in df.columns:
		raise KeyError(f"Target column {target_col} not found")
	y = df[target_col]
	X = df.drop(columns=[target_col])
	return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
	numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
	categorical_cols = [c for c in X.columns if c not in numeric_cols]

	numeric_transformer = Pipeline(steps=[
		("scaler", StandardScaler(with_mean=False)),
	])

	categorical_transformer = Pipeline(steps=[
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_cols),
			("cat", categorical_transformer, categorical_cols),
		]
	)
	return preprocessor, numeric_cols, categorical_cols


def split_train_val_test(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, val_size: float = VAL_SIZE) -> Tuple:
	X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None)
	val_ratio = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp if y_temp.nunique() > 1 else None)
	return X_train, X_val, X_test, y_train, y_val, y_test


def extract_packet_like_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure key packet-level columns exist, casting types where possible.
	This function is dataset-agnostic; it maps common names when present.
	"""
	candidate_mappings: Dict[str, List[str]] = {
		"src_ip": ["src_ip", "Source IP", "srcip", "Src IP", "Source"],
		"dst_ip": ["dst_ip", "Destination IP", "dstip", "Dst IP", "Destination"],
		"src_port": ["src_port", "sport", "Src Port"],
		"dst_port": ["dst_port", "dport", "Dst Port"],
		"protocol": ["protocol", "Protocol", "proto"],
		"packet_size": ["packet_size", "TotLen Fwd Pkts", "pkt_size", "bytes", "iplen"],
		"flags": ["flags", "Flags", "tcp.flags"],
		"timestamp": ["timestamp", "Timestamp", "flowStartTime", "time"],
	}

	result = df.copy()
	for canonical, options in candidate_mappings.items():
		if canonical in result.columns:
			continue
		for opt in options:
			if opt in result.columns:
				result[canonical] = result[opt]
				break
		if canonical not in result.columns:
			# Create placeholder if missing
			result[canonical] = np.nan

	# Cast numeric ports and sizes
	for col in ["src_port", "dst_port", "packet_size"]:
		with np.errstate(all="ignore"):
			result[col] = pd.to_numeric(result[col], errors="coerce")

	return result
