from __future__ import annotations

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from ..types import DType

# Suppress noisy, non-actionable per-token pandas date parsing warnings, while leaving other
# user warnings visible. We narrow by message substring to avoid hiding unrelated issues.
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module="pandas",
	message=r"Could not infer format, so each element will be parsed individually",
)

# Canonical truthy/falsey tokens (lowercased)
BOOLEAN_TRUE_VALUES = {"true", "t", "yes", "y", "1", "on"}
BOOLEAN_FALSE_VALUES = {"false", "f", "no", "n", "0", "off"}
BOOLEAN_CANONICAL_MAP: Dict[str, bool] = {
	**{v: True for v in BOOLEAN_TRUE_VALUES},
	**{v: False for v in BOOLEAN_FALSE_VALUES},
}

# Missing tokens (lowercased). Empty strings and common Na-like strings are treated missing
MISSING_TOKENS = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    None,
}

# Regex helpers for numeric token normalization
NUMERIC_STRIP_RE = re.compile(r"[,_\s\u00a0]")
CURRENCY_PREFIX_RE = re.compile(r"^[\$£€¥₹]")
NEGATIVE_PARENS_RE = re.compile(r"^\(.*\)$")


class AtomicDType(Enum):
	STRING = "string"
	BOOL = "bool"
	INT = "int"
	FLOAT = "float"
	DATE = "date"


def atomic_to_dtype(at: "AtomicDType") -> DType:
	"""Map internal AtomicDType to engine-level DType.

	- STRING -> STR
	- BOOL -> BOOL
	- DATE -> DATE
	- INT/FLOAT -> NUM (engine treats numeric as one logical type)
	"""
	if at is AtomicDType.STRING:
		return DType.STR
	if at is AtomicDType.BOOL:
		return DType.BOOL
	if at is AtomicDType.DATE:
		return DType.DATE
	# Collapse numeric variants to NUM in engine typing
	if at in (AtomicDType.INT, AtomicDType.FLOAT):
		return DType.NUM
	# Fallback
	return DType.STR

@dataclass(frozen=True)
class InferenceConfig:
	# Sampling
	sample_ratio: float = 0.02
	min_sample_size: int = 1_000
	max_sample_size: int = 25_000
	seed: int = 42

	# Thresholds
	bool_threshold: float = 0.95
	numeric_threshold: float = 0.98
	datetime_threshold: float = 0.90
	integer_tolerance: float = 1e-9

	# Datetime parsing
	dayfirst: bool = False
	yearfirst: bool = False

	# Performance features
	unique_weighted_inference: bool = True
	profile: bool = False


@dataclass(frozen=True)
class ColumnInference:
	atomic_type: AtomicDType
	confidence: float
	convertible_ratio: float
	reason: str
	sample_size: int


def _maybe_sample(series: pd.Series, cfg: InferenceConfig) -> pd.Series:
	n = len(series)
	if n == 0:
		return series
	target = min(cfg.max_sample_size, max(cfg.min_sample_size, int(math.ceil(cfg.sample_ratio * n))))
	if n <= target:
		return series
	return series.sample(n=target, random_state=cfg.seed)


def _sanitize_tokens(series: pd.Series) -> pd.Series:
	tok = series.astype("string").str.strip()
	tok = tok.dropna()
	tok = tok[tok.str.len() > 0]
	low = tok.str.lower()
	return low[~low.isin(MISSING_TOKENS)]


def _normalize_numeric_tokens(tokens: pd.Series) -> Tuple[pd.Series, pd.Series]:
	numeric = tokens.astype("string")
	numeric = numeric.str.replace(CURRENCY_PREFIX_RE, "", regex=True)
	percent_mask = numeric.str.endswith("%")
	if percent_mask.any():
		numeric.loc[percent_mask] = numeric.loc[percent_mask].str[:-1]
	paren_mask = numeric.str.match(NEGATIVE_PARENS_RE)
	if paren_mask.any():
		numeric.loc[paren_mask] = "-" + numeric.loc[paren_mask].str[1:-1]
	numeric = numeric.str.replace(NUMERIC_STRIP_RE, "", regex=True)
	numeric = numeric.str.replace("−", "-", regex=False)
	return numeric.astype("string"), percent_mask


def _infer_series(series: pd.Series, cfg: InferenceConfig) -> ColumnInference:
	# Short-circuit for stable dtypes
	if pd.api.types.is_bool_dtype(series):
		return ColumnInference(AtomicDType.BOOL, 1.0, 1.0, "Already boolean dtype", len(series.dropna()))
	if pd.api.types.is_integer_dtype(series):
		return ColumnInference(AtomicDType.INT, 1.0, 1.0, "Already integer dtype", len(series.dropna()))
	if pd.api.types.is_datetime64_any_dtype(series):
		return ColumnInference(AtomicDType.DATE, 1.0, 1.0, "Already datetime dtype", len(series.dropna()))
	if pd.api.types.is_float_dtype(series):
		# Check if actually integer-valued floats
		non_null = series.dropna()
		if non_null.empty:
			return ColumnInference(AtomicDType.STRING, 0.0, 0.0, "Only null values", 0)
		numeric = pd.to_numeric(non_null, errors="coerce")
		if numeric.notna().all() and np.isclose(numeric.to_numpy(), np.round(numeric.to_numpy()), atol=cfg.integer_tolerance).all():
			return ColumnInference(AtomicDType.INT, 0.99, 1.0, "Float column stores integer-compatible values", len(non_null))
		return ColumnInference(AtomicDType.FLOAT, 1.0, 1.0, "Already float dtype", len(non_null))

	non_null = series.dropna()
	if non_null.empty:
		return ColumnInference(AtomicDType.STRING, 0.0, 0.0, "Only null values", 0)

	sampled = _maybe_sample(non_null, cfg)
	tokens = _sanitize_tokens(sampled)
	sample_size = int(len(tokens))
	if tokens.empty:
		return ColumnInference(AtomicDType.STRING, 0.0, 0.0, "Only null/empty after sanitization", sample_size)

	if cfg.unique_weighted_inference:
		# Compute on unique tokens with frequency weights to avoid redundant work while preserving distribution
		value_counts = tokens.value_counts(dropna=False)
		unique_tokens = pd.Series(value_counts.index.astype("string"))
		weights = value_counts.to_numpy()
		total_weight = float(weights.sum()) if weights.size else 0.0

		# Boolean ratio (weighted)
		mapped_bool = unique_tokens.map(BOOLEAN_CANONICAL_MAP)
		bool_hits = mapped_bool.notna().to_numpy(dtype=float)
		bool_ratio = float(np.dot(bool_hits, weights) / total_weight) if total_weight > 0 else 0.0

		# Datetime ratio (weighted) with warning suppression for ambiguous formats
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				category=UserWarning,
				message=r"Could not infer format, so each element will be parsed individually.*",
			)
			parsed_dt = pd.to_datetime(unique_tokens, errors="coerce", dayfirst=cfg.dayfirst, yearfirst=cfg.yearfirst)
		dt_hits = parsed_dt.notna().to_numpy(dtype=float)
		datetime_ratio = float(np.dot(dt_hits, weights) / total_weight) if total_weight > 0 else 0.0

		# Numeric analysis (weighted)
		normalized, percent_mask = _normalize_numeric_tokens(unique_tokens)
		numeric_f = pd.to_numeric(normalized, errors="coerce")
		if percent_mask.any():
			numeric_f.loc[percent_mask] = numeric_f.loc[percent_mask] / 100.0
		float_mask = numeric_f.notna().to_numpy()
		float_hits = float_mask.astype(float)
		float_ratio = float(np.dot(float_hits, weights) / total_weight) if total_weight > 0 else 0.0
		# Among numeric successes, check if values are integer-like (weighted)
		int_ratio = 0.0
		if float_mask.any():
			nn_vals = numeric_f[float_mask].to_numpy()
			nn_weights = weights[float_mask]
			int_like_hits = np.isclose(nn_vals, np.round(nn_vals), atol=cfg.integer_tolerance).astype(float)
			nn_total = float(nn_weights.sum())
			if nn_total > 0:
				int_ratio = float(np.dot(int_like_hits, nn_weights) / nn_total)
	else:
		# Compute directly on tokens without deduplicating
		mapped_bool = tokens.map(BOOLEAN_CANONICAL_MAP)
		bool_ratio = float(mapped_bool.notna().mean()) if not mapped_bool.empty else 0.0

		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				category=UserWarning,
				message=r"Could not infer format, so each element will be parsed individually.*",
			)
			parsed_dt = pd.to_datetime(tokens, errors="coerce", dayfirst=cfg.dayfirst, yearfirst=cfg.yearfirst)
		datetime_ratio = float(parsed_dt.notna().mean()) if not parsed_dt.empty else 0.0

		normalized, percent_mask = _normalize_numeric_tokens(tokens)
		numeric_f = pd.to_numeric(normalized, errors="coerce")
		if percent_mask.any():
			numeric_f.loc[percent_mask] = numeric_f.loc[percent_mask] / 100.0
		float_ratio = float(numeric_f.notna().mean()) if not numeric_f.empty else 0.0
		int_like = numeric_f.dropna()
		int_ratio = 0.0
		if not int_like.empty:
			int_ratio = float(np.isclose(int_like.to_numpy(), np.round(int_like.to_numpy()), atol=cfg.integer_tolerance).mean())

	# Decision logic: prefer bool, then date, then int, then float
	# This order avoids misclassifying date-like numbers and rewards clear signals
	if bool_ratio >= cfg.bool_threshold:
		chosen = ColumnInference(AtomicDType.BOOL, bool_ratio, bool_ratio, "Boolean tokens dominate", sample_size)
		return chosen

	if datetime_ratio >= cfg.datetime_threshold:
		chosen = ColumnInference(AtomicDType.DATE, datetime_ratio, datetime_ratio, "Datetime parsing succeeded for majority", sample_size)
		return chosen

	# Classify as INT only if overall numeric coverage is high AND values are integer-like
	if float_ratio >= cfg.numeric_threshold and int_ratio >= cfg.numeric_threshold:
		# Strong numeric signal; integer-like prevails when both high
		chosen = ColumnInference(AtomicDType.INT, int_ratio, float_ratio, "Numeric tokens with integer-like distribution", sample_size)
		return chosen

	# Do not classify as INT if numeric coverage is weak; avoid false positives on mixed columns

	if float_ratio >= cfg.numeric_threshold:
		chosen = ColumnInference(AtomicDType.FLOAT, float_ratio, float_ratio, "Float tokens detected with high confidence", sample_size)
		return chosen

	# Fallback to string
	confidence = max(0.0, 1.0 - max(bool_ratio, datetime_ratio, float_ratio))
	return ColumnInference(AtomicDType.STRING, confidence, 0.0, "Fallback to string due to mixed/low confidence", sample_size)


def _convert_series(
	series: pd.Series,
	inf: ColumnInference,
	cfg: InferenceConfig,
	*,
	numeric_bool: bool = False,
	numeric_date: bool = False,
) -> pd.Series:
	at = inf.atomic_type
	if at is AtomicDType.STRING:
		s = series.astype("string")
		# Standardize missing to <NA>
		s = s.where(~s.isin([None, ""]))
		return s

	if at is AtomicDType.BOOL:
		lowered = series.astype("string").str.strip().str.lower()
		mapped = lowered.map(BOOLEAN_CANONICAL_MAP)
		if numeric_bool:
			return mapped.astype("Int8")  # 1/0/<NA>
		return mapped.astype("boolean")  # pandas nullable boolean, missing -> <NA>

	if at is AtomicDType.DATE:
		parsed = pd.to_datetime(series, errors="coerce", dayfirst=cfg.dayfirst, yearfirst=cfg.yearfirst)
		if numeric_date:
			# Vectorized epoch seconds as float64 with NaN for missing
			arr = parsed.to_numpy(dtype="int64", copy=False)
			mask = parsed.notna().to_numpy()
			out = np.where(mask, arr / 1e9, np.nan)
			return pd.Series(out, index=series.index, dtype="float64")
		return parsed  # missing -> NaT

	if at in (AtomicDType.INT, AtomicDType.FLOAT):
		# Normalize numeric then to float; int may downcast later if safe and no missing
		tokens = series.astype("string").str.strip()
		norm, percent_mask = _normalize_numeric_tokens(tokens)
		numeric = pd.to_numeric(norm, errors="coerce")
		if percent_mask.any():
			numeric.loc[percent_mask] = numeric.loc[percent_mask] / 100.0
		# Missing/failed -> np.nan
		numeric = numeric.astype("float64")
		if at is AtomicDType.INT:
			# If no missing and integer-like, cast to int64
			nn = numeric.dropna()
			if not nn.empty and np.isclose(nn.to_numpy(), np.round(nn.to_numpy()), atol=cfg.integer_tolerance).all() and numeric.notna().all():
				try:
					return nn.round().astype("int64").reindex(numeric.index)
				except Exception:
					# Fallback to float with nan
					pass
		return numeric

	# Should not reach here
	return series.astype("string")


class AtomicDtypeModelV2:
	"""A simple, robust atomic dtype inference and conversion model (v2).

	- Takes a pandas DataFrame
	- Infers column atomic types: int, float, string, date, bool
	- Converts columns when requested, standardizing missing values per type
	- Uses sampling for scalability
	"""

	def __init__(self, config: Optional[InferenceConfig] = None):
		self.config = config or InferenceConfig()
		self._schema: Dict[str, ColumnInference] = {}
		self._fitted = False
		self._profile = {"fit": {}, "transform": {}}

	def fit(self, data: pd.DataFrame) -> "AtomicDtypeModelV2":
		cfg = self.config
		results: Dict[str, ColumnInference] = {}
		for col in data.columns:
			try:
				if cfg.profile:
					t0 = time.perf_counter()
					res = _infer_series(data[col], cfg)
					dt_ms = (time.perf_counter() - t0) * 1_000
					self._profile["fit"][str(col)] = round(dt_ms, 2)
					results[str(col)] = res
				else:
					results[str(col)] = _infer_series(data[col], cfg)
			except Exception as e:
				# Defensive: fallback to string with low confidence
				results[str(col)] = ColumnInference(AtomicDType.STRING, 0.0, 0.0, f"Inference error: {e}", 0)
		self._schema = results
		self._fitted = True
		return self

	def transform(
		self,
		data: pd.DataFrame,
		*,
		convert: bool = True,
		numeric_bool: bool = False,
		numeric_date: bool = False,
	) -> pd.DataFrame:
		if not self._fitted:
			raise RuntimeError("Model must be fitted before transform().")
		if not convert:
			return data.copy()
		cfg = self.config
		out = data.copy()
		start_total = time.perf_counter() if self.config.profile else None
		for col, inf in self._schema.items():
			if col not in out.columns:
				continue
			# Convert if pandas dtype is not already stable int/bool/date OR differs vs inferred
			s = out[col]
			stable_dtype = pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
			pandas_type_matches = (
				(inf.atomic_type is AtomicDType.BOOL and pd.api.types.is_bool_dtype(s))
				or (inf.atomic_type is AtomicDType.INT and pd.api.types.is_integer_dtype(s))
				or (inf.atomic_type is AtomicDType.DATE and pd.api.types.is_datetime64_any_dtype(s))
				or (inf.atomic_type is AtomicDType.FLOAT and pd.api.types.is_float_dtype(s))
				or (inf.atomic_type is AtomicDType.STRING and pd.api.types.is_string_dtype(s))
			)
			if (not stable_dtype) or (not pandas_type_matches):
				try:
					if self.config.profile:
						t0 = time.perf_counter()
						out[col] = _convert_series(
							s,
							inf,
							cfg,
							numeric_bool=numeric_bool,
							numeric_date=numeric_date,
						)
						dt_ms = (time.perf_counter() - t0) * 1_000
						self._profile["transform"][str(col)] = round(dt_ms, 2)
					else:
						out[col] = _convert_series(
							s,
							inf,
							cfg,
							numeric_bool=numeric_bool,
							numeric_date=numeric_date,
						)
				except Exception:
					# Last-resort: string with missing as <NA>
					out[col] = s.astype("string")
		if self.config.profile and start_total is not None:
			self._profile["transform"]["total_ms"] = round((time.perf_counter() - start_total) * 1_000, 2)
		return out

	def fit_transform(
		self,
		data: pd.DataFrame,
		*,
		convert: bool = True,
		numeric_bool: bool = False,
		numeric_date: bool = False,
	) -> pd.DataFrame:
		self.fit(data)
		return self.transform(
			data,
			convert=convert,
			numeric_bool=numeric_bool,
			numeric_date=numeric_date,
		)

	def schema(self) -> Mapping[str, ColumnInference]:
		if not self._fitted:
			raise RuntimeError("Model has not been fitted yet.")
		return dict(self._schema)

	# --- Schema export helpers ---
	def schema_to_json(self, include_config: bool = True) -> str:
		import json

		payload: Dict[str, Any] = {
			"columns": {
				name: {
					"atomic_type": inf.atomic_type.value,
					"confidence": float(inf.confidence),
					"convertible_ratio": float(inf.convertible_ratio),
					"reason": inf.reason,
					"sample_size": int(inf.sample_size),
				}
				for name, inf in self._schema.items()
			}
		}
		if include_config:
			payload["config"] = asdict(self.config)
		return json.dumps(payload, indent=2)

	def save_schema(self, path: str, include_config: bool = True) -> None:
		json_text = self.schema_to_json(include_config=include_config)
		with open(path, "w", encoding="utf-8") as f:
			f.write(json_text)

	def timings(self) -> Mapping[str, Any]:
		"""Return profiling timings collected during fit/transform (ms)."""
		return {
			"fit": dict(self._profile.get("fit", {})),
			"transform": dict(self._profile.get("transform", {})),
		}

