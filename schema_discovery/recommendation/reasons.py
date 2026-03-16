from __future__ import annotations

# Keep these stable. Do not scatter string literals across the codebase.

# ---- PK reasons
PK_UNIQUE_HIGH = "pk_unique_high"
PK_NULL_LOW = "pk_null_low"
PK_DTYPE_NUMERIC = "pk_dtype_numeric"
PK_REFERENCED_BY_FKS = "pk_referenced_by_fks"

# ---- Composite reasons
COMPOSITE_UNIQUE_HIGH = "composite_unique_high"

# ---- FK reasons
FK_SCORE_HIGH = "fk_score_high"
FK_INCLUSION_HIGH = "fk_inclusion_high"
FK_MISSING_PARENT_LOW = "fk_missing_parent_low"
FK_TYPE_COMPATIBLE = "fk_type_compatible"

# ---- Datatype reasons
DTYPE_UUID_REGEX = "dtype_uuid_regex"
DTYPE_DATE_PARSE_HIGH = "dtype_date_parse_high"
DTYPE_BOOL_PATTERN = "dtype_bool_pattern"
DTYPE_INT_RANGE_SMALL = "dtype_int_range_small"
DTYPE_INT_RANGE_BIG = "dtype_int_range_big"
DTYPE_DECIMAL_LIKE = "dtype_decimal_like"
DTYPE_TEXT_LONG = "dtype_text_long"
DTYPE_VARCHAR_FIT = "dtype_varchar_length_fit"
DTYPE_INT_ID_HEURISTIC = "dtype_int_id_heuristic"
DTYPE_INT_NO_RANGE_DEFAULT = "dtype_int_no_range_default"


# ---- Index reasons
IDX_PRIMARY_KEY = "idx_primary_key"
IDX_FOREIGN_KEY = "idx_foreign_key"
IDX_COMPOSITE_KEY = "idx_composite_key"
IDX_LOW_CARDINALITY_AVOID = "idx_low_cardinality_avoid"
IDX_TEXT_AVOID = "idx_text_avoid"

# ---- Blocker codes (stable)
BLK_NOT_NULL_NULL_RATIO_HIGH = "not_null_null_ratio_high"
BLK_UNIQUE_DUPLICATES_PRESENT = "unique_duplicates_present"
BLK_FK_MISSING_PARENT_HIGH = "fk_missing_parent_high"
BLK_INT_OVERFLOW_RISK = "int_overflow_risk"


BLK_INSUFFICIENT_EVIDENCE = "insufficient_evidence"