# Recommendation Layer Design

---

## Inputs

* Column profiles (null ratio, uniqueness, ranges, dtype, length stats)
* Key candidates (unary, composite, surrogate signals)
* FK candidates (inclusion coverage, missing-parent ratio)
* Normalisation configuration used during discovery

---

## Outputs

Single artifact: `schema_recommendations.json`

Contains:

* structure recommendations
* constraint recommendations
* datatype recommendations
* performance recommendations
* confidence, reason_codes, blockers
* alternatives (runner-ups)
* normalisation_disclosures

---

## Recommendation Categories

### Structure

* Primary key (single or composite)
* Foreign key relationships
* Composite keys
* Surrogate key suggestion

### Constraints

* PRIMARY KEY
* UNIQUE
* NOT NULL
* FOREIGN KEY
* Optional CHECK / DEFAULT (conservative)

### Datatypes

* SMALLINT / INT / BIGINT
* DECIMAL(p,s)
* BOOLEAN
* DATE / TIMESTAMP
* UUID
* VARCHAR(n) / TEXT
* JSON (optional)

### Performance

* PK index suggestion
* FK index suggestion
* Composite index candidate
* Avoid-index warnings (low cardinality / TEXT)

---

## Confidence + Ranking

Each recommendation includes:

* confidence (0–1)
* reason_codes
* blockers (if enforcement unsafe)
* alternatives (ranked runner-ups)
* normalisation_disclosure

---

# JSON Output Design

## Top level

```json
{
  "meta": {
    "generated_at": "ISO_TIMESTAMP",
    "pipeline_version": "0.1",
    "source": "csv_upload"
  },
  "tables": []
}
```

---

## Table structure

```json
{
  "table_name": "orders",
  "structure": {},
  "constraints": {},
  "datatypes": {},
  "performance": {}
}
```

---

## Primary key example

```json
{
  "columns": ["order_id"],
  "decision": "PRIMARY_KEY",
  "confidence": 0.98,
  "reason_codes": ["pk_unique_high", "pk_null_low"],
  "blockers": [],
  "alternatives": [],
  "normalisation_disclosure": {
    "applied": ["trim_whitespace"],
    "notes": ""
  }
}
```

---

## Constraint example

```json
{
  "columns": ["order_id"],
  "constraint": "NOT_NULL",
  "confidence": 0.99,
  "reason_codes": ["pk_null_low"],
  "blockers": []
}
```

---

## Datatype example

```json
{
  "column": "order_id",
  "recommended": {
    "db_type": "BIGINT",
    "confidence": 0.93,
    "reason_codes": ["dtype_int_range_big"]
  },
  "alternatives": []
}
```

---

## Performance example

```json
{
  "indexes_recommended": [
    {
      "columns": ["customer_id"],
      "priority": "HIGH",
      "confidence": 0.90,
      "reason_codes": ["idx_foreign_key"]
    }
  ],
  "indexes_avoid": [
    {
      "columns": ["status"],
      "reason_codes": ["idx_low_cardinality_avoid"]
    }
  ]
}
```
