# Schema Discovery Pipeline

## Overview

This project implements a **schema discovery pipeline** that automatically infers relationships between tables directly from data.

No predefined **primary keys**, **foreign keys**, or **constraints** are assumed.  
All structure is inferred using statistical evidence derived from the dataset itself.

---

## Core Idea

Relational structure leaves **statistical fingerprints** in data.

This pipeline reconstructs schema by:
1) profiling column behaviour  
2) identifying unique identifiers  
3) testing value inclusion  
4) scoring relationship strength  
5) selecting a consistent structure  
6) validating data quality around discovered relationships  
7) exposing results through a stable API  


Each stage is independent, testable, and explainable.

---

## Pipeline Stages

1) Column profiling  
2) Unary UCC discovery  
3) Unary IND discovery  
4) IND scoring  
5) Relationship selection  
6) Optional quality checks  
7) Schema visualisation  

---

## 1. Column Profiling

### What it does
Profiles every column in every table and produces a unified `profiles_df` containing one row per column.

Base statistics (always computed):
- total row count
- null count and null ratio
- number of distinct values
- uniqueness ratios (overall and non-null)
- inferred data type family
- sample values
- basic length statistics for string columns

Enhanced mode (optional):
- atomic type inference and confidence
- completeness and consistency scores
- overall column quality score
- numeric distribution statistics
- date range and freshness metrics
- integer continuity for key-like columns

### Why it matters
This stage is the **single source of truth** for the entire pipeline.

All later decisions rely exclusively on profiler evidence.  
No structural assumptions are introduced downstream.

---

## 2. Unary UCC Discovery (Unique Column Combinations)

### What it does
Identifies columns that uniquely identify rows **within a table**.

A column is a unary UCC if:
- it contains no null values
- all values are unique across rows

### Why it matters
Foreign keys can only reference **unique identifiers**.

This step defines the set of valid parent key candidates for relationship discovery.

---

## 3. Unary IND Discovery (Inclusion Dependencies)

### What it does
Tests whether values in one column are included in the domain of another column.

In simple terms:

FK.column ⊆ PK.column


For each candidate FK → PK pair, the pipeline computes:
- distinct value coverage
- row-level coverage
- non-null row counts
- domain compatibility constraints

Only statistically meaningful candidates are retained.

### Why it matters
All foreign key relationships imply inclusion.

This step discovers **all possible relationships using data only**, without relying on names or metadata.

---

## 4. IND Scoring

### What it does
Assigns a confidence score to each IND candidate.

The score is a weighted combination of:
- distinct coverage (dominant signal)
- row coverage
- value range compatibility penalty
- column name similarity (weak, tie-breaking signal)

Structural signals dominate.  
Naming information is used only to disambiguate otherwise equivalent candidates.

### Why it matters
Large datasets naturally contain coincidental overlaps.

Scoring ranks candidates so only **structurally consistent** relationships survive.

---

## 5. Relationship Selection

### What it does
For each foreign key column:
- selects the single best parent column
- enforces one logical parent per FK
- assigns relationship cardinality:
  - many-to-one
  - one-to-one
- determines optionality using null ratios

Only candidates above a minimum confidence threshold are accepted.

### Why it matters
A usable schema must be **unambiguous and deterministic**.

This step produces a clean, final relational structure.

---

## 6. Optional Quality Checks

These checks validate **data quality around discovered relationships**.  
They do not affect schema discovery itself.

### Relational Missingness
For each accepted relationship:
- detects orphan foreign keys
- detects parent keys with no children
- classifies severity (ok, warning, fail)
- reports likely causes and recommended actions

### Duplicate Detection
The pipeline can detect:
- duplicate values in key-like columns
- duplicate values in natural keys (email, phone, username)
- cross-table identifier overlaps
- exact duplicate rows
- duplicate rows based on business keys
- identifier conflicts (same entity, multiple IDs)
- relationship table cardinality violations

All checks are rule-based and fully explainable.

---

## 7. Schema Visualisation

### What it does
Converts selected relationships into diagram-ready formats.

Supported outputs:
- Graphviz DOT (via API)
- Mermaid ER diagrams (via frontend)

### Why it matters
The inferred schema becomes:
- easy to inspect
- easy to document
- easy to share

---

## API Structure

### Stage-based Output

The API returns **only the stages explicitly requested**.

Examples:

- `include=profiles`  
  → basic profiler runs  
  → minimal, schema-safe output  

- `include=profiles_enhanced`  
  → enhanced profiler runs  
  → same base columns plus diagnostics  

- `include=profiles,profiles_enhanced`  
  → enhanced profiler runs once  
  → API slices output into two views  

---

## Example API Response (Excerpt)

```json
  {
    "meta": {
      "runtime_sec": 0.18,
      "tables": {
        "orders": { "rows": 1200, "cols": 6 }
      },
      "requested_stages": [
        "relationships",
        "profiles",
        "profiles_enhanced"
      ],
      "config_used": {
        "profile_mode": "enhanced",
        "min_relationship_score": 0.9,
        "min_distinct_coverage": 0.9,
        "min_non_null_rows": 20,
        "max_rows_per_stage": 5,
        "debug_bundle": false
      }
    },
    "results": {
      "relationships": {
        "summary": {
          "count": 7,
          "many_to_one": 7,
          "one_to_one": 0,
          "optional": 0,
          "required": 7
        },
        "items": [
          {
            "relationship": "orders.CustomerID -> customers.CustomerID",
            "type": {
              "cardinality": "many_to_one",
              "optional": false
            },
            "confidence": {
              "score": 1.0,
              "distinct_coverage": 1.0,
              "row_coverage": 1.0
            },
            "evidence": {
              "child_non_null_fk_rows": 1200,
              "child_distinct_fk_values": 200,
              "parent_distinct_pk_values": 200
            }
          }
        ]
      }
    }
  }
```

