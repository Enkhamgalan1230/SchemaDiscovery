"""
ERD rendering utilities.

Converts discovered relationships (edges_df) into Mermaid ER diagram syntax.
This is a lightweight way to visualise an inferred schema in notebooks/docs.
"""

from __future__ import annotations

import re
import pandas as pd


def _safe_id(name: str) -> str:
    """
    Mermaid identifiers work best with alphanumerics and underscores.
    """
    s = str(name).strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "T"
    if s[0].isdigit():
        s = "T_" + s
    return s


def edges_to_mermaid_erd(
    edges_df: pd.DataFrame,
    include_columns: bool = True,
    max_cols_per_table: int = 12,
) -> str:
    required = {"fk_table", "fk_column", "pk_table", "pk_column"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df missing required columns: {sorted(missing)}")

    has_card = "cardinality" in edges_df.columns
    has_opt = "optional" in edges_df.columns

    table_cols: dict[str, set[str]] = {}
    for r in edges_df.itertuples(index=False):
        table_cols.setdefault(str(r.fk_table), set()).add(str(r.fk_column))
        table_cols.setdefault(str(r.pk_table), set()).add(str(r.pk_column))

    lines: list[str] = ["erDiagram"]

    if include_columns:
        for table, cols in sorted(table_cols.items(), key=lambda x: x[0].lower()):
            t = _safe_id(table)
            lines.append(f"  {t} {{")
            for c in sorted(cols)[:max_cols_per_table]:
                lines.append(f"    string {_safe_id(c)}")
            if len(cols) > max_cols_per_table:
                lines.append("    string more_columns")
            lines.append("  }")

    for r in edges_df.itertuples(index=False):
        fk_t = _safe_id(r.fk_table)
        pk_t = _safe_id(r.pk_table)

        rel = "||--o{"
        if has_card:
            card = getattr(r, "cardinality")
            rel = "||--||" if card == "one_to_one" else "||--o{"

        # clearer label: parent col -> child col
        col_label = f"{r.pk_column} -> {r.fk_column}"

        if has_opt and bool(getattr(r, "optional")):
            col_label = f"{col_label} (optional)"

        lines.append(f'  {pk_t} {rel} {fk_t} : "{col_label}"')

    return "\n".join(lines)
