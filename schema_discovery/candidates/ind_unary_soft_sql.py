from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from schema_discovery.candidates.ind_unary_soft import (
    SoftIndConfig,
    best_representation_match,
    _numeric_minmax,
    _build_parent_candidates,
    _build_parent_candidates_from_profiles,
    _build_child_candidates,
    _route_pairs,
)
from schema_discovery.quality.key_representations import build_key_representations
from schema_discovery.storage import DuckDBTableStore


def _quote_ident(name: str) -> str:
    if not isinstance(name, str) or name == "":
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _read_column_from_store(store: DuckDBTableStore, table_name: str, column_name: str) -> pd.Series:
    sql = f"""
    SELECT {_quote_ident(column_name)} AS value
    FROM {_quote_ident(table_name)}
    """
    df = store.fetchdf(sql)
    if "value" not in df.columns:
        return pd.Series(dtype="object")
    return df["value"]


class _ColumnCache:
    def __init__(
        self,
        *,
        dfs: Optional[Dict[str, pd.DataFrame]],
        store: Optional[DuckDBTableStore],
        max_items: int = 12,
    ):
        self.dfs = dfs
        self.store = store
        self.max_items = max(1, int(max_items))
        self.cache: OrderedDict[tuple[str, str], pd.Series] = OrderedDict()

    def get(self, table_name: str, column_name: str) -> pd.Series:
        key = (str(table_name), str(column_name))

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        if self.dfs is not None:
            s = self.dfs[table_name][column_name]
        elif self.store is not None:
            s = _read_column_from_store(self.store, table_name, column_name)
        else:
            raise ValueError("Either dfs or store must be provided")

        self.cache[key] = s
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return s


def discover_ind_unary_soft_all_pairs(
    *,
    profiles_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    cfg: Optional[SoftIndConfig] = None,
    dfs: Optional[Dict[str, pd.DataFrame]] = None,
    store: Optional[DuckDBTableStore] = None,
    reject_log_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Hybrid soft IND discovery.

    Matching logic stays in Python.
    Data can come from:
    - dfs for old in-memory path
    - store for DuckDB path
    """
    cfg = cfg or SoftIndConfig()

    if (dfs is None) and (store is None):
        raise ValueError("Provide either dfs or store for soft IND")

    if (dfs is not None) and (store is not None):
        raise ValueError("Provide only one of dfs or store for soft IND")

    if cfg.parent_source == "profile_keylike":
        parents = _build_parent_candidates_from_profiles(profiles_df, cfg)
    elif cfg.parent_source == "ucc_plus_profile":
        p1 = _build_parent_candidates(ucc_df, profiles_df, cfg)
        p2 = _build_parent_candidates_from_profiles(profiles_df, cfg)
        parents = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
    else:
        parents = _build_parent_candidates(ucc_df, profiles_df, cfg)

    children = _build_child_candidates(profiles_df, ucc_df, cfg)
    cand_pairs = _route_pairs(children, parents, cfg)

    out_cols = [
        "fk_table", "fk_column", "pk_table", "pk_column",
        "distinct_coverage", "row_coverage",
        "fk_distinct", "pk_distinct", "intersection_distinct",
        "fk_min", "fk_max", "pk_min", "pk_max",
        "matched_rows", "fk_non_null_rows",
        "fk_rep", "pk_rep",
        "mode",
    ]

    if not cand_pairs:
        return pd.DataFrame(columns=out_cols)

    if dfs is not None:
        available_tables = set(dfs.keys())
        columns_by_table = {str(t): set(df.columns.astype(str)) for t, df in dfs.items()}
    else:
        available_tables = set(store.list_tables())
        columns_by_table = {}
        for t in available_tables:
            desc = store.describe_table(t)
            columns_by_table[str(t)] = set(desc["column_name"].astype(str).tolist())

    cache = _ColumnCache(dfs=dfs, store=store, max_items=12)

    out_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []

    def _reject(fk_t: str, fk_c: str, pk_t: str, pk_c: str, ns: float, reason: str) -> None:
        reject_rows.append(
            {
                "fk_table": fk_t,
                "fk_column": fk_c,
                "pk_table": pk_t,
                "pk_column": pk_c,
                "name_sim": float(ns),
                "reason": str(reason),
            }
        )

    def _effective_non_null(s: pd.Series) -> int:
        x = s.astype("string").str.strip()
        x = x.where(x.str.len() > 0)
        return int(x.notna().sum())

    for fk_table, fk_col, pk_table, pk_col, ns in cand_pairs:
        if fk_table not in available_tables or pk_table not in available_tables:
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "table_missing")
            continue

        if fk_col not in columns_by_table.get(str(fk_table), set()) or pk_col not in columns_by_table.get(str(pk_table), set()):
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "column_missing")
            continue

        fk_s = cache.get(fk_table, fk_col)
        pk_s = cache.get(pk_table, pk_col)

        fk_eff = _effective_non_null(fk_s)
        pk_eff = _effective_non_null(pk_s)

        if fk_eff < int(cfg.min_non_null) or pk_eff < int(cfg.min_non_null):
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "min_non_null_effective")
            continue

        best = best_representation_match(fk_s, pk_s, cfg)

        if best["fk_rep"] is None or best["pk_rep"] is None:
            _reject(
                fk_table, fk_col, pk_table, pk_col, ns,
                f"no_representation_match fk_eff={fk_eff} pk_eff={pk_eff}"
            )
            continue

        if int(best["fk_distinct"]) <= int(cfg.small_domain_fk_distinct_max):
            fk_reps_tmp = build_key_representations(fk_s, other=pk_s, cfg=cfg.rep_cfg)
            fk_tmp = fk_reps_tmp.get(best["fk_rep"], fk_s).dropna().astype("string")

            top1_ratio = 0.0
            if len(fk_tmp):
                vc = fk_tmp.value_counts(dropna=True)
                if not vc.empty:
                    top1_ratio = float(vc.iloc[0]) / float(len(fk_tmp))

            if top1_ratio >= float(cfg.small_domain_top1_ratio_gate) and float(ns) < float(cfg.small_domain_min_name_sim):
                _reject(fk_table, fk_col, pk_table, pk_col, ns, "small_domain_guard")
                continue

        dc = float(best["distinct_coverage"])
        rc = float(best["row_coverage"])
        mr = int(best["matched_rows"])

        accept = False

        if dc >= float(cfg.min_distinct_coverage) and rc >= float(cfg.min_row_coverage):
            accept = True

        if (not accept) and dc >= float(cfg.alt_min_distinct_coverage) and rc >= float(cfg.alt_min_row_coverage):
            accept = True

        if (not accept) and dc >= float(cfg.min_distinct_coverage) and mr >= int(cfg.min_matched_rows):
            accept = True

        if not accept:
            _reject(fk_table, fk_col, pk_table, pk_col, ns, "coverage_fail")
            continue

        fk_reps = build_key_representations(fk_s, other=pk_s, cfg=cfg.rep_cfg)
        pk_reps = build_key_representations(pk_s, other=fk_s, cfg=cfg.rep_cfg)

        fk_best_s = fk_reps.get(best["fk_rep"], fk_s)
        pk_best_s = pk_reps.get(best["pk_rep"], pk_s)

        fk_min, fk_max = _numeric_minmax(fk_best_s)
        pk_min, pk_max = _numeric_minmax(pk_best_s)

        out_rows.append(
            {
                "fk_table": fk_table,
                "fk_column": fk_col,
                "pk_table": pk_table,
                "pk_column": pk_col,
                "distinct_coverage": float(best["distinct_coverage"]),
                "row_coverage": float(best["row_coverage"]),
                "fk_distinct": int(best["fk_distinct"]),
                "pk_distinct": int(best["pk_distinct"]),
                "intersection_distinct": int(best["intersection_distinct"]),
                "fk_min": fk_min,
                "fk_max": fk_max,
                "pk_min": pk_min,
                "pk_max": pk_max,
                "matched_rows": int(best["matched_rows"]),
                "fk_non_null_rows": int(best["fk_non_null_rows"]),
                "fk_rep": str(best["fk_rep"]),
                "pk_rep": str(best["pk_rep"]),
                "mode": "soft",
            }
        )

    if reject_log_path is not None:
        try:
            pd.DataFrame(reject_rows).to_csv(reject_log_path, index=False)
        except Exception:
            pass

    if not out_rows:
        return pd.DataFrame(columns=out_cols)

    return pd.DataFrame(out_rows)[out_cols]