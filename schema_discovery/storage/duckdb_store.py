# schema_discovery/storage/duckdb_store.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """
    Safe identifier quoting for DuckDB table / column names.

    Allows real file-derived and CSV-derived names with spaces, slashes,
    brackets, hashes, etc. by escaping embedded double quotes.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    """
    Safe SQL string literal.
    """
    if not isinstance(value, str):
        value = str(value)
    return "'" + value.replace("'", "''") + "'"


class DuckDBTableStore:
    """
    File-backed table store using DuckDB.

    Responsibilities:
    - Open and configure DuckDB
    - Register CSV / Parquet files as SQL views
    - Expose metadata helpers
    - Expose fetch helpers
    - Fetch a single column only when Python really needs it
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        memory_limit: str = "5GB",
        threads: int = 4,
        temp_directory: str | Path | None = None,
        preserve_insertion_order: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path)

        # Sensible defaults for a machine with limited RAM
        self.con.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
        self.con.execute(f"SET threads = {int(threads)}")
        self.con.execute(
            f"SET preserve_insertion_order = {'true' if preserve_insertion_order else 'false'}"
        )

        if temp_directory is not None:
            temp_dir = Path(temp_directory)
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"SET temp_directory = {_sql_string(str(temp_dir))}")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_parquet(self, table_name: str, path: str | Path) -> None:
        table_q = _quote_ident(table_name)
        path_s = _sql_string(str(Path(path)))
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW {table_q} AS
            SELECT * FROM read_parquet({path_s})
            """
        )

    def register_csv(
        self,
        table_name: str,
        path: str | Path,
        *,
        header: bool = True,
        sample_size: int = -1,
        all_varchar: bool = False,
    ) -> None:
        """
        Register CSV as a view.

        sample_size=-1 asks DuckDB to inspect the full file for type inference.
        all_varchar=True is safer for messy exports if you want to avoid bad type guesses.
        """
        table_q = _quote_ident(table_name)
        path_s = _sql_string(str(Path(path)))

        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW {table_q} AS
            SELECT * FROM read_csv_auto(
                {path_s},
                header={'true' if header else 'false'},
                sample_size={int(sample_size)},
                all_varchar={'true' if all_varchar else 'false'}
            )
            """
        )

    def register_path(
        self,
        table_name: str,
        path: str | Path,
        *,
        csv_all_varchar: bool = False,
    ) -> None:
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == ".parquet":
            self.register_parquet(table_name, p)
        elif suffix == ".csv":
            self.register_csv(table_name, p, all_varchar=csv_all_varchar)
        else:
            raise ValueError(f"Unsupported file type for {p}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def list_tables(self) -> list[str]:
        rows = self.con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
        return [str(r[0]) for r in rows]

    def describe_table(self, table_name: str) -> pd.DataFrame:
        table_q = _quote_ident(table_name)
        return self.con.execute(f"DESCRIBE {table_q}").fetchdf()

    def list_columns(self, table_name: str) -> list[str]:
        desc = self.describe_table(table_name)
        if "column_name" not in desc.columns:
            raise ValueError(f"DESCRIBE output for {table_name} does not contain 'column_name'")
        return [str(x) for x in desc["column_name"].tolist()]

    def count_rows(self, table_name: str) -> int:
        table_q = _quote_ident(table_name)
        row = self.con.execute(f"SELECT COUNT(*) FROM {table_q}").fetchone()
        return int(row[0]) if row is not None else 0

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def execute(self, sql: str) -> None:
        self.con.execute(sql)

    def fetchdf(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    def fetchall(self, sql: str) -> list[tuple[Any, ...]]:
        return self.con.execute(sql).fetchall()

    def fetchone(self, sql: str) -> tuple[Any, ...] | None:
        return self.con.execute(sql).fetchone()

    def fetch_scalar(self, sql: str, default: Any = None) -> Any:
        row = self.fetchone(sql)
        if row is None:
            return default
        return row[0]

    # ------------------------------------------------------------------
    # Column access for later hybrid logic
    # ------------------------------------------------------------------
    def read_column(self, table_name: str, column_name: str) -> pd.Series:
        table_q = _quote_ident(table_name)
        col_q = _quote_ident(column_name)
        df = self.con.execute(f"SELECT {col_q} FROM {table_q}").fetchdf()
        return df[column_name]

    def read_columns(self, table_name: str, columns: Iterable[str]) -> pd.DataFrame:
        cols = list(columns)
        if not cols:
            return pd.DataFrame()

        table_q = _quote_ident(table_name)
        cols_q = ", ".join(_quote_ident(c) for c in cols)
        return self.con.execute(f"SELECT {cols_q} FROM {table_q}").fetchdf()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass