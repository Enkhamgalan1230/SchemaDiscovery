from pathlib import Path


ALLOWED_FILES = [
    "candidates/ind_unary_soft.py",
    "candidates/ind_unary.py",
    "candidates/ucc_unary.py",
    "pipeline/run.py",
    "profiling/profiler.py",
    "profiling/profiler_sql.py",
    "quality/key_normalisation.py",
    "quality/key_representations.py",
    "scoring/fk_score.py",
    "selection/select_edges.py",
    "storage/duckdb_store.py",
    "storage/register.py",
    "viz/erd_graphviz.py",
]


def export_python_files_to_txt(
    root_folder: str = "schema_discovery",
    output_file: str = "schema_discovery_dump.txt",
) -> None:
    root = Path(root_folder)

    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root.resolve()}")

    py_files = []

    for rel_path in ALLOWED_FILES:
        file_path = root / rel_path

        if not file_path.exists():
            print(f"Warning: file not found -> {file_path}")
            continue

        py_files.append(file_path)

    with open(output_file, "w", encoding="utf-8") as out:
        for file_path in py_files:
            out.write(f"{file_path.as_posix()}:\n\n")

            try:
                out.write(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                out.write(f"[Error reading file: {e}]\n")

            out.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Done. Wrote {len(py_files)} files to {output_file}")


if __name__ == "__main__":
    export_python_files_to_txt()