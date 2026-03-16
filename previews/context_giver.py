from __future__ import annotations

from pathlib import Path
import pandas as pd


def csvs_to_preview_txt(
    input_dir: str | Path,
    output_txt: str | Path,
    n_rows: int = 7,
    recursive: bool = True,
) -> None:
    """
    For each CSV in input_dir, write:
      - filename
      - first n_rows of data (including header)
    into one output .txt file.
    """
    input_dir = Path(input_dir)
    output_txt = Path(output_txt)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = sorted(input_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir} (recursive={recursive})")

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_txt.open("w", encoding="utf-8") as out:
        for csv_path in csv_files:
            out.write(f"{csv_path.name}\n")
            out.write("=" * len(csv_path.name) + "\n")

            try:
                # Read only the first n_rows quickly and robustly.
                df = pd.read_csv(
                    csv_path,
                    nrows=n_rows,
                    encoding="utf-8",
                    engine="python",
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError:
                # Fallback for Windows-origin / mixed-encoding files
                df = pd.read_csv(
                    csv_path,
                    nrows=n_rows,
                    encoding="latin-1",
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e:
                out.write(f"[ERROR] Failed to read file: {e}\n\n")
                continue

            if df.empty:
                out.write("[INFO] File read successfully but returned 0 rows.\n\n")
                continue

            # Write the preview exactly like a CSV snippet (header + rows)
            out.write(df.to_csv(index=False, lineterminator="\n"))
            out.write("\n\n")  # spacing between datasets


if __name__ == "__main__":
    # Example usage:
    # Put your CSVs in: data/
    # Output preview to: previews/all_previews.txt
    csvs_to_preview_txt(
        input_dir="data/Anon BML Data",
        output_txt="previews/all_previews.txt",
        n_rows=7,
        recursive=True,
    )
    print("Done -> wrote previews/all_previews.txt")