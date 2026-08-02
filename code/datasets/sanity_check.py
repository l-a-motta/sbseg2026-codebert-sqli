"""
Sanity check: sample N rows from compiled_dataset.csv, print each row as a
readable card, and save the sample to compiled/sanity_check.csv.

All configuration is at the top of this file.
"""

from pathlib import Path
import textwrap

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLE        = 100           # rows to sample and display
SEED            = 42           # fixed seed → reproducible sample; None → fresh draw each run
QUERY_MAX_CHARS = 300          # truncate long queries in the display (full text is in the CSV)
CHUNK_SIZE      = 500_000      # rows per streaming chunk
ESTIMATED_ROWS  = 12_732_002   # from F1.3 — drives proportional sampling

DATASET_PATH = Path(__file__).parent / "compiled" / "compiled_dataset.csv"
OUTPUT_PATH  = Path(__file__).parent / "compiled" / "sanity_check.csv"
# ── End Configuration ─────────────────────────────────────────────────────────

LABEL_NAMES = {0: "0  (benign)", 1: "1  (malicious)"}

# Columns shown first; everything else follows in CSV order (NaN fields skipped)
PRIORITY_COLS = ["query", "label", "source", "split", "injection_type", "attack_technique"]


def sample_dataset(path: Path, n: int, seed, chunk_size: int, estimated_rows: int) -> pd.DataFrame:
    """Stream the CSV in chunks and return a proportional random sample of n rows."""
    oversample = 4  # collect ~4× target per chunk, then trim — ensures n rows survive
    fraction = (n * oversample) / estimated_rows

    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        k = max(1, round(len(chunk) * fraction))
        k = min(k, len(chunk))
        chunks.append(chunk.sample(n=k, random_state=seed))

    combined = pd.concat(chunks, ignore_index=True)
    if len(combined) > n:
        combined = combined.sample(n=n, random_state=seed).reset_index(drop=True)
    return combined


def format_field(col: str, val) -> str:
    """Return a display-friendly string for a single field value."""
    if col == "label":
        try:
            return LABEL_NAMES.get(int(val), str(val))
        except (TypeError, ValueError):
            return str(val)
    if col == "query":
        text = str(val).replace("\n", " ").replace("\r", " ")
        if len(text) > QUERY_MAX_CHARS:
            return text[:QUERY_MAX_CHARS] + f"  […+{len(str(val)) - QUERY_MAX_CHARS} chars]"
        return text
    return str(val)


def print_row_card(index: int, total: int, row: pd.Series) -> None:
    """Print one row as a vertical field: value card."""
    divider = "-" * 72
    print(divider)
    print(f"  RECORD {index + 1} / {total}")
    print(divider)

    ordered = PRIORITY_COLS + [c for c in row.index if c not in PRIORITY_COLS]
    for col in ordered:
        val = row[col]
        if pd.isna(val):
            continue
        label_width = 24
        prefix = f"  {col:<{label_width}}: "
        content = format_field(col, val)
        wrapped = textwrap.fill(
            content,
            width=110,
            initial_indent=prefix,
            subsequent_indent=" " * (label_width + 4),
        )
        print(wrapped)
    print()


def print_summary(df: pd.DataFrame) -> None:
    """Print a one-block summary of the sample before the individual cards."""
    divider = "=" * 72
    print(f"\n{divider}")
    print(f"  SAMPLE SUMMARY  ({len(df)} rows)")
    print(divider)

    label_counts = df["label"].value_counts().to_dict()
    for val, name in LABEL_NAMES.items():
        count = label_counts.get(val, 0)
        print(f"  {'label ' + name:<30}  {count:>4} rows")

    if "source" in df.columns:
        print()
        for src, count in df["source"].value_counts().items():
            print(f"  {'source: ' + str(src):<30}  {count:>4} rows")

    if "split" in df.columns:
        print()
        for sp, count in df["split"].value_counts().items():
            print(f"  {'split: ' + str(sp):<30}  {count:>4} rows")

    print(f"{divider}\n")


def main() -> None:
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        print("        Run compile.py first to generate compiled_dataset.csv.")
        return

    print(f"Sampling {N_SAMPLE} rows from {DATASET_PATH.name} …")
    df = sample_dataset(DATASET_PATH, N_SAMPLE, SEED, CHUNK_SIZE, ESTIMATED_ROWS)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved sample to {OUTPUT_PATH}")

    print_summary(df)

    for i, (_, row) in enumerate(df.iterrows()):
        print_row_card(i, len(df), row)

    print("-" * 72)
    print(f"  Full sample also saved to: {OUTPUT_PATH.name}")
    print("-" * 72)


if __name__ == "__main__":
    main()
