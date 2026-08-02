# Stage 1 — Raw Dataset Exploration (read-only)
#
# Reads every CSV under datasets/raw/ and produces a comprehensive log of
# their schema, volume, and quality characteristics. This script does NOT
# modify any file in raw/.
#
# Output:
#   - Console: concise summary (INFO level)
#   - Log file at code/compilation/logs/explore_raw_YYYYMMDD_HHMMSS.log
#     containing every detail (DEBUG level)
#
# Used to inform the unified column mapping in compile.py.

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

RAW_DIR = Path("code/datasets/raw")
LOG_DIR = Path("code/compilation/logs")

# Files larger than this (in MB) are treated as "large" and streamed.
LARGE_FILE_MB = 100
PEEK_ROWS = 10
CHUNK_SIZE = 100_000

# Encodings tried in order until one works.
ENCODINGS = ["utf-8", "utf-16", "latin-1"]

# Columns with at most this many distinct values get a full value-distribution
# logged (useful for spotting label imbalance, enum-like columns, etc.).
LOW_CARDINALITY_THRESHOLD = 50

# Sample this many rows per label value so we can eyeball whether label=1
# really means "malicious" in this file or whether the convention is inverted.
SAMPLES_PER_LABEL = 3
SAMPLE_TRUNCATE_CHARS = 200

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> Path:
    """Configure root logger with a verbose file handler and a concise console handler."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"explore_raw_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    return log_path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def detect_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, nrows=1, encoding=enc, low_memory=False)
            logging.debug(f"  encoding {enc!r}: OK")
            return enc
        except (UnicodeDecodeError, UnicodeError) as e:
            logging.debug(f"  encoding {enc!r}: FAILED ({type(e).__name__})")
            continue
        except pd.errors.ParserError:
            # Encoding decoded fine but a row was malformed — that's not an
            # encoding problem, so this encoding is acceptable.
            return enc
    raise UnicodeDecodeError(
        "explore_raw", b"", 0, 1, f"None of {ENCODINGS} could decode {path.name}"
    )


def safe_read_csv(path: Path, encoding: str, **kwargs):
    """Read CSV with C engine, fall back to the python engine on ParserError."""
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
    except pd.errors.ParserError as e:
        logging.warning(
            f"  C-engine ParserError ({e}). Retrying with python engine."
        )
        kwargs.pop("low_memory", None)
        return pd.read_csv(path, encoding=encoding, engine="python", **kwargs)


def log_column_stats(series: pd.Series, n_rows: int) -> None:
    """Log per-column statistics at DEBUG level."""
    col = series.name
    dtype = series.dtype
    nan_count = int(series.isna().sum())
    nan_pct = 100 * nan_count / max(n_rows, 1)
    n_unique = series.nunique(dropna=True)

    logging.debug(f"  - {col} ({dtype})")
    logging.debug(f"      NaN: {nan_count:,} ({nan_pct:.2f}%)  unique: {n_unique:,}")

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        non_null = series.dropna().astype(str)
        empty_count = int((non_null.str.len() == 0).sum())
        if len(non_null) > 0:
            lengths = non_null.str.len()
            logging.debug(
                f"      string lengths: min={lengths.min()}  max={lengths.max()}  "
                f"mean={lengths.mean():.1f}  median={int(lengths.median())}"
            )
        logging.debug(f"      empty strings: {empty_count:,}")
    elif pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            logging.debug(
                f"      numeric: min={non_null.min()}  max={non_null.max()}  "
                f"mean={non_null.mean():.4f}"
            )

    if 0 < n_unique <= LOW_CARDINALITY_THRESHOLD:
        value_counts = series.value_counts(dropna=False)
        logging.debug(f"      value distribution:")
        for val, count in value_counts.items():
            pct = 100 * count / max(n_rows, 1)
            logging.debug(f"        {val!r}: {count:,} ({pct:.2f}%)")


def find_label_column(columns) -> str | None:
    """Pick a column called 'label' (case-insensitive). Returns None if absent."""
    return next((c for c in columns if c.lower() == "label"), None)


def find_primary_text_column(df: pd.DataFrame) -> str | None:
    """Pick the string column with the largest mean character length."""
    candidates = []
    for col in df.columns:
        series = df[col]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            continue
        candidates.append((non_null.str.len().mean(), col))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def truncate(text: str, limit: int = SAMPLE_TRUNCATE_CHARS) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def log_label_samples(samples_per_label: dict, label_col: str, text_col: str) -> None:
    """Log a few text samples per distinct label value so we can spot inversion."""
    logging.info(f"  Samples per label (text column: {text_col!r}):")
    for label_value in sorted(samples_per_label.keys(), key=lambda v: (v is None, str(v))):
        logging.info(f"    label={label_value!r}:")
        for i, text in enumerate(samples_per_label[label_value], 1):
            logging.info(f"      [{i}] {truncate(text)}")


def collect_samples_from_df(df: pd.DataFrame, label_col: str, text_col: str) -> dict:
    """Collect SAMPLES_PER_LABEL non-null text samples for each label value."""
    out: dict = {}
    for label_value, group in df.groupby(label_col, dropna=False):
        non_null = group[text_col].dropna().astype(str).head(SAMPLES_PER_LABEL).tolist()
        out[label_value] = non_null
    return out


def log_dataframe_stats(df: pd.DataFrame) -> None:
    n_rows = len(df)
    full_dup_count = int(df.duplicated().sum())
    dup_pct = 100 * full_dup_count / max(n_rows, 1)
    logging.debug(f"  Per-column stats (n={n_rows:,}):")
    logging.debug(f"  Fully-duplicate rows: {full_dup_count:,} ({dup_pct:.2f}%)")

    for col in df.columns:
        log_column_stats(df[col], n_rows)


def explore_small_file(path: Path, encoding: str) -> None:
    logging.info(f"  Reading full file with encoding={encoding!r} ...")
    t0 = datetime.now()
    df = safe_read_csv(path, encoding=encoding)
    elapsed = (datetime.now() - t0).total_seconds()
    logging.info(
        f"  Read complete in {elapsed:.2f}s — shape: {df.shape[0]:,} rows × {df.shape[1]} cols"
    )

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logging.info(f"  In-memory footprint: {mem_mb:.2f} MB")

    logging.info(f"  Columns:")
    for col, dtype in df.dtypes.items():
        logging.info(f"    - {col} ({dtype})")

    log_dataframe_stats(df)

    label_col = find_label_column(df.columns)
    text_col = find_primary_text_column(df)
    if label_col and text_col and label_col != text_col:
        samples = collect_samples_from_df(df, label_col, text_col)
        log_label_samples(samples, label_col, text_col)
    elif text_col and not label_col:
        logging.info(f"  No label column — sampling first {SAMPLES_PER_LABEL} rows from {text_col!r}:")
        for i, text in enumerate(df[text_col].dropna().astype(str).head(SAMPLES_PER_LABEL), 1):
            logging.info(f"    [{i}] {truncate(text)}")

    logging.info(f"  First {PEEK_ROWS} rows:")
    for line in df.head(PEEK_ROWS).to_string(max_colwidth=80).splitlines():
        logging.info(f"    {line}")


def explore_large_file(path: Path, encoding: str) -> None:
    logging.info(f"  Peeking head with encoding={encoding!r} (nrows={PEEK_ROWS})...")
    head = pd.read_csv(path, nrows=PEEK_ROWS, encoding=encoding, low_memory=False)
    logging.info(f"  Columns:")
    for col, dtype in head.dtypes.items():
        logging.info(f"    - {col} ({dtype})")

    logging.info(f"  Streaming full file in chunks of {CHUNK_SIZE:,} ...")
    t0 = datetime.now()
    total_rows = 0
    nan_per_col = {col: 0 for col in head.columns}
    dup_within_chunk = 0
    label_col = find_label_column(head.columns)
    text_col = find_primary_text_column(head)
    label_counts: dict = {}
    samples_per_label: dict = {}

    # Track value distributions for low-cardinality string columns. We stop
    # tracking any column once it exceeds STREAM_DISTINCT_LIMIT so that
    # high-cardinality columns (e.g., the query text itself) do not blow up
    # memory.
    STREAM_DISTINCT_LIMIT = 200
    string_cols = [
        c for c in head.columns
        if head[c].dtype == object or pd.api.types.is_string_dtype(head[c])
    ]
    string_value_counts: dict = {c: {} for c in string_cols}
    abandoned_string_cols: set = set()

    chunks = pd.read_csv(path, chunksize=CHUNK_SIZE, encoding=encoding, low_memory=False)
    for i, chunk in enumerate(chunks, start=1):
        total_rows += len(chunk)
        for col in chunk.columns:
            nan_per_col[col] = nan_per_col.get(col, 0) + int(chunk[col].isna().sum())
        dup_within_chunk += int(chunk.duplicated().sum())
        if label_col:
            for val, count in chunk[label_col].value_counts(dropna=False).items():
                label_counts[val] = label_counts.get(val, 0) + int(count)
            if text_col and label_col != text_col:
                for label_value, group in chunk.groupby(label_col, dropna=False):
                    bucket = samples_per_label.setdefault(label_value, [])
                    if len(bucket) >= SAMPLES_PER_LABEL:
                        continue
                    needed = SAMPLES_PER_LABEL - len(bucket)
                    bucket.extend(
                        group[text_col].dropna().astype(str).head(needed).tolist()
                    )
        for col in string_cols:
            if col in abandoned_string_cols:
                continue
            bucket = string_value_counts[col]
            for val, count in chunk[col].value_counts(dropna=False).items():
                bucket[val] = bucket.get(val, 0) + int(count)
            if len(bucket) > STREAM_DISTINCT_LIMIT:
                abandoned_string_cols.add(col)
                string_value_counts.pop(col, None)
        if i % 10 == 0:
            logging.debug(f"    chunk {i}: cumulative rows={total_rows:,}")

    elapsed = (datetime.now() - t0).total_seconds()
    logging.info(f"  Stream complete in {elapsed:.2f}s — total rows: {total_rows:,}")

    logging.debug(f"  NaN counts per column (across full stream):")
    for col, count in nan_per_col.items():
        pct = 100 * count / max(total_rows, 1)
        logging.debug(f"    - {col}: {count:,} ({pct:.2f}%)")

    logging.debug(
        f"  Within-chunk duplicate rows: {dup_within_chunk:,} "
        "(NOTE: cross-chunk duplicates are not detected in streaming mode)"
    )

    for col, bucket in string_value_counts.items():
        if 0 < len(bucket) <= LOW_CARDINALITY_THRESHOLD:
            logging.debug(f"  Value distribution for {col!r} ({len(bucket)} unique):")
            sorted_items = sorted(bucket.items(), key=lambda kv: -kv[1])
            for val, count in sorted_items:
                pct = 100 * count / max(total_rows, 1)
                logging.debug(f"    {val!r}: {count:,} ({pct:.2f}%)")
    if abandoned_string_cols:
        logging.debug(
            f"  String columns with > {STREAM_DISTINCT_LIMIT} distinct values "
            f"(distribution not tracked): {sorted(abandoned_string_cols)}"
        )

    if label_col and label_counts:
        logging.info(f"  Label distribution ('{label_col}'):")
        for val in sorted(label_counts.keys(), key=lambda v: (v is None, str(v))):
            count = label_counts[val]
            pct = 100 * count / max(total_rows, 1)
            logging.info(f"    {val!r}: {count:,} ({pct:.2f}%)")

    if samples_per_label and text_col:
        log_label_samples(samples_per_label, label_col, text_col)
    elif text_col and not label_col:
        logging.info(f"  No label column — head sample of {text_col!r} shown above.")

    logging.info(f"  First {PEEK_ROWS} rows:")
    for line in head.to_string(max_colwidth=80).splitlines():
        logging.info(f"    {line}")


def explore_file(path: Path) -> None:
    size_mb = file_size_mb(path)
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")

    logging.info("")
    logging.info(f"=== {path.name} ===")
    logging.info(f"  Path: {path.resolve()}")
    logging.info(f"  Size: {size_mb:,.2f} MB ({path.stat().st_size:,} bytes)")
    logging.info(f"  Modified: {mtime}")

    try:
        encoding = detect_encoding(path)
    except UnicodeDecodeError as e:
        logging.error(f"  Could not detect encoding: {e}")
        return
    logging.info(f"  Encoding: {encoding}")

    mode = "LARGE (streamed)" if size_mb >= LARGE_FILE_MB else "small (full read)"
    logging.info(f"  Mode: {mode}")

    try:
        if size_mb >= LARGE_FILE_MB:
            explore_large_file(path, encoding)
        else:
            explore_small_file(path, encoding)
    except Exception as e:
        logging.exception(f"  Failed to explore {path.name}: {e}")


def main() -> None:
    log_path = setup_logging()
    logging.info("=" * 70)
    logging.info("Raw dataset exploration — run started")
    logging.info(f"Log file: {log_path.resolve()}")
    logging.info(f"Working directory: {Path.cwd()}")
    logging.info("=" * 70)

    if not RAW_DIR.exists():
        logging.error(f"Raw directory not found: {RAW_DIR.resolve()}")
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR.resolve()}")

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in {RAW_DIR.resolve()}")
        return

    logging.info(f"Found {len(csv_files)} CSV file(s) in {RAW_DIR.resolve()}:")
    for p in csv_files:
        logging.info(f"  - {p.name}")

    t0 = datetime.now()
    for path in csv_files:
        explore_file(path)
    total_elapsed = (datetime.now() - t0).total_seconds()

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"Exploration complete — total elapsed: {total_elapsed:.2f}s")
    logging.info(f"Full log: {log_path.resolve()}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
