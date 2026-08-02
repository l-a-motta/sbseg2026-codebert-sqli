# Stage 1 — Dataset Compilation
#
# Reads each raw CSV under datasets/raw/, normalises to a unified schema,
# deduplicates by query text across sources, drops invalid rows, and writes
# two output files:
#   - compiled_dataset.csv   : 17-col BERT corpus (text + label + metadata)
#   - compiled_features.csv  : 19-col ML-baseline corpus (from SQLI_Dataset.csv)
#
# Pipeline (see docs/decisions.md, D1.5–D1.19):
#   1. Pass 1 — per-source adapters write rows to a temp file with split=NaN.
#              Tracks (source, label) bucket counts.
#   2. Pass 2 — generate shuffled split labels per bucket, stream the temp
#              file, attach split column, write the final compiled CSV.
#   3. Features — reshape SQLI_Dataset.csv → compiled_features.csv.
#   4. Validation — read both outputs back and log shape, columns, label/split
#                   distributions.
#
# Large files (dataset.csv ~1 GB, rbsqli_dataset.csv ~1.3 GB) are streamed
# in chunks. The raw files are never modified.

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

RAW_DIR = Path("code/datasets/raw")
OUTPUT_DIR = Path("code/datasets/compiled")
LOG_DIR = Path("code/compilation/logs")

COMPILED_DATASET_FILE = "compiled_dataset.csv"
COMPILED_FEATURES_FILE = "compiled_features.csv"
TEMP_PASS1_FILE = "_compiled_dataset_pass1.csv"

# Reproducibility (D1.16).
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

# Drop rows whose query is shorter than this (D1.18).
MIN_QUERY_LENGTH = 2

CHUNK_SIZE = 100_000
ENCODINGS = ["utf-8", "utf-16", "latin-1"]

# Final BERT schema, in this exact column order (17 cols).
BERT_COLUMNS = [
    "query", "label", "source", "needs_hand_label", "split",
    # dataset.csv metadata (D1.7)
    "user_inputs", "attack_stage", "tamper_method", "attack_status",
    "statement_type", "query_template_id", "attack_id", "attack_technique",
    # rbsqli_dataset.csv metadata (D1.19)
    "injection_type", "vulnerability_status", "sql_command", "target_table",
]

# Cleaned column names for compiled_features.csv (the SQLI_Dataset.csv has
# inconsistent spaces and abbreviations).
FEATURES_RENAME = {
    "no_special _char": "no_special_char",
    "no_single_ qout": "no_single_quote",
    "no_double_ qout": "no_double_quote",
    "whit_ space": "white_space",
    "no_logical _operat": "no_logical_operator",
    "no_operat": "no_operator",
    "no_null Value in the SQL Query": "no_null_value",
}

# Sentinel used in dict keys when label is NaN, so the (source, label) tuple
# remains hashable and equal across uses.
NAN_LABEL_KEY = "__NAN__"

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"compile_{timestamp}.log"

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


def detect_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, nrows=1, encoding=enc, low_memory=False)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
        except pd.errors.ParserError:
            return enc
    raise UnicodeDecodeError(
        "compile", b"", 0, 1, f"None of {ENCODINGS} could decode {path.name}"
    )


def safe_read_csv(path: Path, encoding: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
    except pd.errors.ParserError as e:
        logging.warning(
            f"  C-engine ParserError ({e}). Retrying with python engine."
        )
        kwargs.pop("low_memory", None)
        return pd.read_csv(path, encoding=encoding, engine="python", **kwargs)


def label_key(label) -> object:
    """Return a hashable bucket key for label, replacing NaN with a sentinel."""
    return NAN_LABEL_KEY if pd.isna(label) else label


def fill_bert_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has every BERT_COLUMNS column (filling missing with None) and
    return df reordered to BERT_COLUMNS."""
    for col in BERT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[BERT_COLUMNS]


# ── Per-source adapters ──────────────────────────────────────────────────────
#
# Each adapter is a generator yielding DataFrames already in BERT_COLUMNS
# order. Streamed sources (dataset.csv, rbsqli_dataset.csv) yield one chunk
# at a time; small sources yield a single full DataFrame.


def adapt_dataset_csv(path: Path) -> Iterator[pd.DataFrame]:
    encoding = detect_encoding(path)
    logging.info(f"  Streaming with encoding={encoding!r}, chunksize={CHUNK_SIZE:,}")
    reader = pd.read_csv(path, chunksize=CHUNK_SIZE, encoding=encoding, low_memory=False)
    for chunk in reader:
        out = pd.DataFrame({
            "query": chunk["full_query"],
            "label": chunk["label"],
            "source": "dataset",
            "needs_hand_label": chunk["label"].isna(),
            "user_inputs": chunk["user_inputs"],
            "attack_stage": chunk["attack_stage"],
            "tamper_method": chunk["tamper_method"],
            "attack_status": chunk["attack_status"],
            "statement_type": chunk["statement_type"],
            "query_template_id": chunk["query_template_id"],
            "attack_id": chunk["attack_id"],
            "attack_technique": chunk["attack_technique"],
        })
        yield fill_bert_schema(out)


def adapt_rbsqli(path: Path) -> Iterator[pd.DataFrame]:
    encoding = detect_encoding(path)
    logging.info(f"  Streaming with encoding={encoding!r}, chunksize={CHUNK_SIZE:,}")
    reader = pd.read_csv(path, chunksize=CHUNK_SIZE, encoding=encoding, low_memory=False)
    for chunk in reader:
        # D1.11: label = (injection_type != "None_Type")
        labels = (chunk["injection_type"] != "None_Type").astype(int)
        out = pd.DataFrame({
            "query": chunk["sql_query"],
            "label": labels,
            "source": "rbsqli",
            "needs_hand_label": False,
            "injection_type": chunk["injection_type"],
            "vulnerability_status": chunk["vulnerability_status"],
            "sql_command": chunk["sql_command"],
            "target_table": chunk["target_table"],
        })
        yield fill_bert_schema(out)


def adapt_bccc(path: Path) -> Iterator[pd.DataFrame]:
    encoding = detect_encoding(path)
    logging.info(f"  Reading full file with encoding={encoding!r}")
    df = safe_read_csv(path, encoding=encoding)
    out = pd.DataFrame({
        "query": df["Data"],
        "label": 1,  # D1.6: BCCC is attack-only.
        "source": "bccc",
        "needs_hand_label": False,
    })
    yield fill_bert_schema(out)


def adapt_query_label_pair(
    path: Path, source: str, query_col: str, label_col: str
) -> Iterator[pd.DataFrame]:
    """Generic adapter for files with a single text col and a single label col."""
    encoding = detect_encoding(path)
    logging.info(f"  Reading full file with encoding={encoding!r}")
    df = safe_read_csv(path, encoding=encoding)
    label_series = df[label_col]
    out = pd.DataFrame({
        "query": df[query_col],
        "label": label_series,
        "source": source,
        "needs_hand_label": label_series.isna(),
    })
    yield fill_bert_schema(out)


# Adapter dispatch in dedup-priority order (D1.17): the first source to
# emit a given query keeps its row; later sources lose their copy.
# Wave 2 sources (D1.20) are appended last so wave 1 always wins on collisions.
ADAPTERS = [
    # ── Wave 1 ───────────────────────────────────────────────────────────────
    ("dataset",       "DS02_dataset.csv",                lambda p: adapt_dataset_csv(p)),
    ("rbsqli",        "DS01_rbsqli_dataset.csv",         lambda p: adapt_rbsqli(p)),
    ("bccc",          "DS04_BCCC-SFU-SQLInj-2023.csv",   lambda p: adapt_bccc(p)),
    ("train",         "DS03_Train.csv",                  lambda p: adapt_query_label_pair(p, "train", "Query", "Label")),
    ("test",          "DS03_Test.csv",                   lambda p: adapt_query_label_pair(p, "test", "Query", "Label")),
    ("validation",    "DS03_Validation.csv",             lambda p: adapt_query_label_pair(p, "validation", "Query", "Label")),
    ("modified_sql",  "DS06_Modified_SQL_Dataset.csv",   lambda p: adapt_query_label_pair(p, "modified_sql", "Query", "Label")),
    ("sqliv2",        "DS07_sqliv2.csv",                 lambda p: adapt_query_label_pair(p, "sqliv2", "Sentence", "Label")),
    ("sqli",          "DS08_sqli.csv",                   lambda p: adapt_query_label_pair(p, "sqli", "Sentence", "Label")),
    # ── Wave 2 (D1.20) ───────────────────────────────────────────────────────
    ("clean_sql",     "DS09_clean_sql_dataset.csv",      lambda p: adapt_query_label_pair(p, "clean_sql", "Query", "Label")),
    ("sqli_extended", "DS10_sqli-extended.csv",          lambda p: adapt_query_label_pair(p, "sqli_extended", "Sentence", "Label")),
    ("testingdata",   "DS11_testingdata.csv",            lambda p: adapt_query_label_pair(p, "testingdata", "Query", "Label")),
    ("trainingdata",  "DS12_trainingdata.csv",           lambda p: adapt_query_label_pair(p, "trainingdata", "Query", "Label")),
    ("sqliv3",        "DS13_SQLiV3.csv",                 lambda p: adapt_query_label_pair(p, "sqliv3", "Sentence", "Label")),
]


# ── Pass 1: read sources, dedup, drop, write temp ─────────────────────────────


def filter_and_dedupe(
    chunk: pd.DataFrame, seen_hashes: set
) -> tuple[pd.DataFrame, dict]:
    """Drop NaN/short queries, drop cross-source duplicates."""
    n_in = len(chunk)

    # Drop NaN queries (D1.18).
    nan_mask = chunk["query"].isna()
    chunk = chunk.loc[~nan_mask].copy()
    n_drop_nan = int(nan_mask.sum())

    # Drop short queries (D1.18).
    str_queries = chunk["query"].astype(str)
    short_mask = str_queries.str.len() < MIN_QUERY_LENGTH
    chunk = chunk.loc[~short_mask].copy()
    n_drop_short = int(short_mask.sum())

    # Cross-source dedup by sha1(query) (D1.17).
    str_queries = chunk["query"].astype(str)
    keep = []
    for q in str_queries:
        h = hashlib.sha1(q.encode("utf-8", errors="replace")).digest()
        if h in seen_hashes:
            keep.append(False)
        else:
            seen_hashes.add(h)
            keep.append(True)
    n_drop_dup = sum(1 for k in keep if not k)
    chunk = chunk.loc[keep].copy()

    return chunk, {
        "in": n_in,
        "drop_nan": n_drop_nan,
        "drop_short": n_drop_short,
        "drop_dup": n_drop_dup,
        "out": len(chunk),
    }


def normalise_label_int(chunk: pd.DataFrame) -> pd.DataFrame:
    """Cast non-NaN labels to int; leave NaN labels as NaN (float)."""
    label = chunk["label"]
    mask = label.notna()
    if mask.any():
        chunk.loc[mask, "label"] = label[mask].astype(int)
    return chunk


def pass1_write_temp() -> tuple[Path, dict]:
    """Write all sources to the temp CSV. Return (path, bucket_counts)."""
    temp_path = OUTPUT_DIR / TEMP_PASS1_FILE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        logging.info(f"Removing pre-existing temp file: {temp_path}")
        temp_path.unlink()

    seen_hashes: set = set()
    bucket_counts: dict = {}  # (source, label_key) -> count
    header_written = False
    grand = {"in": 0, "drop_nan": 0, "drop_short": 0, "drop_dup": 0, "out": 0}

    for source_key, filename, factory in ADAPTERS:
        path = RAW_DIR / filename
        logging.info("")
        logging.info(f"--- Source: {source_key} ({filename}) ---")
        if not path.exists():
            logging.warning(f"  File not found: {path}; skipping.")
            continue

        t0 = datetime.now()
        src_stats = {"in": 0, "drop_nan": 0, "drop_short": 0, "drop_dup": 0, "out": 0}
        src_label_counts: dict = {}

        for chunk in factory(path):
            filtered, stats = filter_and_dedupe(chunk, seen_hashes)
            for k, v in stats.items():
                src_stats[k] += v

            if len(filtered) == 0:
                continue

            filtered = normalise_label_int(filtered)
            for label_val, n in filtered["label"].value_counts(dropna=False).items():
                src_label_counts[label_val] = src_label_counts.get(label_val, 0) + int(n)
                key = (source_key, label_key(label_val))
                bucket_counts[key] = bucket_counts.get(key, 0) + int(n)

            filtered = filtered[BERT_COLUMNS]
            filtered.to_csv(temp_path, mode="a", index=False, header=not header_written)
            header_written = True

        elapsed = (datetime.now() - t0).total_seconds()
        for k in grand:
            grand[k] += src_stats[k]
        logging.info(
            f"  in={src_stats['in']:,}  drop_nan={src_stats['drop_nan']:,}  "
            f"drop_short={src_stats['drop_short']:,}  drop_dup={src_stats['drop_dup']:,}  "
            f"out={src_stats['out']:,}"
        )
        logging.info(f"  Label distribution after dedup:")
        for val in sorted(src_label_counts.keys(), key=lambda v: (pd.isna(v), str(v))):
            n = src_label_counts[val]
            pct = 100 * n / max(src_stats["out"], 1)
            logging.info(f"    {val!r}: {n:,} ({pct:.2f}%)")
        logging.info(f"  Elapsed: {elapsed:.2f}s")

    logging.info("")
    logging.info("--- Pass 1 totals ---")
    logging.info(
        f"  in={grand['in']:,}  drop_nan={grand['drop_nan']:,}  "
        f"drop_short={grand['drop_short']:,}  drop_dup={grand['drop_dup']:,}  "
        f"out={grand['out']:,}"
    )
    logging.info(f"  Distinct (source, label) buckets: {len(bucket_counts)}")
    logging.debug(f"  Bucket counts:")
    for (s, l), n in sorted(bucket_counts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        logging.debug(f"    ({s!r}, {l!r}): {n:,}")
    logging.info(f"  Temp file: {temp_path} ({temp_path.stat().st_size / (1024*1024):,.2f} MB)")

    return temp_path, bucket_counts


# ── Pass 2: assign stratified splits ─────────────────────────────────────────


def build_bucket_split_iterators(bucket_counts: dict, rng: np.random.Generator) -> dict:
    """For each bucket, generate a shuffled list of split labels and return as iterators.
    Buckets whose label_key is NAN_LABEL_KEY (i.e., needs_hand_label rows) get all NaN
    splits — those rows are not in any partition until labelled."""
    out: dict = {}
    train_pct, val_pct, _test_pct = SPLIT_RATIOS
    for (source, lkey), n in bucket_counts.items():
        if lkey == NAN_LABEL_KEY:
            labels = [None] * n  # leave split=NaN for hand-label rows
        else:
            n_train = int(round(n * train_pct))
            n_val = int(round(n * val_pct))
            n_test = n - n_train - n_val
            labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
            rng.shuffle(labels)
        out[(source, lkey)] = iter(labels)
    return out


def pass2_assign_splits(temp_path: Path, bucket_counts: dict) -> Path:
    final_path = OUTPUT_DIR / COMPILED_DATASET_FILE
    if final_path.exists():
        logging.info(f"Removing pre-existing final file: {final_path}")
        final_path.unlink()

    logging.info("")
    logging.info(f"--- Pass 2: stratified split assignment ---")
    rng = np.random.default_rng(RANDOM_SEED)
    bucket_iters = build_bucket_split_iterators(bucket_counts, rng)

    t0 = datetime.now()
    header_written = False
    split_totals: dict = {}
    rows_written = 0

    reader = pd.read_csv(temp_path, chunksize=CHUNK_SIZE, encoding="utf-8", low_memory=False)
    for chunk in reader:
        # Assign splits per (source, label) bucket, preserving original row order.
        chunk["split"] = None
        for (source, label_val), group in chunk.groupby(
            ["source", "label"], dropna=False, sort=False
        ):
            key = (source, label_key(label_val))
            it = bucket_iters[key]
            n_group = len(group)
            assigned = [next(it) for _ in range(n_group)]
            chunk.loc[group.index, "split"] = assigned

        chunk = chunk[BERT_COLUMNS]
        chunk.to_csv(final_path, mode="a", index=False, header=not header_written)
        header_written = True
        rows_written += len(chunk)

        for val, n in chunk["split"].value_counts(dropna=False).items():
            split_totals[val] = split_totals.get(val, 0) + int(n)

    elapsed = (datetime.now() - t0).total_seconds()
    logging.info(f"  Rows written: {rows_written:,}")
    logging.info(f"  Final split distribution:")
    for val in sorted(split_totals.keys(), key=lambda v: (pd.isna(v), str(v))):
        n = split_totals[val]
        pct = 100 * n / max(rows_written, 1)
        logging.info(f"    {val!r}: {n:,} ({pct:.2f}%)")
    logging.info(f"  Final file: {final_path} ({final_path.stat().st_size / (1024*1024):,.2f} MB)")
    logging.info(f"  Elapsed: {elapsed:.2f}s")

    # Drop the temp file.
    try:
        temp_path.unlink()
        logging.info(f"  Removed temp file: {temp_path}")
    except OSError as e:
        logging.warning(f"  Could not remove temp file {temp_path}: {e}")

    return final_path


# ── Features pipeline (SQLI_Dataset.csv → compiled_features.csv) ─────────────


def compile_features() -> Path | None:
    path = RAW_DIR / "DS05_SQLI_Dataset.csv"
    out_path = OUTPUT_DIR / COMPILED_FEATURES_FILE
    logging.info("")
    logging.info(f"--- SQLI_Dataset.csv -> {out_path.name} ---")
    if not path.exists():
        logging.warning(f"  File not found: {path}; skipping.")
        return None

    t0 = datetime.now()
    encoding = detect_encoding(path)
    logging.info(f"  Reading full file with encoding={encoding!r}")
    df = safe_read_csv(path, encoding=encoding)
    df = df.rename(columns=FEATURES_RENAME)
    df.to_csv(out_path, index=False)
    elapsed = (datetime.now() - t0).total_seconds()

    logging.info(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")
    logging.info(f"  Columns: {list(df.columns)}")
    label_dist = df["label"].value_counts(dropna=False).to_dict()
    logging.info(f"  Label distribution: {label_dist}")
    logging.info(f"  File: {out_path} ({out_path.stat().st_size / (1024*1024):,.2f} MB)")
    logging.info(f"  Elapsed: {elapsed:.2f}s")
    return out_path


# ── Validation ───────────────────────────────────────────────────────────────


def validate_outputs() -> None:
    logging.info("")
    logging.info("--- Validation pass ---")
    for filename, label in [
        (COMPILED_DATASET_FILE, "BERT corpus"),
        (COMPILED_FEATURES_FILE, "Features corpus"),
    ]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            logging.warning(f"  Missing: {path}")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        head = pd.read_csv(path, nrows=5, low_memory=False)
        total = 0
        for ch in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            total += len(ch)
        logging.info(f"  {label}: {path}")
        logging.info(f"    size: {size_mb:,.2f} MB  rows: {total:,}  cols: {head.shape[1]}")
        logging.info(f"    columns: {list(head.columns)}")
        logging.info(f"    head:")
        for line in head.to_string(max_colwidth=60).splitlines():
            logging.info(f"      {line}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    log_path = setup_logging()
    logging.info("=" * 70)
    logging.info("Stage 1 — Dataset Compilation")
    logging.info(f"Log file:       {log_path}")
    logging.info(f"Working dir:    {Path.cwd()}")
    logging.info(f"Random seed:    {RANDOM_SEED}")
    logging.info(f"Split ratios:   {SPLIT_RATIOS}")
    logging.info(f"Min query len:  {MIN_QUERY_LENGTH}")
    logging.info(f"Chunk size:     {CHUNK_SIZE:,}")
    logging.info("=" * 70)

    t_overall = datetime.now()
    temp_path, bucket_counts = pass1_write_temp()
    pass2_assign_splits(temp_path, bucket_counts)
    compile_features()
    validate_outputs()
    elapsed = (datetime.now() - t_overall).total_seconds()

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"Compilation complete — total elapsed: {elapsed:.2f}s")
    logging.info(f"Full log: {log_path}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
