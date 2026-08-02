# Fixture builder — miniature replica of compiled_dataset.csv for validation
#
# Produces a small, committed dataset with the exact schema of the compiled
# corpus, used to validate every pipeline script on its REAL code path before
# spending Colab credits (see docs/RUNBOOK.md and D0.12). The fixture is
# engineered, not random — it guarantees the properties that break code:
#
#   - both labels present in every split (metrics/undersampling never crash)
#   - every (source, label) stratum represented when available
#   - queries longer than 512 tokens included per split, so the
#     sliding-window code actually executes during training AND evaluation
#
# Sampling is approximate (chunk-streamed with periodic downsampling) — the
# fixture is for pipeline validation only, never for reporting statistics.
#
# Run from the repo root: python code/datasets/build_fixture.py

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

SOURCE_CSV   = "code/datasets/compiled/compiled_dataset.csv"
FIXTURE_CSV  = "code/datasets/fixture/fixture_dataset.csv"

# Target rows per split, per label (benign≈70% / malicious≈30%, echoing the corpus).
# Deliberately tiny (~100 rows total): the fixture is a plumbing check — it proves
# every script's real code path runs end-to-end, not that metrics are meaningful.
# Forced stratum/long-query rows may push splits slightly above these targets.
TARGETS = {
    "train": {0: 40, 1: 20},
    "val":   {0: 14, 1: 6},
    "test":  {0: 14, 1: 6},
}

ROWS_PER_STRATUM   = 1      # forced rows per (split, source, label) when available
N_LONG_PER_SPLIT   = 2      # forced long queries (>512 tokens) per split
LONG_CANDIDATES    = 20     # longest queries tracked per split as candidates
ACCUM_FACTOR       = 5      # accumulator keeps at most ACCUM_FACTOR x target rows

TOKENIZER_NAME     = "microsoft/codebert-base"  # used only to verify long queries
MIN_CONTENT_TOKENS = 511    # "long" = needs more than one 512-token window

CHUNK_SIZE   = 100_000
RANDOM_SEED  = 42
LOG_DIR      = Path("code/datasets/logs")

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"build_fixture_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)
    logging.info(f"Log file: {log_path}")


def stream_and_collect():
    """Single pass over the source CSV collecting three structures per split:

    accumulators  — (split, label) -> DataFrame, periodically downsampled to
                    ACCUM_FACTOR x target so memory stays flat
    stratum_rows  — (split, source, label) -> first ROWS_PER_STRATUM rows,
                    never downsampled (protects rare sources)
    long_rows     — split -> top LONG_CANDIDATES rows by query length
    """
    logging.info(f"Streaming {SOURCE_CSV} ...")
    accumulators = {}
    stratum_rows = {}
    long_rows    = {}
    schema       = None
    total_seen   = 0

    reader = pd.read_csv(SOURCE_CSV, chunksize=CHUNK_SIZE, dtype=str)
    for chunk_idx, chunk in enumerate(reader, 1):
        if schema is None:
            schema = list(chunk.columns)
        chunk = chunk.dropna(subset=["query", "label", "split"])
        total_seen += len(chunk)

        for split in TARGETS:
            split_chunk = chunk[chunk["split"] == split]
            if split_chunk.empty:
                continue

            # Track the longest queries per split
            candidates = pd.concat([long_rows.get(split, split_chunk.iloc[0:0]), split_chunk])
            long_rows[split] = candidates.loc[
                candidates["query"].str.len().nlargest(LONG_CANDIDATES).index
            ]

            for label in (0, 1):
                label_chunk = split_chunk[split_chunk["label"].astype(float).astype(int) == label]
                if label_chunk.empty:
                    continue

                # Forced stratum coverage: first ROWS_PER_STRATUM rows per source
                for source, source_rows in label_chunk.groupby("source"):
                    key = (split, source, label)
                    have = len(stratum_rows.get(key, []))
                    if have < ROWS_PER_STRATUM:
                        take = source_rows.head(ROWS_PER_STRATUM - have)
                        stratum_rows[key] = pd.concat([stratum_rows.get(key, take.iloc[0:0]), take])

                # Accumulate + downsample to keep memory flat
                key = (split, label)
                cap = TARGETS[split][label] * ACCUM_FACTOR
                acc = pd.concat([accumulators.get(key, label_chunk.iloc[0:0]), label_chunk])
                if len(acc) > cap:
                    acc = acc.sample(n=cap, random_state=RANDOM_SEED + chunk_idx)
                accumulators[key] = acc

        if chunk_idx % 20 == 0:
            logging.info(f"  ... {chunk_idx} chunks ({total_seen:,} rows seen)")

    logging.info(f"Stream complete: {total_seen:,} rows seen, schema has {len(schema)} columns")
    return accumulators, stratum_rows, long_rows, schema


def verify_long_queries(long_df):
    """Return only rows whose query exceeds one sliding window, verified with
    the real tokenizer when available (character-length fallback otherwise)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        token_counts = [
            len(tokenizer.encode(q, add_special_tokens=False)) for q in long_df["query"]
        ]
        long_df = long_df.assign(_tokens=token_counts)
        verified = long_df[long_df["_tokens"] > MIN_CONTENT_TOKENS].drop(columns="_tokens")
        logging.info(f"  Tokenizer-verified long queries: {len(verified)}/{len(long_df)} candidates")
        return verified
    except Exception as e:
        logging.warning(f"  Tokenizer unavailable ({e}) — falling back to >2500-char proxy")
        return long_df[long_df["query"].str.len() > 2500]


def build_split(split, accumulators, stratum_rows, long_rows):
    """Assemble one split: forced stratum rows + verified long rows + random fill."""
    parts = []

    # 1. Forced (source, label) coverage
    for (s, source, label), rows in stratum_rows.items():
        if s == split:
            parts.append(rows)

    # 2. Forced long queries (sliding-window trigger)
    verified_long = verify_long_queries(long_rows.get(split, pd.DataFrame(columns=["query"])))
    n_long = min(N_LONG_PER_SPLIT, len(verified_long))
    if n_long > 0:
        parts.append(verified_long.head(n_long))
    if n_long < N_LONG_PER_SPLIT:
        logging.warning(
            f"  [{split}] only {n_long}/{N_LONG_PER_SPLIT} queries exceed one window — "
            "sliding-window coverage reduced for this split"
        )

    # 3. Random fill from the accumulators up to the per-label targets
    forced = pd.concat(parts).drop_duplicates() if parts else pd.DataFrame()
    for label in (0, 1):
        target  = TARGETS[split][label]
        n_have  = 0
        if not forced.empty:
            n_have = int((forced["label"].astype(float).astype(int) == label).sum())
        need = max(0, target - n_have)
        acc  = accumulators.get((split, label))
        if acc is None or acc.empty:
            raise ValueError(f"[{split}] no rows found for label={label} — source CSV problem?")
        fill = acc.sample(n=min(need, len(acc)), random_state=RANDOM_SEED)
        parts.append(fill)

    result = (
        pd.concat(parts)
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    n_benign    = int((result["label"].astype(float).astype(int) == 0).sum())
    n_malicious = int((result["label"].astype(float).astype(int) == 1).sum())
    n_sources   = result["source"].nunique()
    logging.info(
        f"  [{split}] {len(result):,} rows — benign={n_benign:,}  malicious={n_malicious:,}"
        f"  sources={n_sources}  long_queries={n_long}"
    )

    if n_benign == 0 or n_malicious == 0:
        raise ValueError(f"[{split}] single-class split — fixture invalid")
    return result


def main():
    setup_logging()
    logging.info("Building validation fixture")
    logging.info(f"Source:  {SOURCE_CSV}")
    logging.info(f"Output:  {FIXTURE_CSV}")
    logging.debug(f"Targets: {TARGETS}")

    if not Path(SOURCE_CSV).exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_CSV}. Run from the repo root.")

    accumulators, stratum_rows, long_rows, schema = stream_and_collect()

    splits = [build_split(s, accumulators, stratum_rows, long_rows) for s in TARGETS]
    fixture = pd.concat(splits, ignore_index=True)[schema]  # restore exact column order

    out_path = Path(FIXTURE_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(out_path, index=False)

    size_kb = out_path.stat().st_size / 1024
    logging.info(f"Fixture written: {len(fixture):,} rows, {len(schema)} cols, {size_kb:,.0f} KB → {out_path}")
    logging.debug(f"Split distribution:\n{fixture['split'].value_counts().to_string()}")
    logging.debug(f"Source distribution:\n{fixture['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
