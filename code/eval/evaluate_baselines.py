# Stage 3 — ML Baseline Evaluation (PP5)
#
# Trains three hand-crafted-feature ML models on the compiled dataset's train
# split and evaluates them on the same test split used by evaluate.py. This
# makes the PP5 comparison between BERT and traditional ML scientifically fair:
# all models see identical training and test data.
#
# Models:
#   logistic_regression  — fast linear baseline
#   random_forest        — ensemble baseline
#   svm_rbf              — kernel SVM; viable at MAX_TRAIN_ROWS <= 500K
#
# Features: 18 hand-crafted features matching the SQLI_Dataset.csv schema (D3.5).
# Output: per-row prediction CSVs and rows appended to evaluation_results.csv.
#
# The configuration constants below are defaults; every one of them can be
# overridden from the command line without editing this file (see --help):
#   python code/eval/evaluate_baselines.py --dataset-path <csv> --results-dir <dir> \
#       --max-train-rows <n|all> --seed <n>
# For a quick end-to-end validation, pass the committed fixture
# (--dataset-path code/datasets/fixture/fixture_dataset.csv) and a scratch
# --results-dir — same code path, miniature data. See docs/RUNBOOK.md.

import argparse
import logging
import math
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_PATH     = "code/datasets/compiled/compiled_dataset.csv"
RESULTS_DIR      = Path("code/eval/results")
CHUNK_SIZE       = 100_000
RANDOM_SEED      = 42
MAX_TRAIN_ROWS   = 100_000   # cap training rows (None = all rows)
                              # Keep at 100K to match Stage A BERT training size.
                              # SVM with RBF becomes impractical above ~500K rows.

LOG_DIR          = Path("code/eval/logs")

# SQL keywords for no_keyword and no_logical_operator features (D3.5)
_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION|WHERE|FROM|HAVING"
    r"|ORDER|GROUP|JOIN|EXEC|EXECUTE|CAST|CONVERT|DECLARE|SLEEP|BENCHMARK"
    r"|WAITFOR|AND|OR)\b",
    re.IGNORECASE,
)
_LOGICAL_OP   = re.compile(r"\b(AND|OR|NOT)\b", re.IGNORECASE)
_OPERATOR     = re.compile(r"(!=|<>|<=|>=|=|<|>)")
_COMMENT      = re.compile(r"(--|/\*)")
_NULL         = re.compile(r"\bNULL\b", re.IGNORECASE)

# ── Command-line overrides ────────────────────────────────────────────────────
# The constants above are the defaults used in the article; each flag below
# overrides one of them for a single run, so no source edit is ever required.


def parse_rows_cap(value):
    """argparse type: a positive integer, or 'all'/'none' meaning no cap."""
    if str(value).lower() in ("all", "none"):
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 — ML baseline evaluation "
                    "(Logistic Regression, Random Forest, Linear SVM).",
    )
    parser.add_argument("--dataset-path", default=DATASET_PATH,
                        help=f"compiled dataset CSV (default: {DATASET_PATH})")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help=f"output directory for prediction CSVs and "
                             f"evaluation_results.csv (default: {RESULTS_DIR})")
    parser.add_argument("--max-train-rows", default=MAX_TRAIN_ROWS, type=parse_rows_cap,
                        help="cap on training rows, or 'all' for no cap "
                             f"(default: {MAX_TRAIN_ROWS})")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int,
                        help=f"random seed (default: {RANDOM_SEED})")
    parser.add_argument("--log-dir", default=str(LOG_DIR),
                        help=f"directory for the DEBUG-level log file (default: {LOG_DIR})")
    return parser.parse_args()


def apply_args(args):
    """Overwrite the configuration globals with the parsed command-line values."""
    global DATASET_PATH, RESULTS_DIR, MAX_TRAIN_ROWS, RANDOM_SEED, LOG_DIR
    DATASET_PATH   = args.dataset_path
    RESULTS_DIR    = Path(args.results_dir)
    MAX_TRAIN_ROWS = args.max_train_rows
    RANDOM_SEED    = args.seed
    LOG_DIR        = Path(args.log_dir)

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"evaluate_baselines_{timestamp}.log"

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


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(text: str) -> dict:
    """Compute 18 hand-crafted features from raw query text (D3.5)."""
    text = str(text)
    return {
        "length":             len(text),
        "no_letter":          sum(c.isalpha()     for c in text),
        "no_digit":           sum(c.isdigit()     for c in text),
        "no_special_char":    sum(not c.isalnum() and not c.isspace() for c in text),
        "entropy":            _entropy(text),
        "no_keyword":         len(_SQL_KEYWORDS.findall(text)),
        "no_semicolon":       text.count(";"),
        "no_single_quote":    text.count("'"),
        "no_double_quote":    text.count('"'),
        "no_percentage":      text.count("%"),
        "white_space":        sum(c.isspace() for c in text),
        "no_punctuation":     sum(not c.isalnum() and not c.isspace() for c in text),
        "no_logical_operator":len(_LOGICAL_OP.findall(text)),
        "no_operator":        len(_OPERATOR.findall(text)),
        "no_or":              len(re.findall(r"\bOR\b",  text, re.IGNORECASE)),
        "no_and":             len(re.findall(r"\bAND\b", text, re.IGNORECASE)),
        "no_comment":         len(_COMMENT.findall(text)),
        "no_null_value":      len(_NULL.findall(text)),
    }


def _features_from_df(df: pd.DataFrame) -> np.ndarray:
    """Apply extract_features row-by-row and return a float32 feature matrix."""
    rows = [extract_features(q) for q in df["query"]]
    return pd.DataFrame(rows).astype(np.float32).values


def _undersample(df: pd.DataFrame, ratio: float = 2.0) -> pd.DataFrame:
    """Randomly drop benign rows until benign:malicious == ratio."""
    n_malicious     = int((df["label"] == 1).sum())
    n_benign_target = int(n_malicious * ratio)
    benign          = df[df["label"] == 0]
    malicious       = df[df["label"] == 1]
    if len(benign) <= n_benign_target:
        return df
    benign_sampled = benign.sample(n=n_benign_target, random_state=RANDOM_SEED)
    return (
        pd.concat([benign_sampled, malicious], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


def load_split(split_name: str, max_rows: int | None = None) -> pd.DataFrame:
    logging.info(f"Loading split='{split_name}'" + (f" (cap={max_rows:,})" if max_rows else "") + " ...")
    reader = pd.read_csv(
        DATASET_PATH,
        usecols=["query", "label", "split", "source", "injection_type"],
        chunksize=CHUNK_SIZE,
        dtype={"query": str, "split": str, "source": str, "injection_type": str},
    )
    chunks = []
    for chunk in reader:
        chunk = chunk[chunk["split"] == split_name].dropna(subset=["query", "label"])
        chunk["label"] = chunk["label"].astype(int)
        chunks.append(chunk[["query", "label", "source", "injection_type"]])
    df = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
        logging.info(f"  Capped to {max_rows:,} rows")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    n_benign    = int((df["label"] == 0).sum())
    n_malicious = int((df["label"] == 1).sum())
    logging.info(f"  {len(df):,} rows — benign={n_benign:,}  malicious={n_malicious:,}")
    return df


def compute_metrics(labels, preds, probs, avg_loss=float("nan")):
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )
    return {
        "loss":      avg_loss,
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       roc_auc_score(labels, probs),
        "mcc":       matthews_corrcoef(labels, preds),
    }


def _build_summary_row(model_id, scope, n_samples, metrics, inference_time_s, energy_kwh):
    return {
        "model_id":         model_id,
        "scope":            scope,
        "n_samples":        n_samples,
        "mcc":              metrics.get("mcc",       float("nan")),
        "auc":              metrics.get("auc",       float("nan")),
        "accuracy":         metrics.get("accuracy",  float("nan")),
        "precision":        metrics.get("precision", float("nan")),
        "recall":           metrics.get("recall",    float("nan")),
        "f1":               metrics.get("f1",        float("nan")),
        "inference_time_s": inference_time_s,
        "energy_kwh":       energy_kwh if energy_kwh is not None else float("nan"),
    }


def append_to_results_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "evaluation_results.csv"
    df_new = pd.DataFrame(rows)
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(results_path, index=False)
    logging.info(f"  Appended {len(rows)} scope row(s) → {results_path}")


def evaluate_baseline(model_id, model, X_test, test_df):
    logging.info(f"\n{'─' * 60}")
    logging.info(f"Evaluating: {model_id}")

    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"sqli-eval-{model_id}",
            output_dir=str(LOG_DIR),
            log_level="error",
        )
        tracker.start()

    infer_start = time.perf_counter()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - infer_start

    energy_kwh = None
    if tracker is not None:
        emissions_kg = tracker.stop()
        energy_kwh   = tracker.final_emissions_data.energy_consumed
        logging.info(f"  Energy: {energy_kwh:.4f} kWh  CO2: {emissions_kg * 1000:.1f} gCO2")

    logging.info(f"  Inference time: {inference_time:.2f}s")

    preds_df = test_df.copy()
    preds_df["pred"]     = preds.astype(int)
    preds_df["prob"]     = probs
    preds_df["model_id"] = model_id

    out_path = RESULTS_DIR / f"{model_id}_predictions.csv"
    preds_df.to_csv(out_path, index=False)
    logging.info(f"  Per-row predictions → {out_path}")

    labels_list = test_df["label"].tolist()
    preds_list  = preds_df["pred"].tolist()
    probs_list  = probs.tolist()

    overall = compute_metrics(labels_list, preds_list, probs_list)
    logging.info(
        f"  PP1 [overall]  mcc={overall['mcc']:.4f}  auc={overall['auc']:.4f}"
        f"  f1={overall['f1']:.4f}  acc={overall['accuracy']:.4f}"
    )

    summary_rows = [
        _build_summary_row(model_id, "overall", len(test_df), overall, inference_time, energy_kwh)
    ]
    append_to_results_csv(summary_rows)


def main():
    setup_logging()
    logging.info("Stage 3 — ML Baseline Evaluation")
    logging.info(f"Dataset:       {DATASET_PATH}")
    logging.info(f"MAX_TRAIN_ROWS:{MAX_TRAIN_ROWS}")

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run from the repository root."
        )

    train_df = load_split("train", max_rows=MAX_TRAIN_ROWS)
    test_df  = load_split("test",  max_rows=None)

    # Balance training split (2:1 benign:malicious, same as D2.3)
    train_df = _undersample(train_df, ratio=2.0)
    n_benign    = int((train_df["label"] == 0).sum())
    n_malicious = int((train_df["label"] == 1).sum())
    logging.info(f"After undersample: train benign={n_benign:,}  malicious={n_malicious:,}")

    logging.info("Extracting features from train split ...")
    t0      = time.perf_counter()
    X_train = _features_from_df(train_df)
    y_train = train_df["label"].values
    logging.info(f"  Train feature matrix: {X_train.shape}  ({time.perf_counter()-t0:.1f}s)")

    logging.info("Extracting features from test split ...")
    t0     = time.perf_counter()
    X_test = _features_from_df(test_df)
    logging.info(f"  Test feature matrix:  {X_test.shape}  ({time.perf_counter()-t0:.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    logging.info("\nTraining Logistic Regression ...")
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    logging.info(f"  Training time: {time.perf_counter()-t0:.1f}s")
    evaluate_baseline("logistic_regression", lr, X_test, test_df)

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    logging.info("\nTraining Random Forest ...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    logging.info(f"  Training time: {time.perf_counter()-t0:.1f}s")
    evaluate_baseline("random_forest", rf, X_test, test_df)

    # ── Model 3: Linear SVM (with calibration for probability output) ─────────
    # LinearSVC is used instead of SVC(kernel='rbf') for scalability.
    # CalibratedClassifierCV adds isotonic regression calibration for predict_proba.
    logging.info("\nTraining Linear SVM (with calibration) ...")
    t0  = time.perf_counter()
    svm = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_SEED, max_iter=2000),
        cv=3,
    )
    svm.fit(X_train, y_train)
    logging.info(f"  Training time: {time.perf_counter()-t0:.1f}s")
    evaluate_baseline("svm_linear", svm, X_test, test_df)

    logging.info("\nAll baselines evaluated. Results in " + str(RESULTS_DIR))


if __name__ == "__main__":
    apply_args(parse_args())
    main()
