# Stage 3 — Evaluation: linear probes for the untuned encoders (PP6, D3.7)
#
# Scores bert-base and codebert-base WITHOUT task fine-tuning, the standard way
# (Alain & Bengio 2017; Peters et al. 2019): freeze the encoder, extract the
# [CLS] vector per query, train a logistic regression on a small training
# subsample, then grade it on the test split. The probe is too simple to add
# knowledge of its own — its score reflects what the pre-trained encoder
# already "knows" about SQLi before any task training.
#
# This replaces the old random-classification-head evaluation of these two
# models in evaluate.py, which always collapsed to MCC≈0 (finding F3.1a).
#
# Long queries are truncated to the first MAX_TOKEN_LENGTH tokens (no sliding
# window): F2.2 found essentially all queries fit a single window, and the
# probe measures representations, not the inference pipeline.
#
# Results are appended to the same evaluation_results.csv as evaluate.py, with
# model_id suffix "-probe" so the scoreboard distinguishes before/after course.
#
# The configuration constants below are defaults; every one of them can be
# overridden from the command line without editing this file (see --help).
# For a quick end-to-end validation, pass the committed fixture
# (--dataset-path code/datasets/fixture/fixture_dataset.csv), a scratch
# --results-dir, and a lower --probe-train-rows. See docs/RUNBOOK.md.

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Reuse data loading, metrics, and results-CSV helpers from the existing scripts
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "fine-tuning"))
import evaluate as _e  # noqa: E402  (also imports train as _t internally)
import train as _t     # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

ENCODERS = [
    ("bert-base-probe",     "bert-base-uncased"),
    ("codebert-base-probe", "microsoft/codebert-base"),
]

DATASET_PATH     = "code/datasets/compiled/compiled_dataset.csv"
RESULTS_DIR      = Path("code/eval/results")
PROBE_TRAIN_ROWS = 50_000   # training-split subsample used to fit the logistic regression
BATCH_SIZE       = 64
MAX_TOKEN_LENGTH = 512
RANDOM_SEED      = 42

LOG_INTERVAL_SECONDS = 150
LOG_DIR              = Path("code/eval/logs")

# ── Command-line overrides ────────────────────────────────────────────────────
# The constants above are the defaults used in the article; each flag below
# overrides one of them for a single run, so no source edit is ever required.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 — linear probes for the untuned encoders (PP6, D3.7).",
    )
    parser.add_argument("--dataset-path", default=DATASET_PATH,
                        help=f"compiled dataset CSV (default: {DATASET_PATH})")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help=f"output directory for prediction CSVs and "
                             f"evaluation_results.csv (default: {RESULTS_DIR})")
    parser.add_argument("--probe-train-rows", default=PROBE_TRAIN_ROWS, type=int,
                        help="training-split subsample used to fit the probe "
                             f"(default: {PROBE_TRAIN_ROWS})")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int,
                        help=f"embedding batch size (default: {BATCH_SIZE})")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int,
                        help=f"random seed (default: {RANDOM_SEED})")
    parser.add_argument("--log-dir", default=str(LOG_DIR),
                        help=f"directory for the DEBUG-level log file (default: {LOG_DIR})")
    return parser.parse_args()


def apply_args(args):
    """Overwrite the configuration globals with the parsed command-line values."""
    global DATASET_PATH, RESULTS_DIR, PROBE_TRAIN_ROWS, BATCH_SIZE, RANDOM_SEED, LOG_DIR
    DATASET_PATH     = args.dataset_path
    RESULTS_DIR      = Path(args.results_dir)
    PROBE_TRAIN_ROWS = args.probe_train_rows
    BATCH_SIZE       = args.batch_size
    RANDOM_SEED      = args.seed
    LOG_DIR          = Path(args.log_dir)

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"linear_probe_{timestamp}.log"

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


def push_config_into_imports():
    """Propagate this script's config into the imported modules before use."""
    _e.DATASET_PATH = DATASET_PATH
    _e.RESULTS_DIR  = RESULTS_DIR

    _t.DATASET_PATH       = DATASET_PATH
    _t.RANDOM_SEED        = RANDOM_SEED
    _t.MAX_ROWS_PER_SPLIT = PROBE_TRAIN_ROWS  # load_split caps via seeded random sample


def embed_batches(queries, tokenizer, model, device, stage):
    """Yield frozen [CLS] vectors batch by batch as (batch_size, hidden) arrays.

    Streaming keeps memory flat: the caller decides whether to accumulate the
    vectors (small train subsample) or consume them immediately (huge test split).
    """
    start    = time.perf_counter()
    last_log = start
    n_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for b in range(n_batches):
            batch_texts = queries[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]  # [CLS] vector per query
            yield cls.cpu().numpy()

            now = time.perf_counter()
            if now - last_log >= LOG_INTERVAL_SECONDS or b + 1 == n_batches:
                elapsed = now - start
                logging.info(
                    f"  [{stage}]  batch {b + 1}/{n_batches}"
                    f"  {(b + 1) / elapsed:.1f} batches/s  elapsed={elapsed:.0f}s"
                )
                last_log = now


def probe_encoder(model_id, model_path, train_df, test_df, device):
    logging.info(f"\n{'─' * 60}")
    logging.info(f"Probe: {model_id}  ({model_path})")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    logging.info(f"  Encoder loaded  (params={sum(p.numel() for p in model.parameters()):,})")

    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"sqli-probe-{model_id}",
            output_dir=str(LOG_DIR),
            log_level="error",
        )
        tracker.start()

    infer_start = time.perf_counter()

    # 1. Fit the probe on frozen embeddings of the training subsample
    #    (small enough to accumulate in memory: 50K × hidden_size ≈ 150 MB)
    logging.info(f"  Embedding {len(train_df):,} training rows ...")
    train_X = np.concatenate(
        list(embed_batches(train_df["query"].tolist(), tokenizer, model, device, f"{model_id} train")),
        axis=0,
    )
    train_y = train_df["label"].to_numpy()

    logging.info("  Fitting logistic regression probe ...")
    probe = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    probe.fit(train_X, train_y)

    # 2. Grade on the test split — predict per batch and keep only the
    #    probabilities, so the ~1.9M-row split never holds 6 GB of vectors.
    logging.info(f"  Embedding + scoring {len(test_df):,} test rows ...")
    prob_chunks = [
        probe.predict_proba(batch_X)[:, 1]
        for batch_X in embed_batches(test_df["query"].tolist(), tokenizer, model, device, f"{model_id} test")
    ]
    probs = np.concatenate(prob_chunks)
    preds = (probs >= 0.5).astype(int)

    inference_time = time.perf_counter() - infer_start

    energy_kwh = None
    if tracker is not None:
        emissions_kg = tracker.stop()
        energy_kwh   = tracker.final_emissions_data.energy_consumed
        logging.info(f"  Energy: {energy_kwh:.4f} kWh  CO2: {emissions_kg * 1000:.1f} gCO2")

    logging.info(f"  Probe time (embed + fit + predict): {inference_time:.0f}s")

    # 3. Per-row predictions CSV — same schema as evaluate.py outputs
    preds_df = test_df.copy()
    preds_df["pred"]     = preds
    preds_df["prob"]     = probs
    preds_df["model_id"] = model_id

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model_id}_predictions.csv"
    preds_df.to_csv(out_path, index=False)
    logging.info(f"  Per-row predictions → {out_path}")

    # 4. Summary row appended to the shared evaluation_results.csv
    overall = _e.compute_metrics(
        preds_df["label"].tolist(),
        preds_df["pred"].tolist(),
        preds_df["prob"].tolist(),
        float("nan"),  # no cross-entropy loss for a frozen-encoder probe
    )
    logging.info(
        f"  PP6 [overall]  mcc={overall['mcc']:.4f}  auc={overall['auc']:.4f}"
        f"  f1={overall['f1']:.4f}  acc={overall['accuracy']:.4f}"
        f"  prec={overall['precision']:.4f}  rec={overall['recall']:.4f}"
    )
    _e.append_to_results_csv([
        _e._build_summary_row(model_id, "overall", len(preds_df), overall, inference_time, energy_kwh)
    ])

    # Free GPU memory before the next encoder
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    setup_logging()
    logging.info("Stage 3 — Linear probes (PP6 untuned-encoder controls, D3.7)")
    logging.info(f"Dataset:          {DATASET_PATH}")
    logging.info(f"Probe train rows: {PROBE_TRAIN_ROWS:,}")
    logging.info(f"Encoders:         {[m[0] for m in ENCODERS]}")

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run from the repository root."
        )

    push_config_into_imports()
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(device)
        logging.info(f"Device:           cuda ({props.name}  {props.total_memory / 1024**3:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Device:           mps (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.warning("No GPU found — embedding on CPU will be slow.")

    # Training subsample: seeded random cap via train.load_split (D3.7)
    train_df = _t.load_split("train")
    # Test split: same loader evaluate.py uses, so all models sit the same exam
    test_df  = _e.load_test_split()

    for model_id, model_path in ENCODERS:
        probe_encoder(model_id, model_path, train_df, test_df, device)

    logging.info("\nAll probes complete. Results in " + str(RESULTS_DIR))


if __name__ == "__main__":
    apply_args(parse_args())
    main()
