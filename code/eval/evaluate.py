# Stage 3 — Evaluation
#
# Evaluates BERT-family models on the compiled SQLi test split and produces:
#   - Per-row prediction CSVs in code/eval/results/
#   - A cumulative summary metrics CSV (evaluation_results.csv)
#   - PP1 (overall) breakdown
#
# Three fine-tuned models run by default (MODELS config, D3.7):
#   codebert-sqli       — the product: CodeBERT fine-tuned on the full corpus
#   bert-sqli-500k      — PP6 control: generic BERT fine-tuned at 500K rows/split
#   codebert-sqli-500k  — PP6 control: CodeBERT fine-tuned at the same 500K budget
#
# The untuned encoders (bert-base, codebert-base) are NOT evaluated here:
# without task training their classification head is randomly initialised and
# scores MCC≈0, which measures nothing (finding F3.1a). They are scored by
# code/eval/linear_probe.py instead (frozen encoder + logistic regression).
#
# Reuses SQLiDataset and compute_metrics from train.py (same sliding-window
# strategy and majority-vote aggregation used during training).
#
# For a quick end-to-end validation, point DATASET_PATH at the committed
# fixture (code/datasets/fixture/fixture_dataset.csv) and RESULTS_DIR at a
# scratch folder — same code path, miniature data. See docs/RUNBOOK.md.

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Add the fine-tuning directory to path so we can import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "fine-tuning"))
import train as _t  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = [
    ("codebert-sqli",      "code/fine-tuning/models/codebert-sqli"),       # product (full corpus)
    ("bert-sqli-500k",     "code/fine-tuning/models/bert-sqli-500k"),      # PP6 control (capped)
    ("codebert-sqli-500k", "code/fine-tuning/models/codebert-sqli-500k"),  # PP6 control (capped)
]

DATASET_PATH     = "code/datasets/compiled/compiled_dataset.csv"
RESULTS_DIR      = Path("code/eval/results")
BATCH_SIZE       = 32
MAX_TOKEN_LENGTH = 512
WINDOW_STRIDE    = 256
CHUNK_SIZE       = 100_000
RANDOM_SEED      = 42

LOG_INTERVAL_SECONDS = 150   # progress line every ~2.5 minutes
LOG_DIR              = Path("code/eval/logs")

# ─────────────────────────────────────────────────────────────────────────────

# Push our config values into the imported module so SQLiDataset uses them
_t.MAX_TOKEN_LENGTH     = MAX_TOKEN_LENGTH
_t.WINDOW_STRIDE        = WINDOW_STRIDE
_t.LOG_INTERVAL_SECONDS = LOG_INTERVAL_SECONDS

SQLiDataset     = _t.SQLiDataset
compute_metrics = _t.compute_metrics

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"evaluate_{timestamp}.log"

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


def load_test_split():
    """Stream the dataset CSV and return the test split as a DataFrame."""
    logging.info("Loading test split ...")

    reader = pd.read_csv(
        DATASET_PATH,
        usecols=["query", "label", "split", "source"],
        chunksize=CHUNK_SIZE,
        dtype={"query": str, "split": str, "source": str},
    )

    chunks = []
    for chunk in reader:
        chunk = chunk[chunk["split"] == "test"].dropna(subset=["query", "label"])
        chunk["label"] = chunk["label"].astype(int)
        chunks.append(chunk[["query", "label", "source"]])
    df = pd.concat(chunks, ignore_index=True)

    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    n_benign    = int((df["label"] == 0).sum())
    n_malicious = int((df["label"] == 1).sum())
    logging.info(f"  {len(df):,} rows — benign={n_benign:,}  malicious={n_malicious:,}")
    logging.debug(f"  source distribution:\n{df['source'].value_counts().to_string()}")
    return df


def run_inference(model, loader, device, model_id):
    """Forward pass with sliding-window majority-vote aggregation.

    Returns:
        results: dict mapping query_idx → {'label', 'pred', 'prob'}
        avg_loss: mean cross-entropy over all windows (informational only)
    """
    model.eval()
    loss_fn    = nn.CrossEntropyLoss()
    total_loss = 0.0
    window_labels, window_preds, window_probs, query_indices = [], [], [], []
    start    = time.perf_counter()
    last_log = start

    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = outputs.logits.argmax(dim=1)

            window_probs.extend(probs.cpu().tolist())
            window_preds.extend(preds.cpu().tolist())
            window_labels.extend(labels.cpu().tolist())
            query_indices.extend(batch["query_idx"].tolist())

            now = time.perf_counter()
            if now - last_log >= LOG_INTERVAL_SECONDS or step == len(loader):
                elapsed = now - start
                logging.info(
                    f"  [{model_id}]  step {step}/{len(loader)}"
                    f"  avg_loss={total_loss/step:.4f}"
                    f"  {step/elapsed:.1f} steps/s  elapsed={elapsed:.0f}s"
                )
                last_log = now

    # Group windows by query; majority vote for pred, average for prob
    label_by_q = {}
    preds_by_q = {}
    probs_by_q = {}
    for lbl, pred, prob, q_idx in zip(window_labels, window_preds, window_probs, query_indices):
        if q_idx not in label_by_q:
            label_by_q[q_idx] = lbl
            preds_by_q[q_idx] = []
            probs_by_q[q_idx] = []
        preds_by_q[q_idx].append(pred)
        probs_by_q[q_idx].append(prob)

    results = {}
    for q_idx in sorted(label_by_q):
        votes = preds_by_q[q_idx]
        results[q_idx] = {
            "label": label_by_q[q_idx],
            "pred":  1 if sum(votes) > len(votes) / 2 else 0,
            "prob":  sum(probs_by_q[q_idx]) / len(probs_by_q[q_idx]),
        }

    return results, total_loss / len(loader)


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


def evaluate_model(model_id, model_path, test_df, device):
    logging.info(f"\n{'─' * 60}")
    logging.info(f"Model: {model_id}  ({model_path})")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    logging.info(f"  Model loaded  (params={sum(p.numel() for p in model.parameters()):,})")

    dataset = SQLiDataset(test_df["query"].tolist(), test_df["label"].tolist(), tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logging.info(f"  {len(dataset):,} windows  {len(loader):,} batches")

    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"sqli-eval-{model_id}",
            output_dir=str(LOG_DIR),
            log_level="error",
        )
        tracker.start()

    infer_start    = time.perf_counter()
    results, avg_loss = run_inference(model, loader, device, model_id)
    inference_time = time.perf_counter() - infer_start

    energy_kwh = None
    if tracker is not None:
        emissions_kg = tracker.stop()
        energy_kwh   = tracker.final_emissions_data.energy_consumed
        logging.info(f"  Energy: {energy_kwh:.4f} kWh  CO2: {emissions_kg * 1000:.1f} gCO2")

    logging.info(f"  Inference time: {inference_time:.0f}s ({inference_time / 3600:.2f}h)")

    # Merge predictions into test_df (row_id == query_idx by construction)
    preds_df = test_df.copy()
    preds_df["pred"]     = [results[i]["pred"] for i in range(len(test_df))]
    preds_df["prob"]     = [results[i]["prob"]  for i in range(len(test_df))]
    preds_df["model_id"] = model_id

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model_id}_predictions.csv"
    preds_df.to_csv(out_path, index=False)
    logging.info(f"  Per-row predictions → {out_path}")

    summary_rows = []

    # PP1 — Overall metrics
    overall = compute_metrics(
        preds_df["label"].tolist(),
        preds_df["pred"].tolist(),
        preds_df["prob"].tolist(),
        avg_loss,
    )
    logging.info(
        f"  PP1 [overall]  mcc={overall['mcc']:.4f}  auc={overall['auc']:.4f}"
        f"  f1={overall['f1']:.4f}  acc={overall['accuracy']:.4f}"
        f"  prec={overall['precision']:.4f}  rec={overall['recall']:.4f}"
    )
    summary_rows.append(
        _build_summary_row(model_id, "overall", len(preds_df), overall, inference_time, energy_kwh)
    )

    append_to_results_csv(summary_rows)

    # Free GPU memory before loading the next model
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    setup_logging()
    logging.info("Stage 3 — Evaluation")
    logging.info(f"Dataset:    {DATASET_PATH}")
    logging.info(f"Models:     {[m[0] for m in MODELS]}")

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run from the repository root."
        )

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(device)
        logging.info(f"Device:     cuda ({props.name}  {props.total_memory / 1024**3:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Device:     mps (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.warning("No GPU found — inference on CPU will be slow.")

    test_df = load_test_split()

    for model_id, model_path in MODELS:
        evaluate_model(model_id, model_path, test_df, device)

    logging.info("\nAll models evaluated. Results in code/eval/results/")


if __name__ == "__main__":
    main()
