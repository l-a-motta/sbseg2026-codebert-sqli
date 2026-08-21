# Stage 2 — Fine-tuning
#
# Fine-tunes a BERT-family encoder on the compiled SQLi dataset for binary
# classification (label 0 = benign, label 1 = malicious).
#
# Three named runs make up the thesis experiment design (see PRESETS / D3.7):
#   codebert-sqli       — the product: CodeBERT on the full corpus
#   bert-sqli-500k      — PP6 control: generic BERT, capped at 500K rows/split
#   codebert-sqli-500k  — PP6 control: CodeBERT at the same capped budget
# Select one with the --preset command-line flag (or apply_preset("<name>")
# when driving this module from a notebook).
#
# Long queries (exceeding 510 content tokens) are split into overlapping
# 512-token windows (sliding-window strategy, D2.6). Each window is trained
# as an independent example. During evaluation, per-window predictions are
# aggregated per query via majority vote.
#
# The configuration constants below are defaults; every one of them can be
# overridden from the command line without editing this file (see --help):
#   python code/fine-tuning/train.py --preset codebert-sqli
#   python code/fine-tuning/train.py --model-name bert-base-uncased --batch-size 32
# For a quick end-to-end validation before a full run, pass the committed
# fixture (--dataset-path code/datasets/fixture/fixture_dataset.csv) and a
# scratch --output-dir — same code path, miniature data. See docs/RUNBOOK.md.

import argparse
import gc
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME        = "microsoft/codebert-base"
DATASET_PATH      = "code/datasets/compiled/compiled_dataset.csv"
OUTPUT_DIR        = "code/fine-tuning/models/codebert-sqli"

MAX_TOKEN_LENGTH  = 512
WINDOW_STRIDE     = 256   # stride between overlapping windows; content per window = MAX_TOKEN_LENGTH - 2

BATCH_SIZE        = 16
EPOCHS            = 3
LEARNING_RATE     = 2e-5
UNDERSAMPLE_RATIO = 2.0   # benign:malicious ratio after undersampling the training split
RANDOM_SEED       = 42

MAX_ROWS_PER_SPLIT = None  # Cap rows loaded per split (None = all rows). Random sample preserving class distribution.

# Mid-epoch checkpoint interval in optimizer steps (None = checkpoint only at
# epoch boundaries). On the full corpus one epoch is ~200K steps — tens of
# hours on a T4, far longer than a Colab session, so without step checkpoints
# a disconnect loses the whole epoch (T2.1). 5,000 steps ≈ 45–60 min of T4
# compute; each save writes the model + optimizer state (~1.5 GB) to
# OUTPUT_DIR/checkpoint, so much smaller intervals waste time on Drive I/O.
CHECKPOINT_EVERY_STEPS = 5_000

LOG_INTERVAL_SECONDS = 150  # Print a progress line every ~2.5 minutes during training

LOG_DIR           = Path("code/fine-tuning/logs")
CHUNK_SIZE        = 100_000

# ── Named training runs (thesis experiment design, D3.7) ─────────────────────
# apply_preset() sets the three values that differ between runs; everything
# else (hyperparameters, seed, undersampling) stays identical so each PP6
# comparison changes exactly one variable at a time.

PRESETS = {
    # The product: CodeBERT fine-tuned on the full corpus (headline model).
    "codebert-sqli": {
        "MODEL_NAME":         "microsoft/codebert-base",
        "OUTPUT_DIR":         "code/fine-tuning/models/codebert-sqli",
        "MAX_ROWS_PER_SPLIT": None,
    },
    # PP6 control: generic BERT at a capped budget.
    "bert-sqli-500k": {
        "MODEL_NAME":         "bert-base-uncased",
        "OUTPUT_DIR":         "code/fine-tuning/models/bert-sqli-500k",
        "MAX_ROWS_PER_SPLIT": 500_000,
    },
    # PP6 control: CodeBERT at the same capped budget — the fair partner of
    # bert-sqli-500k (same data volume, so the only remaining difference
    # between the two is the code-domain pre-training).
    "codebert-sqli-500k": {
        "MODEL_NAME":         "microsoft/codebert-base",
        "OUTPUT_DIR":         "code/fine-tuning/models/codebert-sqli-500k",
        "MAX_ROWS_PER_SPLIT": 500_000,
    },
}


def apply_preset(name):
    """Overwrite MODEL_NAME, OUTPUT_DIR and MAX_ROWS_PER_SPLIT from PRESETS[name]."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    for key, value in PRESETS[name].items():
        globals()[key] = value


# ── Command-line overrides ────────────────────────────────────────────────────
# The constants above are the defaults used in the article; each flag below
# overrides one of them for a single run, so no source edit is ever required.
# --preset is applied first; explicit flags win over the preset.

_UNSET = object()  # distinguishes "flag not given" from any real value


def parse_rows_cap(value):
    """argparse type: a positive integer, or 'all'/'none' meaning no cap."""
    if str(value).lower() in ("all", "none"):
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 — fine-tune a BERT-family encoder on the compiled SQLi dataset.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="named training run from the thesis experiment design (D3.7); "
                             "sets --model-name, --output-dir and --max-rows-per-split")
    parser.add_argument("--model-name", default=_UNSET,
                        help=f"HuggingFace encoder to fine-tune (default: {MODEL_NAME})")
    parser.add_argument("--output-dir", default=_UNSET,
                        help=f"directory for the best model and checkpoints (default: {OUTPUT_DIR})")
    parser.add_argument("--max-rows-per-split", default=_UNSET, type=parse_rows_cap,
                        help="cap on rows loaded per split, or 'all' for no cap (default: all)")
    parser.add_argument("--dataset-path", default=DATASET_PATH,
                        help=f"compiled dataset CSV (default: {DATASET_PATH})")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int,
                        help=f"training batch size (default: {BATCH_SIZE})")
    parser.add_argument("--epochs", default=EPOCHS, type=int,
                        help=f"number of training epochs (default: {EPOCHS})")
    parser.add_argument("--learning-rate", default=LEARNING_RATE, type=float,
                        help=f"AdamW learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--undersample-ratio", default=UNDERSAMPLE_RATIO, type=float,
                        help=f"benign:malicious ratio after undersampling (default: {UNDERSAMPLE_RATIO})")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int,
                        help=f"random seed (default: {RANDOM_SEED})")
    parser.add_argument("--log-dir", default=str(LOG_DIR),
                        help=f"directory for the DEBUG-level log file (default: {LOG_DIR})")
    return parser.parse_args()


def apply_args(args):
    """Overwrite the configuration globals with the parsed command-line values."""
    global MODEL_NAME, OUTPUT_DIR, MAX_ROWS_PER_SPLIT, DATASET_PATH
    global BATCH_SIZE, EPOCHS, LEARNING_RATE, UNDERSAMPLE_RATIO, RANDOM_SEED, LOG_DIR

    if args.preset is not None:
        apply_preset(args.preset)

    # Preset-controlled values: only override when the flag was given explicitly.
    if args.model_name is not _UNSET:
        MODEL_NAME = args.model_name
    if args.output_dir is not _UNSET:
        OUTPUT_DIR = args.output_dir
    if args.max_rows_per_split is not _UNSET:
        MAX_ROWS_PER_SPLIT = args.max_rows_per_split

    DATASET_PATH      = args.dataset_path
    BATCH_SIZE        = args.batch_size
    EPOCHS            = args.epochs
    LEARNING_RATE     = args.learning_rate
    UNDERSAMPLE_RATIO = args.undersample_ratio
    RANDOM_SEED       = args.seed
    LOG_DIR           = Path(args.log_dir)

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_{timestamp}.log"

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


def get_ram_gb():
    return psutil.Process().memory_info().rss / 1024 ** 3


def get_gpu_mem_gb(device):
    """Return (allocated_gb, peak_gb) for the active device, or (None, None)."""
    if device.type == "cuda":
        return (
            torch.cuda.memory_allocated(device) / 1024 ** 3,
            torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        )
    if device.type == "mps":
        try:
            alloc = torch.mps.current_allocated_memory() / 1024 ** 3
            return alloc, alloc
        except Exception:
            return None, None
    return None, None


def get_gpu_utilization():
    """Return GPU utilization % for CUDA devices, or None if unavailable."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.utilization()
    except Exception:
        pass
    return None


def log_system_stats(device, stage):
    parts = [f"[{stage}]", f"ram={get_ram_gb():.2f}GB"]
    alloc, peak = get_gpu_mem_gb(device)
    if alloc is not None:
        parts.append(f"gpu_mem={alloc:.2f}GB  gpu_peak={peak:.2f}GB")
    util = get_gpu_utilization()
    if util is not None:
        parts.append(f"gpu_util={util}%")
    logging.debug("  ".join(parts))


def load_split(split_name):
    """Read one split from the compiled CSV, streaming in chunks to avoid OOM."""
    logging.info(f"Loading split='{split_name}' ...")

    reader = pd.read_csv(
        DATASET_PATH,
        usecols=["query", "label", "split"],
        chunksize=CHUNK_SIZE,
        dtype={"query": str, "split": str},
    )

    chunks = []
    for chunk in reader:
        chunk = chunk[chunk["split"] == split_name].dropna(subset=["query", "label"])
        chunk["label"] = chunk["label"].astype(int)
        chunks.append(chunk[["query", "label"]])
    df = pd.concat(chunks, ignore_index=True)
    if MAX_ROWS_PER_SPLIT is not None and len(df) > MAX_ROWS_PER_SPLIT:
        df = df.sample(n=MAX_ROWS_PER_SPLIT, random_state=RANDOM_SEED).reset_index(drop=True)
        logging.info(f"  Capped to {MAX_ROWS_PER_SPLIT:,} rows (MAX_ROWS_PER_SPLIT)")

    n_benign    = int((df["label"] == 0).sum())
    n_malicious = int((df["label"] == 1).sum())
    logging.info(f"  {len(df):,} rows — benign={n_benign:,}  malicious={n_malicious:,}")
    logging.debug(f"  query length stats:\n{df['query'].str.len().describe().to_string()}")
    return df


def undersample_benign(df):
    """Drop benign rows from the training split until benign:malicious == UNDERSAMPLE_RATIO."""
    n_malicious     = int((df["label"] == 1).sum())
    n_benign_target = int(n_malicious * UNDERSAMPLE_RATIO)
    benign          = df[df["label"] == 0]
    malicious       = df[df["label"] == 1]

    if len(benign) <= n_benign_target:
        logging.info("  Undersampling: benign already at or below target ratio, skipping.")
        return df

    benign_sampled = benign.sample(n_benign_target, random_state=RANDOM_SEED)
    result = (
        pd.concat([benign_sampled, malicious], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    logging.info(
        f"  After undersample: benign={n_benign_target:,}  malicious={n_malicious:,}"
        f"  ratio={UNDERSAMPLE_RATIO:.1f}:1"
    )
    return result


def compute_class_weights(df):
    """Inverse-frequency weights for CrossEntropyLoss from the training distribution."""
    n_total     = len(df)
    n_benign    = int((df["label"] == 0).sum())
    n_malicious = int((df["label"] == 1).sum())
    w_benign    = n_total / (2 * n_benign)
    w_malicious = n_total / (2 * n_malicious)
    logging.info(f"  Class weights: benign={w_benign:.4f}  malicious={w_malicious:.4f}")
    return torch.tensor([w_benign, w_malicious], dtype=torch.float)


class SQLiDataset(Dataset):
    """Stores raw token-ID arrays per query; defers windowing and padding to __getitem__.

    Memory strategy (D2.7): each query is encoded once to a compact numpy int32
    array at construction. A flat index of (query_idx, window_start_token) pairs
    maps each logical window to its source query and offset. __getitem__ slices
    the stored token IDs and pads on the fly — no 512-element padded lists are
    held in memory. Typical footprint: ~200 bytes/query vs ~28 KB/window in the
    fully pre-tokenised approach (see D2.7).
    """

    def __init__(self, queries, labels, tokenizer):
        content_size = MAX_TOKEN_LENGTH - 2

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # Encode every query once to raw token IDs; store as compact int32 arrays.
        log_every = max(1, len(queries) // 5)
        self.token_ids = []
        for i, text in enumerate(queries, 1):
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.token_ids.append(np.array(ids, dtype=np.int32))
            if i % log_every == 0 or i == len(queries):
                logging.debug(f"  Encoding: {i:,}/{len(queries):,} queries")

        self.labels = np.array(labels, dtype=np.int8)

        # Build flat index: each row is (query_idx, window_start_token_offset).
        index_rows = []
        for q_idx, ids in enumerate(self.token_ids):
            if len(ids) <= content_size:
                index_rows.append((q_idx, 0))
            else:
                start = 0
                while True:
                    index_rows.append((q_idx, start))
                    if start + content_size >= len(ids):
                        break
                    start += WINDOW_STRIDE

        self.flat_index = np.array(index_rows, dtype=np.int32)

        n_queries = len(queries)
        n_windows = len(self.flat_index)
        total_tokens = sum(len(ids) for ids in self.token_ids)
        approx_mb = (total_tokens * 4 + n_windows * 8) / 1024 / 1024
        logging.debug(
            f"  Dataset: {n_queries:,} queries → {n_windows:,} windows"
            f" ({n_windows - n_queries:,} extra from long queries)"
        )
        logging.debug(f"  Token-ID store: ~{approx_mb:.1f} MB")

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        q_idx = int(self.flat_index[idx, 0])
        start = int(self.flat_index[idx, 1])
        content_size = MAX_TOKEN_LENGTH - 2

        chunk = self.token_ids[q_idx][start : start + content_size].tolist()
        ids   = [self.cls_token_id] + chunk + [self.sep_token_id]
        pad   = MAX_TOKEN_LENGTH - len(ids)
        mask  = [1] * len(ids) + [0] * pad
        ids   = ids + [self.pad_token_id] * pad

        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label":          torch.tensor(int(self.labels[q_idx]), dtype=torch.long),
            "query_idx":      torch.tensor(q_idx, dtype=torch.long),
        }


def compute_metrics(labels, preds, probs, avg_loss):
    return {
        "loss":      avg_loss,
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       roc_auc_score(labels, probs),
        "mcc":       matthews_corrcoef(labels, preds),
    }


def train_epoch(model, loader, optimizer, loss_fn, device, epoch,
                save_step_checkpoint=None, skip_steps=0):
    """Train one epoch. Each window is an independent training example.
    Reported metrics are window-level (not aggregated per query).

    skip_steps > 0 resumes an interrupted epoch (T2.1): the first skip_steps
    batches are skipped — the loader order is reproducible because it is
    seeded per epoch — and metrics cover only the remaining batches. Every
    CHECKPOINT_EVERY_STEPS processed steps, save_step_checkpoint persists a
    resumable state.
    """
    model.train()
    total_loss = 0.0
    processed  = 0
    all_labels, all_preds, all_probs = [], [], []
    epoch_start  = time.perf_counter()
    last_log_t   = epoch_start

    if skip_steps:
        # Skipped batches still pay collation cost (__getitem__), but no
        # forward/backward — fast-forwarding is a small fraction of a step.
        logging.info(f"  Skipping {skip_steps:,} already-trained batches (mid-epoch resume) ...")

    for step, batch in enumerate(loader, 1):
        if step <= skip_steps:
            continue

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        processed  += 1
        total_loss += loss.item()
        all_probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].detach().cpu().tolist())
        all_preds.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        now = time.perf_counter()
        if now - last_log_t >= LOG_INTERVAL_SECONDS or step == len(loader):
            elapsed       = now - epoch_start
            steps_per_sec = processed / elapsed
            running_f1    = f1_score(all_labels, all_preds, zero_division=0)
            logging.info(
                f"  epoch {epoch}  step {step}/{len(loader)}"
                f"  avg_loss={total_loss/processed:.4f}  f1={running_f1:.4f}"
                f"  {steps_per_sec:.1f} steps/s  elapsed={elapsed:.0f}s"
            )
            log_system_stats(device, f"epoch {epoch} step {step}")
            last_log_t = now

        if (
            save_step_checkpoint is not None
            and CHECKPOINT_EVERY_STEPS is not None
            and processed % CHECKPOINT_EVERY_STEPS == 0
            and step < len(loader)   # epoch-boundary checkpoint follows anyway
        ):
            save_step_checkpoint(epoch, step)

    return compute_metrics(all_labels, all_preds, all_probs, total_loss / max(processed, 1))


def evaluate(model, loader, loss_fn, device, epoch):
    """Evaluate and aggregate per-window predictions to query level via majority vote."""
    model.eval()
    total_loss    = 0.0
    window_labels = []
    window_preds  = []
    window_probs  = []
    query_indices = []
    eval_start    = time.perf_counter()
    last_log_t    = eval_start

    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            window_probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
            window_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            window_labels.extend(labels.cpu().tolist())
            query_indices.extend(batch["query_idx"].tolist())

            now = time.perf_counter()
            if now - last_log_t >= LOG_INTERVAL_SECONDS or step == len(loader):
                elapsed       = now - eval_start
                steps_per_sec = step / elapsed
                logging.info(
                    f"  epoch {epoch}  eval step {step}/{len(loader)}"
                    f"  avg_loss={total_loss/step:.4f}"
                    f"  {steps_per_sec:.1f} steps/s  elapsed={elapsed:.0f}s"
                )
                log_system_stats(device, f"epoch {epoch} eval step {step}")
                last_log_t = now

    # Group windows back to their source queries
    label_by_query = {}
    preds_by_query = {}
    probs_by_query = {}

    for label, pred, prob, q_idx in zip(window_labels, window_preds, window_probs, query_indices):
        if q_idx not in label_by_query:
            label_by_query[q_idx] = label
            preds_by_query[q_idx] = []
            probs_by_query[q_idx] = []
        preds_by_query[q_idx].append(pred)
        probs_by_query[q_idx].append(prob)

    # Majority vote: predict malicious if more than half of windows predict malicious.
    # Average window probabilities for AUC.
    final_labels, final_preds, final_probs = [], [], []
    for q_idx in sorted(label_by_query):
        votes = preds_by_query[q_idx]
        final_labels.append(label_by_query[q_idx])
        final_preds.append(1 if sum(votes) > len(votes) / 2 else 0)
        final_probs.append(sum(probs_by_query[q_idx]) / len(probs_by_query[q_idx]))

    return compute_metrics(final_labels, final_preds, final_probs, total_loss / len(loader))


def log_metrics(split_name, metrics, level="query"):
    logging.info(
        f"  {split_name:5s} [{level}-level]"
        f"  loss={metrics['loss']:.4f}"
        f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}"
        f"  mcc={metrics['mcc']:.4f}  auc={metrics['auc']:.4f}"
        f"  prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}"
    )


def main():
    setup_logging()
    logging.info(f"Model:       {MODEL_NAME}")
    logging.info(f"Dataset:     {DATASET_PATH}")
    epochs = EPOCHS
    logging.debug(
        f"Config — batch={BATCH_SIZE}  epochs={EPOCHS}  lr={LEARNING_RATE}"
        f"  max_tokens={MAX_TOKEN_LENGTH}  window_stride={WINDOW_STRIDE}"
        f"  undersample_ratio={UNDERSAMPLE_RATIO}  seed={RANDOM_SEED}"
    )

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device:      {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logging.info(f"GPU:         {props.name}  {props.total_memory / 1024**3:.1f} GB")
    elif device.type == "cpu":
        logging.warning("No GPU found — training on CPU will be slow.")

    # Load data
    train_df = load_split("train")
    val_df   = load_split("val")

    # Balance training split
    train_df = undersample_benign(train_df)
    weights  = compute_class_weights(train_df)

    # Tokenizer and datasets
    logging.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logging.info("Building datasets (encoding queries, building window index) ...")
    train_ds = SQLiDataset(train_df["query"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds   = SQLiDataset(val_df["query"].tolist(),   val_df["label"].tolist(),   tokenizer)
    log_system_stats(device, "after dataset build")

    # The train loader is created per epoch (inside the epoch loop) with a
    # per-epoch seeded generator, so the shuffle order is reproducible across
    # restarts — required for mid-epoch resume (T2.1).
    val_loader      = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_train_batches = (len(train_ds) + BATCH_SIZE - 1) // BATCH_SIZE
    logging.info(f"Train: {len(train_ds):,} windows  {n_train_batches:,} batches")
    logging.info(f"Val:   {len(val_ds):,} windows  {len(val_loader):,} batches")

    # Checkpoint paths — derived from OUTPUT_DIR so Drive overrides propagate automatically.
    out_path              = Path(OUTPUT_DIR)
    checkpoint_dir        = out_path / "checkpoint"
    checkpoint_state_path = checkpoint_dir / "training_state.pt"

    resuming = checkpoint_state_path.exists()

    # Model — load from last checkpoint if resuming, otherwise from HuggingFace.
    logging.info("Loading model ...")
    if resuming:
        logging.info(f"  Checkpoint detected: {checkpoint_state_path}")
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir), num_labels=2)
        # safetensors memory-maps checkpoint/model.safetensors on load; Windows
        # cannot overwrite a mapped file, so the next checkpoint save would crash
        # (os error 1224). Cloning every tensor moves the weights to fresh memory
        # and releases the mapping. No-op cost on Linux/Colab.
        for param in model.parameters():
            param.data = param.data.clone()
        for buffer in model.buffers():
            buffer.data = buffer.data.clone()
        gc.collect()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    log_system_stats(device, "after model load")

    loss_fn   = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_mcc    = -1.0
    start_epoch = 1
    resume_step = 0

    if resuming:
        state       = torch.load(checkpoint_state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_mcc    = state["best_mcc"]
        resume_step = state.get("step", 0)

        if resume_step > 0:
            # Interrupted mid-epoch (T2.1): re-enter the same epoch and skip
            # the batches already trained.
            start_epoch = state["epoch"]
            logging.info(
                f"  Resuming mid-epoch: epoch {start_epoch}, after step {resume_step:,}"
                f"  best_mcc_so_far={best_mcc:.4f}"
            )
        else:
            start_epoch = state["epoch"] + 1
            logging.info(f"  Resuming from epoch {start_epoch}  best_mcc_so_far={best_mcc:.4f}")

        if resume_step == 0 and state.get("eval_pending", False):
            pending_epoch = state["epoch"]
            logging.info(f"  Epoch {pending_epoch} training completed but eval was interrupted — running eval now ...")
            t0 = time.perf_counter()
            val_metrics  = evaluate(model, val_loader, loss_fn, device, pending_epoch)
            eval_elapsed = time.perf_counter() - t0
            log_metrics("val", val_metrics, level="query")
            logging.info(f"  Timing: eval={eval_elapsed:.0f}s")
            log_system_stats(device, f"epoch {pending_epoch} eval complete")

            if val_metrics["mcc"] > best_mcc:
                best_mcc = val_metrics["mcc"]
                out_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out_path)
                tokenizer.save_pretrained(out_path)
                logging.info(f"  Saved best model (val_mcc={best_mcc:.4f}) → {OUTPUT_DIR}")

            torch.save(
                {
                    "epoch":                pending_epoch,
                    "step":                 0,
                    "best_mcc":             best_mcc,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_pending":         False,
                },
                checkpoint_state_path,
            )
            logging.info(f"  Checkpoint updated (eval_pending cleared, epoch={pending_epoch})")

        if start_epoch > epochs:
            logging.info("All epochs already completed — nothing to do.")
            return

    def save_step_checkpoint(epoch, step):
        """Mid-epoch checkpoint (T2.1): everything needed to resume after `step`."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch":                epoch,
                "step":                 step,
                "best_mcc":             best_mcc,
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_pending":         False,
            },
            checkpoint_state_path,
        )
        logging.info(f"  Step checkpoint saved (epoch={epoch}  step={step:,}) → {checkpoint_dir}")

    # Energy tracker — supports NVIDIA, AMD, Apple Silicon, and CPU.
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name="sqli-bert-finetune",
            output_dir=str(LOG_DIR),
            log_level="error",
        )
        tracker.start()
        logging.info("Energy tracking: codecarbon active")
    else:
        logging.info("Energy tracking: codecarbon not installed — elapsed time only")

    training_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        logging.info(f"── Epoch {epoch}/{epochs} {'─' * 50}")

        # Per-epoch seeded shuffle: order is reproducible across restarts, so
        # a mid-epoch resume skips exactly the batches already trained (T2.1).
        epoch_generator = torch.Generator().manual_seed(RANDOM_SEED + epoch)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            generator=epoch_generator,
        )

        skip_steps  = resume_step if epoch == start_epoch else 0
        resume_step = 0

        t0            = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            save_step_checkpoint, skip_steps,
        )
        train_elapsed = time.perf_counter() - t0
        log_metrics("train", train_metrics, level="window")

        # Mid-epoch checkpoint: training done, eval pending. Saves model weights so
        # if the session is interrupted before/during eval, we can resume from here.
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch":                epoch,
                "step":                 0,
                "best_mcc":             best_mcc,
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_pending":         True,
            },
            checkpoint_state_path,
        )
        logging.info(f"  Mid-epoch checkpoint saved (eval_pending=True, epoch={epoch})")

        t0           = time.perf_counter()
        val_metrics  = evaluate(model, val_loader, loss_fn, device, epoch)
        eval_elapsed = time.perf_counter() - t0
        log_metrics("val", val_metrics, level="query")

        logging.info(f"  Timing: train={train_elapsed:.0f}s  eval={eval_elapsed:.0f}s")
        log_system_stats(device, f"epoch {epoch} end")

        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            out_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
            logging.info(f"  Saved best model (val_mcc={best_mcc:.4f}) → {OUTPUT_DIR}")

        # Final checkpoint: eval done, clear eval_pending. Model weights in checkpoint_dir
        # are unchanged since eval doesn't mutate the model, so only state file is updated.
        torch.save(
            {
                "epoch":                epoch,
                "step":                 0,
                "best_mcc":             best_mcc,
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_pending":         False,
            },
            checkpoint_state_path,
        )
        logging.info(f"  Checkpoint saved → {checkpoint_dir}  (epoch={epoch}  best_mcc={best_mcc:.4f})")

    total_elapsed = time.perf_counter() - training_start
    logging.info(f"Training complete. Best val MCC: {best_mcc:.4f}")
    logging.info(f"Total training time: {total_elapsed:.0f}s ({total_elapsed / 3600:.2f}h)")

    if tracker is not None:
        emissions_kg = tracker.stop()
        energy_kwh   = tracker.final_emissions_data.energy_consumed
        logging.info(f"Energy consumed: {energy_kwh:.4f} kWh  CO2 equivalent: {emissions_kg * 1000:.1f} gCO2")

    log_system_stats(device, "training complete")


if __name__ == "__main__":
    apply_args(parse_args())
    main()
