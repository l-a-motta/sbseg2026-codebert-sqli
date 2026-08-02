# Stage 4 — Comparison
#
# Reads evaluation_results.csv (produced by evaluate.py, linear_probe.py and
# evaluate_baselines.py) and per-row prediction CSVs, then generates comparison
# tables and charts for all research questions:
#
#   PP1 — Overall model comparison (all models, scope='overall')
#   PP4 — Fine-tuned BERT vs Llama 3.2 3B (Ollama) on shared 3K sample
#   PP5 — Fine-tuned BERT vs traditional ML algorithms on full test split
#   PP6 — Matched-budget 2x2 (D3.7): {bert-base, codebert-base} x
#         {linear probe, fine-tuned @ 500K rows/split}
#
# Outputs: CSVs + PNGs in code/comparison/output/

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("code/eval/results")
OUTPUT_DIR  = Path("code/comparison/output")
LOG_DIR     = Path("code/comparison/logs")

# Display order for models in PP1/PP5 charts (left to right)
MODEL_ORDER = [
    "bert-base-probe",
    "codebert-base-probe",
    "bert-sqli-500k",
    "codebert-sqli-500k",
    "codebert-sqli",
    "logistic_regression",
    "random_forest",
    "svm_linear",
    "llama3.2-3b",
]

# Human-readable labels for charts
MODEL_LABELS = {
    "bert-base-probe":      "BERT-base (probe)",
    "codebert-base-probe":  "CodeBERT-base (probe)",
    "bert-sqli-500k":       "BERT (fine-tuned, 500K)",
    "codebert-sqli-500k":   "CodeBERT (fine-tuned, 500K)",
    "codebert-sqli":        "CodeBERT-SQLi (full corpus)",
    "logistic_regression":  "Logistic Regression",
    "random_forest":        "Random Forest",
    "svm_linear":           "SVM (Linear)",
    "llama3.2-3b":          "Llama 3.2 3B",
}

# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"compare_{timestamp}.log"

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


def load_results() -> pd.DataFrame:
    results_path = RESULTS_DIR / "evaluation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {results_path}\n"
            "Run evaluate.py and evaluate_baselines.py first."
        )
    df = pd.read_csv(results_path)
    logging.info(f"Loaded {len(df)} rows from {results_path}")
    logging.debug(f"Columns: {list(df.columns)}")
    logging.debug(f"Models present: {df['model_id'].unique().tolist()}")
    logging.debug(f"Scopes present: {df['scope'].unique().tolist()}")
    return df


def _ordered_labels(model_ids):
    """Return (model_id list, display label list) in MODEL_ORDER, others appended."""
    present = set(model_ids)
    ordered = [m for m in MODEL_ORDER if m in present]
    extras  = [m for m in model_ids if m not in set(MODEL_ORDER)]
    ordered += extras
    labels  = [MODEL_LABELS.get(m, m) for m in ordered]
    return ordered, labels


def _bar_chart(
    model_ids, values, title, ylabel, out_path, color_highlight=None, figsize=(10, 5)
):
    ordered_ids, ordered_labels = _ordered_labels(model_ids)
    ordered_values = [
        values.get(m, float("nan")) for m in ordered_ids
    ]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [
        "#2196F3" if (color_highlight and m == color_highlight) else "#90CAF9"
        for m in ordered_ids
    ]
    bars = ax.bar(ordered_labels, ordered_values, color=colors, edgecolor="white")

    for bar, val in zip(bars, ordered_values):
        if not pd.isna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xticklabels(ordered_labels, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"  Saved: {out_path}")


def pp1_overall(df: pd.DataFrame):
    """PP1: overall metrics for all models. Table + bar charts per metric."""
    logging.info("\n── PP1 — Overall comparison ───────────────────────────────")
    overall = df[df["scope"] == "overall"].copy()
    if overall.empty:
        logging.warning("  No 'overall' scope rows found — skipping PP1.")
        return

    metrics = ["mcc", "auc", "f1", "accuracy", "precision", "recall"]

    # Sort by MCC descending
    ordered_ids, ordered_labels = _ordered_labels(overall["model_id"].tolist())
    table = overall.set_index("model_id").reindex(ordered_ids)
    table.index = ordered_labels
    table_out = table[metrics + ["inference_time_s", "energy_kwh", "n_samples"]]

    out_csv = OUTPUT_DIR / "pp1_overall_comparison.csv"
    table_out.to_csv(out_csv)
    logging.info(f"  Saved: {out_csv}")

    # ASCII-only in print() calls: box-drawing chars crash on cp1252 consoles (Windows)
    print("\n-- PP1: Overall Model Comparison (sorted by MCC) --")
    print(table_out.sort_values("mcc", ascending=False).to_string(float_format="{:.4f}".format))

    for metric in metrics:
        values = dict(zip(overall["model_id"], overall[metric]))
        _bar_chart(
            model_ids=list(values.keys()),
            values=values,
            title=f"PP1 — {metric.upper()} by Model",
            ylabel=metric.upper(),
            out_path=OUTPUT_DIR / f"pp1_{metric}_comparison.png",
            color_highlight="codebert-sqli",
        )


def pp4_bert_vs_llm(df: pd.DataFrame):
    """PP4: fine-tuned BERT vs Llama 3.2 3B on the shared 3K sample."""
    logging.info("\n── PP4 — Fine-tuned BERT vs Llama 3.2 3B ──────────────────")
    pp4_models = ["codebert-sqli", "llama3.2-3b"]
    pp4_scope  = "overall_pp4sample"

    pp4 = df[(df["scope"] == pp4_scope) & (df["model_id"].isin(pp4_models))].copy()
    if pp4.empty:
        logging.warning("  No PP4 scope rows found — skipping PP4.")
        logging.info("  (Run evaluate_ollama.py to generate PP4 results.)")
        return

    metrics = ["mcc", "auc", "f1", "accuracy", "precision", "recall"]
    out_csv = OUTPUT_DIR / "pp4_bert_vs_llm.csv"
    pp4.set_index("model_id")[metrics + ["n_samples"]].to_csv(out_csv)
    logging.info(f"  Saved: {out_csv}")

    for metric in ["mcc", "f1", "auc"]:
        values = dict(zip(pp4["model_id"], pp4[metric]))
        _bar_chart(
            model_ids=list(values.keys()),
            values=values,
            title=f"PP4 — {metric.upper()}: Fine-tuned BERT vs Llama 3.2 3B (3K sample)",
            ylabel=metric.upper(),
            out_path=OUTPUT_DIR / f"pp4_{metric}_comparison.png",
            color_highlight="codebert-sqli",
            figsize=(5, 5),
        )

    print("\n-- PP4: Fine-tuned BERT vs Llama 3.2 3B (3K sample) --")
    print(pp4.set_index("model_id")[metrics + ["n_samples"]]
          .to_string(float_format="{:.4f}".format))


def pp5_bert_vs_baselines(df: pd.DataFrame):
    """PP5: fine-tuned BERT vs traditional ML algorithms on the full test split."""
    logging.info("\n── PP5 — Fine-tuned BERT vs Traditional ML Algorithms ──────")
    pp5_models = ["codebert-sqli", "logistic_regression", "random_forest", "svm_linear"]
    pp5 = df[(df["scope"] == "overall") & (df["model_id"].isin(pp5_models))].copy()
    if pp5.empty:
        logging.warning("  No PP5 models found in 'overall' scope — skipping PP5.")
        return

    metrics = ["mcc", "auc", "f1", "accuracy", "precision", "recall"]
    ordered_ids, ordered_labels = _ordered_labels(pp5["model_id"].tolist())
    table = pp5.set_index("model_id").reindex(ordered_ids)
    table.index = ordered_labels

    out_csv = OUTPUT_DIR / "pp5_bert_vs_baselines.csv"
    table[metrics + ["inference_time_s", "energy_kwh", "n_samples"]].to_csv(out_csv)
    logging.info(f"  Saved: {out_csv}")

    for metric in ["mcc", "f1", "auc"]:
        values = dict(zip(pp5["model_id"], pp5[metric]))
        _bar_chart(
            model_ids=list(values.keys()),
            values=values,
            title=f"PP5 — {metric.upper()}: Fine-tuned BERT vs Traditional ML",
            ylabel=metric.upper(),
            out_path=OUTPUT_DIR / f"pp5_{metric}_comparison.png",
            color_highlight="codebert-sqli",
        )

    print("\n-- PP5: Fine-tuned BERT vs Traditional ML Algorithms (full test split) --")
    print(table[metrics + ["n_samples"]].sort_values("mcc", ascending=False)
          .to_string(float_format="{:.4f}".format))


def pp6_finetuning_impact(df: pd.DataFrame):
    """PP6: matched-budget 2x2 (D3.7) — {bert-base, codebert-base} x {probe, fine-tuned @500K}.

    Reading the table down a column isolates code pre-training; across a row
    isolates the fine-tuning contribution. The full-corpus product model is
    excluded — PP6 deltas are computed within the matched budget only.
    """
    logging.info("\n── PP6 — Fine-tuning impact (matched-budget 2x2, D3.7) ─────")
    pp6_models = ["bert-base-probe", "codebert-base-probe", "bert-sqli-500k", "codebert-sqli-500k"]
    pp6 = df[(df["scope"] == "overall") & (df["model_id"].isin(pp6_models))].copy()
    if pp6.empty:
        logging.warning("  No PP6 models found in 'overall' scope — skipping PP6.")
        return

    metrics = ["mcc", "auc", "f1", "accuracy", "precision", "recall"]
    ordered_ids, ordered_labels = _ordered_labels(pp6["model_id"].tolist())
    table = pp6.set_index("model_id").reindex(ordered_ids)
    table.index = ordered_labels

    out_csv = OUTPUT_DIR / "pp6_finetuning_impact.csv"
    table[metrics + ["inference_time_s", "energy_kwh", "n_samples"]].to_csv(out_csv)
    logging.info(f"  Saved: {out_csv}")

    for metric in ["mcc", "f1", "auc"]:
        values = dict(zip(pp6["model_id"], pp6[metric]))
        _bar_chart(
            model_ids=list(values.keys()),
            values=values,
            title=f"PP6 — {metric.upper()}: Matched-Budget 2x2 (probe vs fine-tuned)",
            ylabel=metric.upper(),
            out_path=OUTPUT_DIR / f"pp6_{metric}_comparison.png",
            color_highlight="codebert-sqli-500k",
            figsize=(7, 5),
        )

    print("\n-- PP6: Matched-Budget 2x2 (D3.7) --")
    print(table[metrics + ["n_samples"]].to_string(float_format="{:.4f}".format))


def summary_scatter(df: pd.DataFrame):
    """Inference time vs MCC scatter plot for all models (bubble = energy)."""
    logging.info("\n── Summary: inference time vs MCC scatter ─────────────────")
    overall = df[df["scope"] == "overall"].copy()
    if overall.empty or overall["inference_time_s"].isna().all():
        logging.warning("  No timing data — skipping summary scatter.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in overall.iterrows():
        m   = row["model_id"]
        lbl = MODEL_LABELS.get(m, m)
        t   = row["inference_time_s"]
        mcc = row["mcc"]
        kwh = row["energy_kwh"]
        size = max(50, (kwh * 1000) if not pd.isna(kwh) else 50)
        ax.scatter(t, mcc, s=size, alpha=0.7, label=lbl)
        ax.annotate(lbl, (t, mcc), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Inference Time (s)")
    ax.set_ylabel("MCC")
    ax.set_title("Inference Time vs MCC — bubble area ∝ energy (kWh)")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "summary_time_vs_mcc.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logging.info(f"  Saved: {out_png}")


def main():
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 4 — Comparison")
    logging.info(f"Results dir: {RESULTS_DIR}")
    logging.info(f"Output dir:  {OUTPUT_DIR}")

    df = load_results()

    pp1_overall(df)
    pp4_bert_vs_llm(df)
    pp5_bert_vs_baselines(df)
    pp6_finetuning_impact(df)
    summary_scatter(df)

    logging.info(f"\nAll charts and tables saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
