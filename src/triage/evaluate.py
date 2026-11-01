"""Compare baseline and LLM classifiers on the held-out eval set.

Run as:  PYTHONPATH=src python -m triage.evaluate
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from triage import baseline, llm

_ROOT = Path(__file__).parent.parent.parent
EVAL_PATH = _ROOT / "data" / "eval" / "eval_set.csv"
REPORTS_PATH = _ROOT / "reports"

CATEGORY_LABELS = ["access", "email", "hardware", "network", "other", "software"]
URGENCY_LABELS = ["high", "low", "medium"]

# Haiku-tier pricing (per 1 M tokens) used for cost estimate
_INPUT_PRICE_PER_M = 1.00
_OUTPUT_PRICE_PER_M = 5.00


# ---------------------------------------------------------------------------
# Pure helper functions (tested without API)
# ---------------------------------------------------------------------------

def compute_metrics(true: list[str], pred: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(true, pred)), 4),
        "macro_f1": round(float(f1_score(true, pred, average="macro", zero_division=0)), 4),
    }


def find_top_disagreements(
    eval_df: pl.DataFrame,
    baseline_results: list[dict],
    llm_results: list[dict],
    n: int = 5,
) -> list[dict]:
    """Return up to *n* tickets where LLM and baseline disagree most."""
    b_by_id = {r["ticket_id"]: r for r in baseline_results}
    l_by_id = {r["ticket_id"]: r for r in llm_results}
    rows = []
    for ticket in eval_df.select(["ticket_id", "subject", "category", "urgency"]).to_dicts():
        tid = ticket["ticket_id"]
        b = b_by_id.get(tid, {})
        lp = l_by_id.get(tid, {})
        score = int(b.get("category") != lp.get("category")) + int(
            b.get("urgency") != lp.get("urgency")
        )
        if score > 0:
            rows.append({
                "ticket_id": tid,
                "subject": ticket["subject"],
                "true_category": ticket["category"],
                "true_urgency": ticket["urgency"],
                "baseline_category": b.get("category", "?"),
                "baseline_urgency": b.get("urgency", "?"),
                "llm_category": lp.get("category", "?"),
                "llm_urgency": lp.get("urgency", "?"),
                "score": score,
            })
    rows.sort(key=lambda x: -x["score"])
    return rows[:n]


def save_confusion_matrix(
    true: list[str],
    pred: list[str],
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    cm = confusion_matrix(true, pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(
    metrics: dict,
    avg_latency: float,
    cost_per_1k: float,
    disagreements: list[dict],
    reports_path: Path = REPORTS_PATH,
) -> None:
    reports_path.mkdir(parents=True, exist_ok=True)
    b, lm = metrics["baseline"], metrics["llm"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Classifier Evaluation Report",
        "",
        f"Generated: {now}  ",
        "Eval set: 50 tickets (stratified by category, seed=42)",
        "",
        "## Summary",
        "",
        f"| Metric | Baseline (TF-IDF + LR) | LLM ({llm.MODEL}) |",
        "|---|---|---|",
        f"| Cat accuracy | {b['category']['accuracy']:.3f} | {lm['category']['accuracy']:.3f} |",
        f"| Cat macro F1 | {b['category']['macro_f1']:.3f} | {lm['category']['macro_f1']:.3f} |",
        f"| Urgency accuracy  | {b['urgency']['accuracy']:.3f} | {lm['urgency']['accuracy']:.3f} |",
        f"| Urgency macro F1  | {b['urgency']['macro_f1']:.3f} | {lm['urgency']['macro_f1']:.3f} |",
        f"| Avg latency / ticket | — | {avg_latency:.2f} s |",
        f"| Est. cost / 1,000 tickets | — | ${cost_per_1k:.3f} |",
        "",
        "> Cost estimate uses haiku-tier pricing "
        f"(${_INPUT_PRICE_PER_M:.2f}/M input, ${_OUTPUT_PRICE_PER_M:.2f}/M output tokens).",
        "",
        "## Confusion Matrices",
        "",
        "### Category — Baseline",
        "![Category confusion matrix — baseline](confusion_matrix_category_baseline.png)",
        "",
        "### Category — LLM",
        "![Category confusion matrix — LLM](confusion_matrix_category_llm.png)",
        "",
        "### Urgency — Baseline",
        "![Urgency confusion matrix — baseline](confusion_matrix_urgency_baseline.png)",
        "",
        "### Urgency — LLM",
        "![Urgency confusion matrix — LLM](confusion_matrix_urgency_llm.png)",
        "",
        "## Top Disagreements (LLM ≠ Baseline)",
        "",
        "| Ticket | Subject | Ground truth | Baseline | LLM |",
        "|---|---|---|---|---|",
    ]

    for d in disagreements:
        tid = d["ticket_id"][:8] + "…"
        subj = d["subject"][:45].replace("|", "\\|")
        truth = f"{d['true_category']} / {d['true_urgency']}"
        base = f"{d['baseline_category']} / {d['baseline_urgency']}"
        lm_pred = f"{d['llm_category']} / {d['llm_urgency']}"
        lines.append(f"| {tid} | {subj} | {truth} | {base} | {lm_pred} |")

    if not disagreements:
        lines.append("| — | Perfect agreement on all tickets | — | — | — |")

    (reports_path / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    import os

    import anthropic

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    eval_df = pl.read_csv(EVAL_PATH)
    tickets = eval_df.to_dicts()
    true_cats = eval_df["category"].to_list()
    true_urgs = eval_df["urgency"].to_list()

    # Baseline — train from silver (excluding eval) if models absent
    cat_model, urg_model = baseline.load_or_train()
    baseline_preds = baseline.predict(tickets, cat_model, urg_model)

    # LLM — one call per ticket, track latency and token usage
    client = anthropic.Anthropic()
    llm_preds: list[dict] = []
    token_usage: list[dict] = []

    for ticket in tickets:
        start = time.perf_counter()
        try:
            result = llm.classify(ticket, client=client, _usage=token_usage)
        except llm.LLMUnavailableError as exc:
            logging.getLogger(__name__).warning(
                "LLM failed for ticket %s during evaluation, using fallback: %s",
                ticket.get("ticket_id"), exc,
            )
            result = dict(llm._FALLBACK)
        result["ticket_id"] = ticket["ticket_id"]
        result["latency"] = time.perf_counter() - start
        llm_preds.append(result)

    avg_latency = sum(r["latency"] for r in llm_preds) / len(llm_preds)

    if token_usage:
        avg_in = sum(u["input_tokens"] for u in token_usage) / len(token_usage)
        avg_out = sum(u["output_tokens"] for u in token_usage) / len(token_usage)
        cost_per_1k = 1_000 * (
            avg_in * _INPUT_PRICE_PER_M + avg_out * _OUTPUT_PRICE_PER_M
        ) / 1_000_000
    else:
        cost_per_1k = 0.0

    baseline_cats = [p["category"] for p in baseline_preds]
    baseline_urgs = [p["urgency"] for p in baseline_preds]
    llm_cats = [p["category"] for p in llm_preds]
    llm_urgs = [p["urgency"] for p in llm_preds]

    metrics = {
        "baseline": {
            "category": compute_metrics(true_cats, baseline_cats),
            "urgency": compute_metrics(true_urgs, baseline_urgs),
        },
        "llm": {
            "category": compute_metrics(true_cats, llm_cats),
            "urgency": compute_metrics(true_urgs, llm_urgs),
        },
    }

    save_confusion_matrix(
        true_cats, baseline_cats, CATEGORY_LABELS,
        "Category — Baseline", REPORTS_PATH / "confusion_matrix_category_baseline.png",
    )
    save_confusion_matrix(
        true_cats, llm_cats, CATEGORY_LABELS,
        "Category — LLM", REPORTS_PATH / "confusion_matrix_category_llm.png",
    )
    save_confusion_matrix(
        true_urgs, baseline_urgs, URGENCY_LABELS,
        "Urgency — Baseline", REPORTS_PATH / "confusion_matrix_urgency_baseline.png",
    )
    save_confusion_matrix(
        true_urgs, llm_urgs, URGENCY_LABELS,
        "Urgency — LLM", REPORTS_PATH / "confusion_matrix_urgency_llm.png",
    )

    disagreements = find_top_disagreements(eval_df, baseline_preds, llm_preds)
    write_report(metrics, avg_latency, cost_per_1k, disagreements)

    print("Report written to reports/evaluation.md")
    for name, m in metrics.items():
        print(
            f"{name:10s}  cat acc={m['category']['accuracy']:.3f} "
            f"f1={m['category']['macro_f1']:.3f}  "
            f"urg acc={m['urgency']['accuracy']:.3f} "
            f"f1={m['urgency']['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
