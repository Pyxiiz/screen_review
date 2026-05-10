from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from two_stage_screen.csv_io import read_csv_rows


def normalize_label(s: str) -> str:
    return (s or "").strip().lower()


def compare_gold_predictions(
    gold_path: Path,
    pred_path: Path,
    *,
    id_column: str,
    gold_column: str,
    pred_column: str,
) -> dict:
    _, gold_rows = read_csv_rows(gold_path)
    _, pred_rows = read_csv_rows(pred_path)
    gold_by_id = {}
    for r in gold_rows:
        rid = normalize_label(r.get(id_column, ""))
        if rid:
            gold_by_id[rid] = normalize_label(r.get(gold_column, ""))
    pred_by_id = {}
    for r in pred_rows:
        rid = normalize_label(r.get(id_column, ""))
        if rid:
            pred_by_id[rid] = normalize_label(r.get(pred_column, ""))

    paired = sorted(set(gold_by_id) & set(pred_by_id))
    missing_pred = sorted(set(gold_by_id) - set(pred_by_id))
    missing_gold = sorted(set(pred_by_id) - set(gold_by_id))

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    valid_decisions = {"include", "exclude", "unclear"}
    skipped = []
    correct = 0
    total = 0
    for rid in paired:
        g = gold_by_id[rid]
        p = pred_by_id[rid]
        if not g or not p:
            skipped.append({"id": rid, "reason": "empty_label"})
            continue
        if g not in valid_decisions:
            skipped.append({"id": rid, "reason": f"gold_not_in_allowlist:{g}"})
            continue
        if p not in valid_decisions:
            skipped.append({"id": rid, "reason": f"pred_not_in_allowlist:{p}"})
            continue
        confusion[(g, p)] += 1
        total += 1
        if g == p:
            correct += 1

    accuracy = (correct / total) if total else 0.0
    marginal_gold = Counter()
    marginal_pred = Counter()
    for (g, p), cnt in confusion.items():
        marginal_gold[g] += cnt
        marginal_pred[p] += cnt

    macro_recall_by_gold_class: dict[str, float | None] = {}
    classes = sorted({k[0] for k in confusion} | {"include", "exclude", "unclear"})
    for c in classes:
        tp = confusion.get((c, c), 0)
        denom = marginal_gold.get(c, 0)
        macro_recall_by_gold_class[c] = (tp / denom) if denom else None

    return {
        "paired_rows": len(paired),
        "evaluated_pairs": total,
        "accuracy": accuracy,
        "confusion_pairs": [{"gold": k[0], "pred": k[1], "count": v} for k, v in sorted(confusion.items())],
        "marginal_counts_gold": dict(marginal_gold),
        "marginal_counts_pred": dict(marginal_pred),
        "macro_recall_given_gold": macro_recall_by_gold_class,
        "skipped_pairs": skipped,
        "missing_in_predictions": missing_pred,
        "missing_in_gold": missing_gold,
    }
