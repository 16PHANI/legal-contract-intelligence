import argparse
import json
import logging
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]
SKIP_SENTINELS = {"NOT_FOUND", "EXTRACTION_FAILED", "SUMMARIZATION_FAILED", ""}


def _normalize(text):
    """Normalize text for token-level comparison (SQuAD/CUAD protocol)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction, ground_truth):
    """Compute token-level F1 between prediction and one ground-truth span."""
    pred_tokens = _normalize(prediction).split()
    gt_tokens   = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_clause(prediction, ground_truth_spans):
    """Score prediction against multiple GT spans; return max F1."""
    if prediction in SKIP_SENTINELS:
        return 0.0
    if not ground_truth_spans:
        return None
    return max(_token_f1(prediction, gt) for gt in ground_truth_spans)


def evaluate_results(results, contracts):
    """Evaluate pipeline output against CUAD ground-truth annotations.

    Uses Token F1 (the CUAD standard metric) which gives partial credit
    for correct clause text with minor boundary differences.
    """
    gt_lookup = {c["id"]: c.get("ground_truth_answers", {}) for c in contracts
                 if c.get("ground_truth_answers")}

    per_contract_scores = []
    clause_type_scores = {ct: [] for ct in CLAUSE_TYPES}
    n_skipped = 0

    for row in results:
        cid = row["contract_id"]
        gt = gt_lookup.get(cid)
        if gt is None:
            n_skipped += 1
            continue

        scores = {}
        for ct in CLAUSE_TYPES:
            prediction = row.get(ct, "")
            f1 = score_clause(prediction, gt.get(ct, []))
            scores[ct] = f1
            if f1 is not None:
                clause_type_scores[ct].append(f1)

        valid = [s for s in scores.values() if s is not None]
        per_contract_scores.append({
            "contract_id":        cid,
            "termination_f1":     scores.get("termination_clause"),
            "confidentiality_f1": scores.get("confidentiality_clause"),
            "liability_f1":       scores.get("liability_clause"),
            "mean_f1":            sum(valid) / len(valid) if valid else None,
        })

    per_clause_avg = {
        ct: (sum(s) / len(s) if s else 0.0)
        for ct, s in clause_type_scores.items()
    }
    all_f1 = [f for s in clause_type_scores.values() for f in s]
    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0

    return {
        "per_contract":    per_contract_scores,
        "per_clause_type": per_clause_avg,
        "overall_f1":      round(overall_f1, 4),
        "n_evaluated":     len(per_contract_scores),
        "n_skipped":       n_skipped,
    }


def print_report(report):
    print("\n" + "=" * 50)
    print("Evaluation Report - Token F1 (CUAD metric)")
    print("=" * 50)
    print(f"  Contracts evaluated : {report['n_evaluated']}")
    print(f"  Contracts skipped   : {report['n_skipped']} (no ground truth)")
    pt = report["per_clause_type"]
    print(f"  Termination     F1  : {pt['termination_clause']:.4f}")
    print(f"  Confidentiality F1  : {pt['confidentiality_clause']:.4f}")
    print(f"  Liability       F1  : {pt['liability_clause']:.4f}")
    print(f"  Overall         F1  : {report['overall_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Evaluate pipeline results against CUAD ground truth")
    parser.add_argument("--results", default="output/results.json")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    from loader import load_from_huggingface
    contracts = load_from_huggingface(n_contracts=args.n)
    report = evaluate_results(results, contracts)
    print_report(report)

    report_path = args.results.replace(".json", "_eval.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
