#!/usr/bin/env python3
"""
Evaluation Metrics — FIN-Debate
================================
Computes accuracy, macro precision/recall/F1, per-class metrics,
and confusion matrices for all experiments. Exports summary CSV.

Input CSV must have columns: ground_truth, zero_shot, bm25_rag, faiss_rag, delphi, full_pipeline

Usage:
    python compute_metrics.py --input data/all_verdicts_300.csv
    python compute_metrics.py --input data/all_verdicts_300.csv --output-dir results/

Requirements: pip install pandas scikit-learn tabulate
"""

import argparse
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from tabulate import tabulate

EXPERIMENTS = {
    "Zero-Shot":     "zero_shot",
    "BM25 RAG":      "bm25_rag",
    "FAISS RAG":     "faiss_rag",
    "Delphi":        "delphi",
    "Full Pipeline": "full_pipeline",
}
LABELS = ["TRUE", "FALSE", "HALF TRUE"]
GT_COL = "ground_truth"


def normalize_verdict(v):
    v = str(v).upper().strip().replace("-", " ")
    if "HALF" in v:   return "HALF TRUE"
    if "TRUE" in v:   return "TRUE"
    if "FALSE" in v:  return "FALSE"
    return v


def compute_all(df):
    """Compute metrics for each experiment; return (overall_rows, classwise_rows, export_rows)."""
    y_true = df[GT_COL]
    overall, classwise, export = [], [], []

    for name, col in EXPERIMENTS.items():
        y_pred = df[col]
        acc = accuracy_score(y_true, y_pred)
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        mp, mr, mf1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

        overall.append([name, f"{acc*100:.1f}", f"{mp*100:.1f}", f"{mr*100:.1f}", f"{mf1*100:.1f}"])

        for i, label in enumerate(LABELS):
            classwise.append([name, label, f"{p_cls[i]*100:.1f}", f"{r_cls[i]*100:.1f}", f"{f1_cls[i]*100:.1f}"])
            export.append({"method": name, "class": label, "accuracy": round(acc*100, 1),
                           "precision": round(p_cls[i]*100, 1), "recall": round(r_cls[i]*100, 1),
                           "f1": round(f1_cls[i]*100, 1)})
        classwise.append([name, "Macro Avg", f"{mp*100:.1f}", f"{mr*100:.1f}", f"{mf1*100:.1f}"])
        export.append({"method": name, "class": "Macro Avg", "accuracy": round(acc*100, 1),
                        "precision": round(mp*100, 1), "recall": round(mr*100, 1), "f1": round(mf1*100, 1)})

    return overall, classwise, export


def main():
    parser = argparse.ArgumentParser(description="Compute fact-checking evaluation metrics")
    parser.add_argument("--input", required=True, help="CSV with ground_truth + experiment verdict columns")
    parser.add_argument("--output-dir", default=".", help="Directory for metrics_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} claims from {args.input}\n")

    # Normalize all verdict columns
    df[GT_COL] = df[GT_COL].apply(normalize_verdict)
    for col in EXPERIMENTS.values():
        df[col] = df[col].apply(normalize_verdict)
    y_true = df[GT_COL]

    overall, classwise, export = compute_all(df)

    # Table 1: Overall
    print(tabulate(overall, headers=["Method", "Accuracy (%)", "Macro P (%)", "Macro R (%)", "Macro F1 (%)"],
                   tablefmt="grid", stralign="center"))
    print()

    # Table 2: Per-class
    print(tabulate(classwise, headers=["Method", "Class", "Precision (%)", "Recall (%)", "F1 (%)"],
                   tablefmt="grid", stralign="center"))
    print()

    # Table 3: Confusion matrices
    for name, col in EXPERIMENTS.items():
        cm = confusion_matrix(y_true, df[col], labels=LABELS)
        print(f"\n{name}:")
        print(tabulate(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                        headers="keys", tablefmt="grid", showindex=True))

    # Table 4: Full classification reports
    for name, col in EXPERIMENTS.items():
        print(f"\n{name}:")
        print(classification_report(y_true, df[col], labels=LABELS, digits=3, zero_division=0))

    # Export CSV
    import os
    out_path = os.path.join(args.output_dir, "metrics_summary.csv")
    pd.DataFrame(export).to_csv(out_path, index=False)
    print(f"\nExported to {out_path}")


if __name__ == "__main__":
    main()
