#!/usr/bin/env python3
"""Compare predicted and ground truth classifications and output a contingency table.

This script accepts two CSV files:
1. Predictions CSV (default assumes a header row): image name, predicted class, score.
2. Ground truth CSV (no header): image name, actual class, annotation.

Only images present in both files are compared.

Usage:
    python compare_results.py --pred test_results.csv --truth ground_truth.csv --out table.csv

The resulting contingency table is written to a CSV file with actual classes as rows
and predicted classes as columns. Class labels always follow ``CLASS_NAMES`` from
``config.py``, and a prediction accuracy statistic is appended to the output.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict

from config import CLASS_NAMES


def load_csv(path: str, has_header: bool) -> Dict[str, str]:
    """Load CSV mapping image name to class.

    Args:
        path: Path to CSV file.
        has_header: Whether the first row is a header.

    Returns:
        Dictionary mapping image name to class label.
    """
    mapping: Dict[str, str] = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            mapping[row[0]] = row[1]
    return mapping


def build_contingency(pred: Dict[str, str], truth: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """Build contingency table counts constrained to known classes."""
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    common_keys = pred.keys() & truth.keys()
    for key in common_keys:
        actual = truth[key]
        predicted = pred[key]
        if actual in CLASS_NAMES and predicted in CLASS_NAMES:
            table[actual][predicted] += 1
    return table


def save_contingency(table: Dict[str, Dict[str, int]], path: str) -> None:
    """Save contingency table to a CSV file and output prediction accuracy."""

    if not table:
        print("No overlapping images found.")
        return

    actual_classes = CLASS_NAMES
    pred_classes = CLASS_NAMES

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Actual\\Pred'] + pred_classes)
        for actual in actual_classes:
            row = [actual] + [table.get(actual, {}).get(pred, 0) for pred in pred_classes]
            writer.writerow(row)

        total = sum(table.get(a, {}).get(p, 0) for a in CLASS_NAMES for p in CLASS_NAMES)
        correct = sum(table.get(a, {}).get(a, 0) for a in CLASS_NAMES)
        accuracy = correct / total if total else 0.0

        writer.writerow([])
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])

    print(f"Prediction accuracy: {accuracy:.4f}")
    print(f"Contingency table saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contingency table for predictions.")
    parser.add_argument('--pred', required=True, help='CSV file with predictions (has header).')
    parser.add_argument('--truth', required=True, help='CSV file with ground truth (no header).')
    parser.add_argument('--out', default='contingency_table.csv', help='Output CSV file for the contingency table.')
    args = parser.parse_args()

    pred_mapping = load_csv(args.pred, has_header=True)
    truth_mapping = load_csv(args.truth, has_header=False)

    table = build_contingency(pred_mapping, truth_mapping)
    save_contingency(table, args.out)


if __name__ == '__main__':
    main()
