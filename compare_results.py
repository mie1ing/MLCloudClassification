#!/usr/bin/env python3
"""Compare predicted and ground truth classifications and output a contingency table.

This script accepts two CSV files:
1. Predictions CSV (default assumes a header row): image name, predicted class, score.
2. Ground truth CSV (no header): image name, actual class, annotation.

Only images present in both files are compared.

Usage:
    python compare_results.py --pred test_results.csv --truth ground_truth.csv --out table.csv
    --plot --plot-out table.png

The resulting contingency table is written to a CSV file with actual classes as rows
and predicted classes as columns. Class labels always follow ``CLASS_NAMES`` from
``config.py``, and a prediction accuracy statistic is appended to the output.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

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


def save_contingency(table: Dict[str, Dict[str, int]], path: str) -> Tuple[List[str], List[str], np.ndarray, float]:
    """Save contingency table to a CSV file and output prediction accuracy.

    Returns the class labels, table as an array and accuracy for optional plotting.
    """

    if not table:
        print("No overlapping images found.")
        return CLASS_NAMES, CLASS_NAMES, np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int), 0.0

    actual_classes = CLASS_NAMES
    pred_classes = CLASS_NAMES
    data: List[List[int]] = []

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Actual\\Pred'] + pred_classes)
        for actual in actual_classes:
            row_data = [table.get(actual, {}).get(pred, 0) for pred in pred_classes]
            data.append(row_data)
            writer.writerow([actual] + row_data)

        total = sum(table.get(a, {}).get(p, 0) for a in CLASS_NAMES for p in CLASS_NAMES)
        correct = sum(table.get(a, {}).get(a, 0) for a in CLASS_NAMES)
        accuracy = correct / total if total else 0.0

        writer.writerow([])
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])

    print(f"Prediction accuracy: {accuracy:.4f}")
    print(f"Contingency table saved to {path}")
    return actual_classes, pred_classes, np.array(data), accuracy


def plot_table(actual: List[str], pred: List[str], table: np.ndarray,
               accuracy: float, out: str | None, show: bool) -> None:
    """Plot the contingency table as a heatmap."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = table.astype(float).copy()
    mask = np.ones_like(disp, dtype=bool)
    np.fill_diagonal(mask, False)
    disp[mask] *= -1

    M = float(np.max(np.abs(disp))) if disp.size else 1.0
    if M == 0.0:
        M = 1.0
    norm = TwoSlopeNorm(vmin=-M, vcenter=0.0, vmax=M)

    im = ax.imshow(disp, cmap='RdBu', norm=norm)

    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(pred, rotation=45, ha='right')
    ax.set_yticks(range(len(actual)))
    ax.set_yticklabels(actual)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table[i, j], ha='center', va='center', color='black')

    diag_len = min(table.shape[0], table.shape[1])
    for k in range(diag_len):
        rect = Rectangle((k - 0.5, k - 0.5), 1, 1,
                         fill=False, edgecolor='blue', linewidth=2.5, zorder=3)
        ax.add_patch(rect)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    title = 'Contingency Table'
    title += f' (Accuracy: {accuracy:.4f})'
    ax.set_title(title)

    fig.tight_layout()

    if out:
        plt.savefig(out)
        print(f'Saved plot to {out}')
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contingency table for predictions.")
    parser.add_argument('--pred', required=True, help='CSV file with predictions (has header).')
    parser.add_argument('--truth', required=True, help='CSV file with ground truth (no header).')
    parser.add_argument('--out', default='contingency_table.csv', help='Output CSV file for the contingency table.')
    parser.add_argument('--plot', action='store_true', help='Generate a plot of the contingency table.')
    parser.add_argument('--plot-out', help='Output image path for the contingency table plot.')
    parser.add_argument('--show-plot', action='store_true', help='Display the plot interactively.')
    args = parser.parse_args()

    pred_mapping = load_csv(args.pred, has_header=True)
    truth_mapping = load_csv(args.truth, has_header=False)

    table = build_contingency(pred_mapping, truth_mapping)
    actual, pred, data, acc = save_contingency(table, args.out)
    if args.plot:
        plot_table(actual, pred, data, acc, args.plot_out, args.show_plot)


if __name__ == '__main__':
    main()
