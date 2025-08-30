#!/usr/bin/env python3
"""Plot contingency table produced by compare_results.py."""

import argparse
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_table(path: str) -> Tuple[List[str], List[str], np.ndarray, float | None]:
    """Load contingency table and accuracy from CSV file."""
    actual_classes: List[str] = []
    pred_classes: List[str] = []
    data: List[List[int]] = []
    accuracy: float | None = None

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if len(header) < 2:
            raise ValueError("Invalid contingency table format")
        pred_classes = header[1:]

        for row in reader:
            if not row or not row[0] or row[0] == 'Statistic':
                break
            actual_classes.append(row[0])
            data.append([int(x) for x in row[1:]])

        for row in reader:
            if row and row[0] == 'Accuracy' and len(row) > 1:
                try:
                    accuracy = float(row[1])
                except ValueError:
                    accuracy = None
                break

    return actual_classes, pred_classes, np.array(data), accuracy


def plot_table(actual: List[str], pred: List[str], table: np.ndarray,
               accuracy: float | None, out: str | None, show: bool) -> None:
    """Plot the contingency table as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(table, cmap='Blues')

    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(pred, rotation=45, ha='right')
    ax.set_yticks(range(len(actual)))
    ax.set_yticklabels(actual)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table[i, j], ha='center', va='center', color='black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    title = 'Contingency Table'
    if accuracy is not None:
        title += f' (Accuracy: {accuracy:.4f})'
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if out:
        plt.savefig(out)
        print(f'Saved plot to {out}')
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot contingency table CSV')
    parser.add_argument('--table', required=True, help='CSV file produced by compare_results.py')
    parser.add_argument('--out', help='Output image path (e.g., table.png)')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively')
    args = parser.parse_args()

    actual, pred, data, acc = load_table(args.table)
    plot_table(actual, pred, data, acc, args.out, args.show)


if __name__ == '__main__':
    main()
