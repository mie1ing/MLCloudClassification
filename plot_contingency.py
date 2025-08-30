#!/usr/bin/env python3
"""Plot contingency table produced by compare_results.py."""

import argparse
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm


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

    # 1) 上色前将非对角元素取相反数（仅用于色彩映射）
    disp = table.astype(float).copy()
    mask = np.ones_like(disp, dtype=bool)
    # 对于非方阵，fill_diagonal 也会按 min(n_rows, n_cols) 处理主对角
    np.fill_diagonal(mask, False)
    disp[mask] *= -1

    # 2) 以 0 为中点的归一化，使用对称范围 [-M, M]
    M = float(np.max(np.abs(disp))) if disp.size else 1.0
    if M == 0.0:
        M = 1.0
    norm = TwoSlopeNorm(vmin=-M, vcenter=0.0, vmax=M)

    # 使用适合正负值的发散色图
    im = ax.imshow(disp, cmap='RdBu', norm=norm)

    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(pred, rotation=45, ha='right')
    ax.set_yticks(range(len(actual)))
    ax.set_yticklabels(actual)

    # 单元格文本仍显示原始计数（不带符号）
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table[i, j], ha='center', va='center', color='black')

    # 高亮对角线单元格：为每个对角格添加红色描边框
    diag_len = min(table.shape[0], table.shape[1])
    for k in range(diag_len):
        rect = Rectangle(
            (k - 0.5, k - 0.5),
            1, 1,
            fill=False,
            edgecolor='blue',
            linewidth=2.5,
            zorder=3
        )
        ax.add_patch(rect)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    title = 'Contingency Table'
    if accuracy is not None:
        title += f' (Accuracy: {accuracy:.4f})'
    ax.set_title(title)

    # 3) 最终成图不显示 colorbar（故移除 fig.colorbar）
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