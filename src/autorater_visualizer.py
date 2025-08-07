# -*- coding: utf-8 -*-
"""Utilities for visualizing evaluation results.

This module reads the ``ratings.json`` files produced by
:class:`EvaluationEngine` and generates charts under each item's ``eval``
folder. The first visualization implemented is a multi-line plot showing how
scores evolve over iterations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


_METRICS = [
    "content_correspondence",
    "compositional_alignment",
    "fidelity_completeness",
    "stylistic_congruence",
    "overall_semantic_intent",
]


class AutoRaterVisualizer:
    """Create plots from ``ratings.json`` produced by :class:`EvaluationEngine`."""

    def __init__(self, ratings_path: str | Path) -> None:
        self.ratings_path = Path(ratings_path)
        self.data: List[Dict] = json.loads(self.ratings_path.read_text())

    # ------------------------------------------------------------------
    def plot_score_trajectories(self, group_by: str = "comparison_type") -> Path:
        """Plot metric scores over iteration steps.

        Parameters
        ----------
        group_by:
            Either ``"comparison_type"`` or ``"anchor"``. Entries are grouped
            by this field and plotted in separate subplots.
        """
        if group_by not in {"comparison_type", "anchor"}:
            raise ValueError("group_by must be 'comparison_type' or 'anchor'")

        groups: Dict[str, List[Dict]] = {}
        for rec in self.data:
            groups.setdefault(rec[group_by], []).append(rec)

        n = len(groups)
        fig, axes = plt.subplots(n, 1, figsize=(7, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, (group_val, items) in zip(axes, groups.items()):
            steps = sorted({rec["step"] for rec in items})
            for metric in _METRICS:
                series = []
                for step in steps:
                    vals = [rec[metric]["score"] for rec in items if rec["step"] == step]
                    avg = sum(vals) / len(vals) if vals else float("nan")
                    series.append(avg)
                ax.plot(steps, series, marker="o", label=metric.replace("_", " "))

            ax.set_title(f"{group_by}: {group_val}")
            ax.set_xlabel("Iteration Step")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 10)
            ax.legend(loc="lower right")

        fig.tight_layout()
        out_path = self.ratings_path.parent / f"score_trajectories_{group_by}.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def generate_visualizations(exp_root: str, group_by: str = "comparison_type") -> None:
    """Generate visualizations for every item under ``exp_root``.

    Parameters
    ----------
    exp_root:
        Path to the experiment directory that contains item subfolders.
    group_by:
        Field to group by when plotting. Passed to
        :meth:`AutoRaterVisualizer.plot_score_trajectories`.
    """
    root = Path(exp_root)
    for item_dir in root.iterdir():
        ratings = item_dir / "eval" / "ratings.json"
        if ratings.is_file():
            vis = AutoRaterVisualizer(ratings)
            vis.plot_score_trajectories(group_by=group_by)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("exp_root", help="Experiment root directory")
    parser.add_argument(
        "--group-by",
        choices=["comparison_type", "anchor"],
        default="comparison_type",
        dest="group_by",
        help="Group results by this field",
    )
    args = parser.parse_args()
    generate_visualizations(args.exp_root, group_by=args.group_by)
