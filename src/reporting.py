# -*- coding: utf-8 -*-
"""Generate interactive line charts from ``ratings.json`` files.

This module scans experiment output directories for ``ratings.json`` files,
flattens the rating metrics, and produces an interactive line chart for each
image. The resulting visualization is saved as ``line_overview.html`` in the
corresponding ``eval`` folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import altair as alt
import pandas as pd

# Map possible metric keys to canonical metric names.
_METRIC_KEYS: Dict[str, str] = {
    "content_correspondence": "content_correspondence",
    "compositional_alignment": "compositional_alignment",
    "fidelity_completeness": "fidelity_completeness",
    "style_consistency": "style_consistency",
    "stylistic_congruence": "style_consistency",
    "overall": "overall",
    "overall_semantic_intent": "overall",
}


def _flatten_records(data: Sequence[Dict]) -> pd.DataFrame:
    """Flatten raw rating records into a tidy DataFrame."""
    rows: List[Dict[str, object]] = []
    for rec in data:
        step = rec.get("step")
        comparison_type = rec.get("comparison_type")
        anchor = rec.get("anchor")
        for key, metric in _METRIC_KEYS.items():
            metric_data = rec.get(key)
            if isinstance(metric_data, dict) and "score" in metric_data:
                rows.append(
                    {
                        "step": step,
                        "comparison_type": comparison_type,
                        "anchor": anchor,
                        "metric": metric,
                        "score": metric_data["score"],
                    }
                )
    return pd.DataFrame(rows)


def _build_chart(df: pd.DataFrame) -> alt.FacetChart:
    """Create an Altair line chart from flattened ratings."""
    base = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "step:O",
                sort=alt.SortField("step", order="ascending"),
                title="Step",
            ),
            y=alt.Y(
                "score:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 10]),
            ),
            color=alt.Color("metric:N", title="Metric"),
            strokeDash=alt.StrokeDash("anchor:N", title="Anchor"),
            tooltip=[
                "step:O",
                "metric:N",
                "score:Q",
                "comparison_type:N",
                "anchor:N",
            ],
        )
    )
    facet = base.facet(column=alt.Column("comparison_type:N", title=None))
    return facet.interactive()


def generate_line_overview(exp_root: str | Path) -> None:
    """Generate ``line_overview.html`` for each image under ``exp_root``."""
    root = Path(exp_root)
    for item_dir in root.iterdir():
        if not item_dir.is_dir():
            continue
        ratings_path = item_dir / "eval" / "ratings.json"
        if not ratings_path.is_file():
            continue
        data = json.loads(ratings_path.read_text())
        df = _flatten_records(data)
        if df.empty:
            continue
        chart = _build_chart(df)
        chart.save(ratings_path.parent / "line_overview.html")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive line charts from ratings.json files"
    )
    parser.add_argument(
        "exp_root",
        help="Experiment directory containing image subfolders",
    )
    args = parser.parse_args()
    generate_line_overview(args.exp_root)
