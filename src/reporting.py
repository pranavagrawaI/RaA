# -*- coding: utf-8 -*-
"""Generate a single, self-contained HTML report from `ratings.json` files."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Template

# --- VEGA LIBRARIES (as string constants) ---
VEGA_LITE_URL = "https://cdn.jsdelivr.net/npm/vega-lite@5"
VEGA_URL = "https://cdn.jsdelivr.net/npm/vega@5"
VEGA_EMBED_URL = "https://cdn.jsdelivr.net/npm/vega-embed@6"

VEGA_LITE_JS = f'<script src="{VEGA_LITE_URL}"></script>'
VEGA_JS = f'<script src="{VEGA_URL}"></script>'
VEGA_EMBED_JS = f'<script src="{VEGA_EMBED_URL}"></script>'


# --- JINJA2 HTML TEMPLATE ---
HTML_TEMPLATE_STR = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    {% if not static_charts %}
    {{ vega_js }}
    {{ vega_lite_js }}
    {{ vega_embed_js }}
    {% endif %}
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; display: flex; }
        .nav { width: 220px; background-color: #fff; border-right: 1px solid #dee2e6; padding: 1rem; height: 100vh; position: fixed; }
        .nav a { display: block; padding: 0.5rem 1rem; color: #495057; text-decoration: none; border-radius: 0.25rem; margin-bottom: 0.5rem; }
        .nav a:hover { background-color: #e9ecef; }
        .container { margin-left: 240px; width: calc(100% - 240px); padding: 2rem; }
        .header { border-bottom: 1px solid #dee2e6; padding-bottom: 1rem; margin-bottom: 2rem; }
        h1, h2 { color: #343a40; }
        h1 { font-size: 2.5rem; }
        h2 { font-size: 1.75rem; border-bottom: 1px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 3rem; }
        .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card { background-color: #fff; border: 1px solid #dee2e6; border-radius: 0.25rem; padding: 1.5rem; text-align: center; }
        .card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #6c757d; }
        .card .value { font-size: 2rem; font-weight: bold; color: #495057; }
        .top-drops table { width: 100%; border-collapse: collapse; }
        .top-drops th, .top-drops td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
        .top-drops th { background-color: #f8f9fa; }
        .top-drops tbody tr:hover { background-color: #e9ecef; cursor: pointer; }
        footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 0.875rem; }
        .chart-container { text-align: center; }
        .chart-container img { max-width: 100%; height: auto; }
        .dark { background-color: #212529; color: #f8f9fa; }
        .dark .nav { background-color: #343a40; border-right-color: #495057; }
        .dark .nav a { color: #adb5bd; }
        .dark .nav a:hover { background-color: #495057; }
        .dark h1, .dark h2 { color: #e9ecef; }
        .dark .card { background-color: #343a40; border-color: #495057; }
        .dark .card h3 { color: #adb5bd; }
        .dark .card .value { color: #f8f9fa; }
        .dark .top-drops th { background-color: #343a40; }
        .dark .top-drops tbody tr:hover { background-color: #495057; }
    </style>
</head>
<body class="{{ theme }}">
    <div class="nav">
        <h3>Navigation</h3>
        <a href="#summary">Summary</a>
        <a href="#score-trends">Score Trends</a>
        <a href="#availability">Data Availability</a>
        <a href="#anchor-vs-metric">Anchor vs. Metric</a>
        <a href="#deltas">Step-to-Step Deltas</a>
        <a href="#stability">Stability Bands</a>
        <a href="#top-drops">Top Drops</a>
    </div>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p><strong>Item ID:</strong> {{ item_id }} | <strong>Report Generated:</strong> {{ timestamp }}</p>
        </div>
        
        <h2 id="summary">Summary</h2>
        <div class="summary-cards">
            <div class="card">
                <h3>Coverage</h3>
                <div class="value">{{ "%.1f"|format(summary_cards.coverage * 100) }}%</div>
            </div>
            {% for comp_type, stats in summary_cards.by_type.items() %}
            <div class="card">
                <h3>Best Metric ({{ comp_type }})</h3>
                <div class="value">{{ stats.best_metric }}</div>
            </div>
            <div class="card">
                <h3>Most Volatile ({{ comp_type }})</h3>
                <div class="value">{{ stats.most_volatile_metric }}</div>
            </div>
            <div class="card">
                <h3>Top Anchor ({{ comp_type }})</h3>
                <div class="value">{{ stats.top_anchor }}</div>
            </div>
            {% endfor %}
        </div>

        {% if static_charts %}
            <h2 id="score-trends">Score Trends</h2>
            <div class="chart-container"><img src="data:image/png;base64,{{ static_charts.trends }}" alt="Score Trends"></div>
            <h2 id="availability">Data Availability</h2>
            <div class="chart-container"><img src="data:image/png;base64,{{ static_charts.availability }}" alt="Data Availability"></div>
            <h2 id="anchor-vs-metric">Anchor vs. Metric Performance</h2>
            <div class="chart-container"><img src="data:image/png;base64,{{ static_charts.anchor_vs_metric }}" alt="Anchor vs. Metric"></div>
            <h2 id="deltas">Step-to-Step Deltas</h2>
            <div class="chart-container"><img src="data:image/png;base64,{{ static_charts.deltas }}" alt="Deltas"></div>
            <h2 id="stability">Stability Bands</h2>
            <div class="chart-container"><img src="data:image/png;base64,{{ static_charts.stability }}" alt="Stability"></div>
        {% else %}
            <h2 id="score-trends">Score Trends</h2>
            <div id="vis-score-trends"></div>
            <h2 id="availability">Data Availability</h2>
            <div id="vis-availability"></div>
            <h2 id="anchor-vs-metric">Anchor vs. Metric Performance</h2>
            <div id="vis-anchor-vs-metric"></div>
            <h2 id="deltas">Step-to-Step Deltas</h2>
            <div id="vis-deltas"></div>
            <h2 id="stability">Stability Bands</h2>
            <div id="vis-stability"></div>
        {% endif %}

        <div class="top-drops" id="top-drops">
            <h2>Top Drops</h2>
            <table>
                <thead>
                    <tr>
                        <th>Item ID</th><th>Comparison Type</th><th>Metric</th><th>Anchor</th><th>Step</th><th>Prev Score</th><th>Score</th><th>Delta</th><th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                {% for drop in top_drops %}
                    <tr {% if not static_charts %}onclick="highlightPoint('{{ drop.comparison_type }}|{{ drop.metric }}|{{ drop.anchor }}|{{ drop.step }}')"{% endif %}>
                        <td>{{ drop.item_id }}</td><td>{{ drop.comparison_type }}</td><td>{{ drop.metric }}</td><td>{{ drop.anchor }}</td><td>{{ drop.step }}</td><td>{{ "%.2f"|format(drop.prev_score) }}</td><td>{{ "%.2f"|format(drop.score) }}</td><td>{{ "%.2f"|format(drop.delta) }}</td><td>{{ drop.reason }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p><strong>Input File:</strong> {{ file_path }}</p>
            <p><strong>Input Hash (SHA-256):</strong> {{ file_hash }}</p>
        </footer>
    </div>
    {% if not static_charts %}
    <script>
        const vegaSpecs = {{ vega_specs | tojson }};
        const vegaData = {{ vega_data | tojson }};
        let vegaView;

        function highlightPoint(key) {
            if (vegaView) {
                vegaView.signal('highlightKey', key).run();
                const el = document.getElementById('vis-score-trends');
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
            const options = {
                actions: true,
                renderer: 'canvas',
                theme: '{{ theme }}' === 'dark' ? 'dark' : 'default'
            };
            
            function embedChart(id, spec, data) {
                const el = document.getElementById(id);
                if (!el || !spec) return;
                vegaEmbed(el, spec, options)
                    .then(result => {
                        if (id === 'vis-score-trends') {
                            vegaView = result.view;
                        }
                        result.view.data('source_data', data);
                        result.view.run();
                    })
                    .catch(console.error);
            }

            embedChart('vis-score-trends', vegaSpecs.trends, vegaData.trends);
            embedChart('vis-availability', vegaSpecs.availability, vegaData.availability);
            embedChart('vis-anchor-vs-metric', vegaSpecs.anchor_vs_metric, vegaData.anchor_vs_metric);
            embedChart('vis-deltas', vegaSpecs.deltas, vegaData.deltas);
            embedChart('vis-stability', vegaSpecs.stability, vegaData.stability);
        });

        document.addEventListener('keydown', function(e) {
            if (e.key === '/') {
                e.preventDefault();
                // This is a placeholder for a search/filter input
                console.log('Search/filter triggered');
            }
        });
    </script>
    {% endif %}
</body>
</html>
"""
HTML_TEMPLATE = Template(HTML_TEMPLATE_STR)

# Map possible metric keys to canonical metric names.
_METRIC_KEYS: Dict[str, str] = {
    "compositional_alignment": "compositional_alignment",
    "content_correspondence": "content_correspondence",
    "fidelity_completeness": "fidelity_completeness",
    "style_consistency": "style_consistency",
    "stylistic_congruence": "style_consistency",
    "overall": "overall",
    "overall_semantic_intent": "overall",
}


def discover_ratings_files(path: Path) -> List[Path]:
    """Find all `ratings.json` files to process."""
    if path.is_file():
        if path.name == "ratings.json":
            return [path]
        else:
            return []
    if path.is_dir():
        return sorted(path.glob("**/eval/ratings.json"))
    return []


def load_and_normalize_data(file_path: Path) -> tuple[pd.DataFrame | None, str]:
    """Load, normalize, and clean data from a `ratings.json` file."""
    if not file_path.is_file():
        return None, ""

    try:
        content = file_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        raw_data = json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None, ""

    records: List[Dict[str, Any]] = []
    for record in raw_data:
        item_id = record.get("item_id", "unknown_item")
        step = record.get("step")
        comparison_type = record.get("comparison_type")
        anchor = record.get("anchor")

        for key, value in record.items():
            if key in _METRIC_KEYS:
                canonical_metric = _METRIC_KEYS[key]
                if isinstance(value, dict) and "score" in value:
                    score = value.get("score", -1)
                    reason = value.get("reason", "")
                    records.append(
                        {
                            "item_id": item_id,
                            "step": step,
                            "comparison_type": comparison_type,
                            "anchor": anchor,
                            "metric": canonical_metric,
                            "score": score,
                            "reason": reason,
                            "valid": score >= 0,
                        }
                    )

    if not records:
        return pd.DataFrame(), file_hash

    df = pd.DataFrame(records)
    df = df.sort_values(by=["comparison_type", "metric", "anchor", "step"]).reset_index(
        drop=True
    )
    return df, file_hash


def calculate_aggregates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all data aggregates needed for the report."""
    if df.empty:
        return {}

    valid_df = df[df["valid"]].copy()
    if valid_df.empty:
        return {"summary_cards": {"coverage": 0}}

    coverage = len(valid_df) / len(df) if len(df) > 0 else 0

    summary_by_type = {}
    for comp_type, group in valid_df.groupby("comparison_type"):
        metric_group = group.groupby("metric")["score"]
        anchor_group = group.groupby("anchor")["score"]
        summary_by_type[comp_type] = {
            "best_metric": metric_group.mean().idxmax(),
            "most_volatile_metric": metric_group.std().idxmax(),
            "top_anchor": anchor_group.mean().idxmax(),
        }

    valid_df["delta"] = (
        valid_df.groupby(["item_id", "comparison_type", "metric", "anchor"])["score"]
        .diff()
        .fillna(0)
    )
    valid_df["prev_score"] = valid_df["score"] - valid_df["delta"]

    top_drops = valid_df.nsmallest(10, "delta").to_dict("records")

    return {
        "summary_cards": {
            "coverage": coverage,
            "by_type": summary_by_type,
        },
        "availability": valid_df.groupby(["comparison_type", "metric", "step"])
        .size()
        .reset_index(name="count"),
        "anchor_vs_metric": valid_df.groupby(["comparison_type", "metric", "anchor"])[
            "score"
        ]
        .mean()
        .reset_index(),
        "deltas": valid_df,
        "stability": valid_df.groupby(["comparison_type", "metric", "step"])["score"]
        .agg(["mean", "std"])
        .reset_index(),
        "top_drops": top_drops,
    }


def generate_vega_specs(df: pd.DataFrame, aggregates: Dict[str, Any]) -> Dict[str, Any]:
    """Generate all Vega-Lite specifications."""
    trends_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Scores by step. Rows=metric, Cols=comparison_type.",
        "params": [
            {
                "name": "anchorFilter",
                "value": [],
                "bind": {
                    "input": "checkbox",
                    "options": df["anchor"].unique().tolist(),
                    "labels": df["anchor"].unique().tolist(),
                },
            },
            {
                "name": "highlightKey",
                "value": "",
                "description": "Format: type|metric|anchor|step; set by Top Drops click handler.",
            },
        ],
        "data": {"name": "source_data"},
        "transform": [
            {"filter": "datum.valid == true"},
            {
                "calculate": "datum.comparison_type + '|' + datum.metric + '|' + datum.anchor + '|' + datum.step",
                "as": "key",
            },
            {
                "filter": "anchorFilter.length == 0 || anchorFilter.indexOf(datum.anchor) >= 0"
            },
        ],
        "facet": {
            "row": {"field": "metric", "type": "ordinal", "title": "Metric"},
            "column": {"field": "comparison_type", "type": "ordinal", "title": "Type"},
        },
        "spec": {
            "width": 260,
            "height": 160,
            "layer": [
                {
                    "mark": {"type": "line"},
                    "encoding": {
                        "x": {
                            "field": "step",
                            "type": "ordinal",
                            "title": "Step",
                            "sort": "ascending",
                        },
                        "y": {
                            "field": "score",
                            "type": "quantitative",
                            "title": "Score",
                            "scale": {"domain": [0, 10]},
                        },
                        "color": {
                            "field": "anchor",
                            "type": "nominal",
                            "title": "Anchor",
                        },
                    },
                },
                {
                    "mark": {"type": "point", "filled": True, "size": 60},
                    "encoding": {
                        "x": {"field": "step", "type": "ordinal"},
                        "y": {"field": "score", "type": "quantitative"},
                        "color": {"field": "anchor", "type": "nominal"},
                        "opacity": {
                            "condition": {
                                "test": "datum.key === highlightKey",
                                "value": 1,
                            },
                            "value": 0.9,
                        },
                        "tooltip": [
                            {"field": "item_id", "type": "nominal", "title": "Item"},
                            {
                                "field": "comparison_type",
                                "type": "nominal",
                                "title": "Type",
                            },
                            {"field": "anchor", "type": "nominal", "title": "Anchor"},
                            {"field": "metric", "type": "nominal", "title": "Metric"},
                            {"field": "step", "type": "ordinal", "title": "Step"},
                            {
                                "field": "score",
                                "type": "quantitative",
                                "format": ".2f",
                                "title": "Score",
                            },
                            {"field": "reason", "type": "nominal", "title": "Reason"},
                        ],
                    },
                },
            ],
        },
    }

    availability_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Data availability heatmap.",
        "data": {"name": "source_data"},
        "facet": {
            "column": {
                "field": "comparison_type",
                "type": "ordinal",
                "title": "Comparison Type",
            }
        },
        "spec": {
            "mark": "rect",
            "encoding": {
                "x": {"field": "step", "type": "ordinal"},
                "y": {"field": "metric", "type": "ordinal"},
                "color": {"field": "count", "type": "quantitative", "title": "Count"},
                "tooltip": [
                    {"field": "comparison_type", "type": "nominal"},
                    {"field": "metric", "type": "nominal"},
                    {"field": "step", "type": "ordinal"},
                    {"field": "count", "type": "quantitative"},
                ],
            },
        },
    }

    anchor_vs_metric_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Mean score by metric and anchor.",
        "data": {"name": "source_data"},
        "facet": {
            "column": {
                "field": "comparison_type",
                "type": "ordinal",
                "title": "Comparison Type",
            }
        },
        "spec": {
            "mark": "bar",
            "encoding": {
                "x": {"field": "metric", "type": "ordinal"},
                "y": {"field": "score", "type": "quantitative", "title": "Mean Score"},
                "xOffset": {"field": "anchor", "type": "nominal"},
                "color": {"field": "anchor", "type": "nominal"},
                "tooltip": [
                    {"field": "comparison_type", "type": "nominal"},
                    {"field": "metric", "type": "nominal"},
                    {"field": "anchor", "type": "nominal"},
                    {"field": "score", "type": "quantitative", "format": ".2f"},
                ],
            },
        },
    }

    deltas_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Step-to-step score deltas.",
        "data": {"name": "source_data"},
        "facet": {
            "row": {"field": "metric", "type": "ordinal", "title": "Metric"},
            "column": {
                "field": "comparison_type",
                "type": "ordinal",
                "title": "Comparison Type",
            },
        },
        "spec": {
            "width": 260,
            "height": 160,
            "mark": "line",
            "encoding": {
                "x": {"field": "step", "type": "ordinal", "sort": "ascending"},
                "y": {"field": "delta", "type": "quantitative", "title": "Delta"},
                "color": {"field": "anchor", "type": "nominal"},
                "tooltip": [
                    {"field": "comparison_type", "type": "nominal"},
                    {"field": "metric", "type": "nominal"},
                    {"field": "anchor", "type": "nominal"},
                    {"field": "step", "type": "ordinal"},
                    {"field": "delta", "type": "quantitative", "format": ".2f"},
                ],
            },
        },
    }

    stability_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Mean score with stability bands.",
        "data": {"name": "source_data"},
        "facet": {
            "row": {"field": "metric", "type": "ordinal", "title": "Metric"},
            "column": {
                "field": "comparison_type",
                "type": "ordinal",
                "title": "Comparison Type",
            },
        },
        "spec": {
            "width": 260,
            "height": 160,
            "encoding": {
                "x": {"field": "step", "type": "ordinal", "sort": "ascending"}
            },
            "layer": [
                {
                    "mark": "errorband",
                    "encoding": {
                        "y": {
                            "field": "mean",
                            "type": "quantitative",
                            "title": "Mean Score",
                        },
                        "yError": {"field": "std", "type": "quantitative"},
                    },
                },
                {
                    "mark": "line",
                    "encoding": {"y": {"field": "mean", "type": "quantitative"}},
                },
            ],
        },
    }

    return {
        "trends": trends_spec,
        "availability": availability_spec,
        "anchor_vs_metric": anchor_vs_metric_spec,
        "deltas": deltas_spec,
        "stability": stability_spec,
    }


def generate_static_charts(
    df: pd.DataFrame, aggregates: Dict[str, Any], theme: str
) -> Dict[str, str]:
    """Generate static PNG charts using matplotlib."""
    if theme == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    charts = {}

    def save_b64() -> str:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close("all")  # Close all figures
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Score Trends
    g = sns.FacetGrid(
        data=df[df["valid"]],
        row="metric",
        col="comparison_type",
        hue="anchor",
        height=4,
        aspect=1.5,
        sharey=True,
    )
    g.map(sns.lineplot, "step", "score", marker="o")
    g.add_legend()
    g.set_axis_labels("Step", "Score")
    plt.ylim(0, 10)
    charts["trends"] = save_b64()

    # Availability
    availability_df = aggregates["availability"]

    def heatmap_pivot(data, **kwargs):
        pivot_data = data.pivot(index="metric", columns="step", values="count")
        sns.heatmap(pivot_data, **kwargs)

    g = sns.FacetGrid(data=availability_df, col="comparison_type", height=4, aspect=1.5)
    g.map_dataframe(heatmap_pivot, annot=True, cmap="viridis")
    g.set_axis_labels("Step", "Metric")
    charts["availability"] = save_b64()

    # Anchor vs Metric
    anchor_df = aggregates["anchor_vs_metric"]
    g = sns.catplot(
        data=anchor_df,
        x="metric",
        y="score",
        hue="anchor",
        col="comparison_type",
        kind="bar",
        height=4,
        aspect=1.5,
    )
    g.set_axis_labels("Metric", "Mean Score")
    charts["anchor_vs_metric"] = save_b64()

    # Deltas
    deltas_df = aggregates["deltas"]
    g = sns.FacetGrid(
        data=deltas_df,
        row="metric",
        col="comparison_type",
        hue="anchor",
        height=4,
        aspect=1.5,
        sharey=True,
    )
    g.map(sns.lineplot, "step", "delta")
    g.add_legend()
    g.set_axis_labels("Step", "Score Delta")
    charts["deltas"] = save_b64()

    # Stability
    stability_df = aggregates["stability"]
    g = sns.FacetGrid(
        data=stability_df,
        row="metric",
        col="comparison_type",
        height=4,
        aspect=1.5,
        sharey=False,
    )

    def plot_stability(data, **kwargs):
        plt.fill_between(
            data["step"],
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            alpha=0.3,
        )
        sns.lineplot(x=data["step"], y=data["mean"], **kwargs)

    g.map_dataframe(plot_stability)
    g.set_axis_labels("Step", "Mean Score")
    charts["stability"] = save_b64()

    return charts


def render_html_report(template: Template, output_path: Path, context: Dict[str, Any]):
    """Render the final HTML report."""
    html = template.render(context)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)


def main():
    """CLI entry point."""
    try:
        parser = argparse.ArgumentParser(
            description="Generate self-contained HTML reports from ratings.json files."
        )
        parser.add_argument(
            "path",
            type=Path,
            help="Path to a single ratings.json file or a root experiment directory.",
        )
        parser.add_argument(
            "--output-name",
            default="report.html",
            help="Name of the output HTML file (default: report.html).",
        )
        parser.add_argument(
            "--theme",
            choices=["light", "dark"],
            default="light",
            help="Visual theme for the report (default: light).",
        )
        parser.add_argument(
            "--static",
            action="store_true",
            help="Render static PNG charts instead of interactive Vega-Lite charts.",
        )
        args = parser.parse_args()

        ratings_files = discover_ratings_files(args.path)
        if not ratings_files:
            print(f"No `ratings.json` files found at: {args.path}", file=sys.stderr)
            sys.exit(1)

        for file_path in ratings_files:
            print(f"Processing {file_path}...")
            df, file_hash = load_and_normalize_data(file_path)

            if df is None or df.empty:
                print(
                    f"  Skipping {file_path} due to loading error or no valid data.",
                    file=sys.stderr,
                )
                continue

            aggregates = calculate_aggregates(df)

            context = {
                "title": "Evaluation Report",
                "item_id": df["item_id"].iloc[0] if not df.empty else "N/A",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "theme": args.theme,
                "summary_cards": aggregates.get("summary_cards", {}),
                "top_drops": aggregates.get("top_drops", []),
                "file_path": file_path.resolve(),
                "file_hash": file_hash,
                "static_charts": None,
            }

            if args.static:
                print("  Generating static charts...")
                context["static_charts"] = generate_static_charts(
                    df, aggregates, args.theme
                )
            else:
                print("  Generating Vega-Lite specs...")
                context["vega_js"] = VEGA_JS
                context["vega_lite_js"] = VEGA_LITE_JS
                context["vega_embed_js"] = VEGA_EMBED_JS
                context["vega_specs"] = generate_vega_specs(df, aggregates)
                context["vega_data"] = {
                    "trends": df.to_dict(orient="records"),
                    "availability": aggregates.get("availability").to_dict(
                        orient="records"
                    ),
                    "anchor_vs_metric": aggregates.get("anchor_vs_metric").to_dict(
                        orient="records"
                    ),
                    "deltas": aggregates.get("deltas").to_dict(orient="records"),
                    "stability": aggregates.get("stability").to_dict(orient="records"),
                }

            output_path = file_path.parent / args.output_name
            render_html_report(HTML_TEMPLATE, output_path, context)

            asset_path = file_path.parent / "report_assets.json.gz"
            with gzip.open(asset_path, "wt", encoding="utf-8") as f:
                df.to_json(f, orient="records", indent=2)

            print(f"  Successfully generated report: {output_path}")
            print(f"  Successfully saved assets: {asset_path}")

    except Exception:
        print("An unexpected error occurred:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
