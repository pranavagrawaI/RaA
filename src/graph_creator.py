# -*- coding: utf-8 -*-
"""
Graph creation utilities for plotting evaluation scores over generations.

This module provides the GraphCreator class which handles the generation
of evaluation score visualizations from ratings JSON files. Charts show
scores across generations, grouped by comparison type and anchor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


CRITERIA = [
    "content_correspondence",
    "compositional_alignment",
    "fidelity_completeness",
    "stylistic_congruence",
    "overall_semantic_intent",
]

ComparisonType = Literal["image-image", "text-text", "image-text", "text-image"]
AnchorType = Literal["original", "previous", "same-step"]


@dataclass(frozen=True)
class Key:
    """Represents a unique key for a specific evaluation scenario."""

    comparison_type: ComparisonType
    anchor: AnchorType
    direction: Optional[str] = None


class GraphCreator:
    """Class responsible for creating evaluation charts from rating data."""

    def __init__(self):
        """Initialize the GraphCreator with default settings."""
        self.criteria = CRITERIA
        self.colors = {
            "content_correspondence": "#1f77b4",
            "compositional_alignment": "#ff7f0e",
            "fidelity_completeness": "#2ca02c",
            "stylistic_congruence": "#d62728",
            "overall_semantic_intent": "#9467bd",
        }

    def generate_charts_for_eval(self, eval_dir: Path) -> List[Path]:
        """Generate evaluation charts for the given directory.

        Args:
            eval_dir (Path): The directory containing evaluation data.

        Returns:
            List[Path]: A list of paths to the generated chart images.
        """
        eval_dir = eval_dir.resolve()
        charts: List[Path] = []
        records = self._load_records(eval_dir)
        if not records:
            return charts
        item_id = self._extract_item_id(eval_dir, records)

        # Auto-detect loop type from available data
        has_imgimg_orig = any(
            r.get("comparison_type") == "image-image" and r.get("anchor") == "original"
            for r in records
        )
        has_txttxt_orig = any(
            r.get("comparison_type") == "text-text" and r.get("anchor") == "original"
            for r in records
        )
        if has_imgimg_orig and not has_txttxt_orig:
            loop_type = "I-T-I"
        elif has_txttxt_orig and not has_imgimg_orig:
            loop_type = "T-I-T"
        else:
            loop_type = "UNKNOWN"

        # Define expected keys based on loop type
        wanted_keys = self._get_wanted_keys(loop_type)

        missing = []
        for key in wanted_keys:
            series = list(self._iter_series(records, key))
            if not series:
                missing.append(key)
            out = self._plot_group(item_id, eval_dir, key, series)
            if out:
                charts.append(out)

        if missing:
            print(f"[reporting] Loop type detected: {loop_type}")
            print("[reporting] Missing expected data for the following groupings:")
            for key in missing:
                print(f"  - {key.comparison_type} | {key.anchor} | {key.direction}")

        if charts:
            index = {
                "item_id": item_id,
                "charts": [str(p.name) for p in charts],
            }
            try:
                with (eval_dir / "charts_index.json").open("w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2)
            except (OSError, IOError) as e:
                print(f"[reporting] Failed to write charts_index.json: {e}")

        return charts

    def generate_charts_for_experiment(self, root: Path) -> List[Path]:
        """Generate charts for all evaluation directories found under the given root path.

        Args:
            root (Path): The root path to search for evaluation directories.

        Returns:
            List[Path]: A list of paths to all generated chart images.
        """
        eval_dirs = self.discover_eval_dirs(root)
        if not eval_dirs:
            return []

        all_charts: List[Path] = []
        for eval_dir in eval_dirs:
            charts = self.generate_charts_for_eval(eval_dir)
            all_charts.extend(charts)
            print(f"Generated {len(charts)} charts -> {eval_dir}")

        return all_charts

    def _get_wanted_keys(self, loop_type: str) -> List[Key]:
        """Get the expected evaluation keys based on loop type."""
        if loop_type == "I-T-I":
            # Image -> Text -> Image: starts with image, no text-image with original
            return [
                Key("image-image", "original"),
                Key("image-image", "previous"),
                Key("text-text", "previous"),
                Key("image-text", "same-step"),
                Key("image-text", "original"),
                Key("image-text", "previous"),
                Key("text-image", "previous"),  # Only previous, not original
            ]
        elif loop_type == "T-I-T":
            # Text -> Image -> Text: starts with text, no image-image with original
            return [
                Key("image-image", "previous"),  # Only previous, not original
                Key("text-text", "original"),
                Key("text-text", "previous"),
                Key("image-text", "same-step"),
                Key("image-text", "previous"),  # Only previous, not original
                Key("text-image", "original"),
                Key("text-image", "previous"),
            ]
        else:
            # Unknown loop type: try all combinations
            return [
                Key("image-image", "original"),
                Key("image-image", "previous"),
                Key("text-text", "original"),
                Key("text-text", "previous"),
                Key("image-text", "same-step"),
                Key("image-text", "original"),
                Key("image-text", "previous"),
                Key("text-image", "original"),
                Key("text-image", "previous"),
            ]

    @staticmethod
    def _sanitize_filename(s: str) -> str:
        """Sanitize a string to be used as a filename."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "-", s)

    @staticmethod
    def _collect_eval_files(eval_dir: Path) -> List[Path]:
        """Collect all ratings JSON files from the evaluation directory."""
        return sorted(eval_dir.glob("ratings_*.json"))

    def _load_records(self, eval_dir: Path) -> List[Dict[str, Any]]:
        """Load all evaluation records from the directory."""
        records: List[Dict[str, Any]] = []
        for fp in self._collect_eval_files(eval_dir):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[reporting] Failed to decode JSON in {fp}: {e}")
                continue
            except OSError as e:
                print(f"[reporting] Failed to open {fp}: {e}")
                continue
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        records.append(rec)
            elif isinstance(data, dict):
                records.append(data)
        return records

    @staticmethod
    def _extract_item_id(eval_dir: Path, records: List[Dict[str, Any]]) -> str:
        """Extract the item ID from records or directory name."""
        for rec in records:
            if isinstance(rec, dict) and rec.get("item_id"):
                return str(rec["item_id"])
        parent = eval_dir.parent.name
        return parent or "item"

    def _iter_series(
        self, records: List[Dict[str, Any]], wanted: Key
    ) -> Iterable[Tuple[int, Dict[str, Optional[float]]]]:
        """Yield (step, scores_by_criteria) for a given grouping key.

        Only returns steps where at least one criterion has a non-null score.
        """
        for rec in records:
            ctype = rec.get("comparison_type")
            anchor = rec.get("anchor")
            if ctype != wanted.comparison_type or anchor != wanted.anchor:
                continue

            step_val = rec.get("step")
            if step_val is None:
                continue
            try:
                step = int(step_val)
            except (ValueError, TypeError) as e:
                print(
                    f"[reporting] Failed to parse step value '{step_val}' in record: {e}"
                )
                continue  # Skip this record if step cannot be parsed

            scores: Dict[str, Optional[float]] = {}
            has_any = False
            for crit in self.criteria:
                val = rec.get(crit)
                score: Optional[float] = None
                if isinstance(val, dict) and "score" in val:
                    try:
                        score = float(val["score"])
                    except (ValueError, TypeError):
                        print("[reporting] failed to assign score.")
                        score = None
                # Filter out placeholders like -1.0 if present
                if score is not None and score < 0:
                    score = None
                if score is not None:
                    has_any = True
                scores[crit] = score
            if has_any:
                yield step, scores

    def _plot_group(
        self,
        item_id: str,
        eval_dir: Path,
        key: Key,
        series: List[Tuple[int, Dict[str, Optional[float]]]],
    ) -> Optional[Path]:
        """Plot a group of evaluation data and save as PNG."""
        if not series:
            return None
        series.sort(key=lambda x: x[0])
        steps = [s for s, _ in series]
        # Prepare y values per criterion, keeping None as gaps
        ys: Dict[str, List[Optional[float]]] = {
            crit: [vals.get(crit) for _, vals in series] for crit in self.criteria
        }

        plt.figure(figsize=(9.5, 5.5), dpi=140)

        for crit in self.criteria:
            y = ys[crit]
            # Plot with gaps for None
            x_non_null = [x for x, v in zip(steps, y) if v is not None]
            y_non_null = [float(v) for v in y if v is not None]
            if not x_non_null:
                continue
            plt.plot(
                x_non_null,
                y_non_null,
                marker="o",
                linewidth=1.8,
                label=crit.replace("_", " ").title(),
                color=self.colors.get(crit),
                linestyle="--",
            )

        title_parts = [item_id, key.comparison_type, key.anchor]
        if key.direction:
            title_parts.append(key.direction)
        plt.title(" | ".join(title_parts))
        plt.xlabel("Generation (step)")
        plt.ylabel("Score")

        # Set y-axis limits based on data
        all_vals: List[float] = []
        for crit in self.criteria:
            for v in ys[crit]:
                if v is not None:
                    all_vals.append(float(v))
        if all_vals:
            ymax = max(all_vals)
            if ymax <= 1.0:
                plt.ylim(0, 1.0)
            elif ymax <= 10.0:
                plt.ylim(0, 11.0)
            else:
                plt.ylim(0, ymax * 1.05)

        if steps:
            min_step = min(steps)
            max_step = max(steps)
            plt.xticks(range(min_step, max_step + 1))

        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        # File name
        base = f"chart_{key.comparison_type}_{key.anchor}"
        if key.direction:
            base += f"_{key.direction}"
        fname = self._sanitize_filename(base) + ".png"
        out_path = eval_dir / fname
        plt.savefig(out_path)
        plt.close()
        return out_path

    @staticmethod
    def discover_eval_dirs(root: Path) -> List[Path]:
        """Discover evaluation directories under the given root path."""
        root = root.resolve()
        eval_dirs: List[Path] = []

        if root.name == "eval" and root.is_dir():
            return [root]

        # If this is an item folder with eval subfolder
        candidate = root / "eval"
        if candidate.is_dir():
            eval_dirs.append(candidate)

        # If this is an experiment folder containing multiple items
        for child in root.iterdir() if root.is_dir() else []:
            c_eval = child / "eval"
            if c_eval.is_dir():
                eval_dirs.append(c_eval)

        # As a last resort, deep scan (one level deeper) for any eval dirs
        if not eval_dirs and root.is_dir():
            for p in root.rglob("eval"):
                if p.is_dir():
                    eval_dirs.append(p)

        # Deduplicate
        seen = set()
        uniq: List[Path] = []
        for d in eval_dirs:
            if d not in seen:
                uniq.append(d)
                seen.add(d)
        return uniq
