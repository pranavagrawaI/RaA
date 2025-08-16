"""Generate a qualitative summary of evaluation metrics using Gemini.

This script reads JSON rating files from an evaluation folder,
prepares a structured prompt, and uses a Gemini model to produce a
qualitative narrative. The resulting text is saved to
``qualitative_summary.txt`` in the same folder and the first lines are
printed to stdout.

Usage:
    python reporting_summary.py --eval-folder exp_025/eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from google import genai
from google.genai import types


CRITERIA = [
    "content_correspondence",
    "compositional_alignment",
    "fidelity_completeness",
    "stylistic_congruence",
    "overall_semantic_intent",
]

SYSTEM_PROMPT = (
    "You are an expert evaluation analyst. Based on the numeric synopsis "
    "and stepwise reasons, craft a cohesive narrative describing what "
    "changed, why it changed, why it matters, and what to try next. Avoid "
    "lists or JSON. Do not repeat raw numbers beyond those provided in the "
    "synopsis."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise evaluation metrics with Gemini"
    )
    parser.add_argument(
        "--eval-folder",
        type=str,
        default="exp_025/eval",
        help="Path to the eval folder containing JSON files",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash", help="Gemini model name"
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Thinking budget for Gemini (tokens)",
    )
    return parser.parse_args()


def _load_records(eval_dir: Path) -> List[Dict]:
    records: List[Dict] = []
    if not eval_dir.exists():
        print(f"[WARN] Eval folder '{eval_dir}' not found", file=sys.stderr)
        return records
    for path in eval_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
            else:
                print(
                    f"[WARN] Unexpected JSON structure in {path.name}; skipping",
                    file=sys.stderr,
                )
        except FileNotFoundError:
            print(f"[WARN] Missing file {path}", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON in {path}", file=sys.stderr)
    return records


def _avg_scores(records: Iterable[Dict]) -> Dict[str, Dict[int, float]]:
    scores: Dict[str, Dict[int, List[float]]] = {
        c: defaultdict(list) for c in CRITERIA
    }
    for rec in records:
        step = rec.get("step")
        if step is None:
            continue
        for c in CRITERIA:
            if c in rec and isinstance(rec[c], dict):
                val = rec[c].get("score")
                if isinstance(val, (int, float)):
                    scores[c][int(step)].append(val)
    return {c: {s: mean(vs) for s, vs in step_map.items()} for c, step_map in scores.items()}


def _reasons(records: Iterable[Dict]) -> Dict[str, Dict[int, str]]:
    reasons: Dict[str, Dict[int, str]] = {c: {} for c in CRITERIA}
    for rec in records:
        step = rec.get("step")
        if step is None:
            continue
        for c in CRITERIA:
            if c in rec and isinstance(rec[c], dict):
                reason = rec[c].get("reason")
                if isinstance(reason, str) and step not in reasons[c]:
                    reasons[c][int(step)] = reason.strip()
    return reasons


def _classify_pattern(values: List[float]) -> str:
    if not values:
        return "stable"
    if max(values) - min(values) < 0.2:
        return "stable"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs):
        return "mono"
    sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
    if sign_changes >= 2:
        return "osc"
    return "step"


def _magnitude(delta: float) -> str:
    delta = abs(delta)
    if delta < 0.5:
        return "small"
    if delta < 1.5:
        return "mod"
    return "large"


def build_prompt(eval_dir: Path, records: List[Dict]) -> str:
    run_id = eval_dir.parent.name
    steps = sorted({rec.get("step") for rec in records if rec.get("step") is not None})
    comparisons = sorted({rec.get("comparison_type") for rec in records if rec.get("comparison_type")})
    anchors = sorted({rec.get("anchor") for rec in records if rec.get("anchor")})

    scores = _avg_scores(records)
    reasons = _reasons(records)

    header_lines = [
        f"Run id: {run_id}",
        f"Steps: {', '.join(map(str, steps)) if steps else 'none'}",
        f"Comparison types: {', '.join(comparisons) if comparisons else 'none'}",
    ]
    header_lines.append(
        f"Anchors: {', '.join(anchors) if anchors else 'none'}"
    )

    synopsis_lines: List[str] = []
    for crit, step_map in scores.items():
        if not step_map:
            continue
        ordered = [v for _, v in sorted(step_map.items())]
        start, end = ordered[0], ordered[-1]
        synopsis_lines.append(
            f"* {crit.replace('_', ' ')}: {start:.1f} → {end:.1f} ({_magnitude(end - start)}), pattern: {_classify_pattern(ordered)}"
        )

    reason_lines: List[str] = []
    for crit, step_map in reasons.items():
        if not step_map:
            continue
        reason_lines.append(f"{crit.replace('_', ' ')}:")
        for step in sorted(step_map):
            reason_lines.append(f"  step {step}: {step_map[step]}")

    prompt_sections = [
        "\n".join(header_lines),
        "Numeric synopsis:\n" + "\n".join(synopsis_lines) if synopsis_lines else "Numeric synopsis: none",
        "Reasons:\n" + "\n".join(reason_lines) if reason_lines else "Reasons: none",
    ]
    return "\n\n".join(prompt_sections)


def generate_summary(prompt: str, model: str, budget: int) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_budget=budget),
            ),
        )
    except Exception as exc:  # broad to capture network/auth issues
        print(f"[ERROR] Gemini request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    return getattr(response, "text", "").strip()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_folder)
    records = _load_records(eval_dir)

    if not records:
        summary = "No evaluation data available."  # Do not call Gemini
    else:
        prompt = build_prompt(eval_dir, records)
        summary = generate_summary(prompt, args.model, args.thinking_budget)

    out_path = eval_dir / "qualitative_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n".join(summary.splitlines()[:15]))


if __name__ == "__main__":
    main()
