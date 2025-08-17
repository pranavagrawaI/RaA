# -*- coding: utf-8 -*-
"""Simple script to generate qualitative summaries from JSON evaluation data using Gemini."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# For each entry in experiment folder take all rating_*.json and pass it to gemini.


def _collect_eval_files(eval_dir: Path) -> List[Path]:
    """Collect the four expected evaluation JSON files in a consistent order."""
    expected_files = [
        "ratings_image-image.json",
        "ratings_image-text.json",
        "ratings_text-image.json",
        "ratings_text-text.json",
    ]

    found_files = []
    for filename in expected_files:
        file_path = eval_dir / filename
        if file_path.exists():
            found_files.append(file_path)

    return found_files


def _load_json_file(file_path: Path) -> dict:
    """Load a JSON file and return its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_individual_eval_files(eval_dir: Path) -> List[dict]:
    """Load all JSON files in the directory and return as individual dictionaries."""
    if not eval_dir.exists():
        raise FileNotFoundError(f"Directory not found: {eval_dir}")

    json_files = _collect_eval_files(eval_dir)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {eval_dir}")

    eval_data = []
    for json_file in json_files:
        content = _load_json_file(json_file)
        eval_data.append({"filename": json_file.name, "data": content})

    print(f"Found {len(json_files)} JSON files")
    return eval_data


def generate_summary(
    eval_data: List[dict], system_instruction_file: Path, item_id: str
) -> str:
    """Generate summary using Gemini API with individual evaluation files."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set"

    # Load system instruction
    with open(system_instruction_file, "r", encoding="utf-8") as f:
        system_instruction = f.read().strip()

    client = genai.Client(api_key=api_key)

    # Construct prompt with item list and individual eval files
    prompt_parts = [f"Item ID: {item_id}", "", "Evaluation Data:", ""]

    for eval_file in eval_data:
        prompt_parts.append(f"=== {eval_file['filename']} ===")
        prompt_parts.append(json.dumps(eval_file["data"], indent=2))
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    print("--- PROMPT BEGIN ---")
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    print("--- PROMPT END ---")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        return getattr(response, "text", "No response text").strip()
    except (ValueError, TypeError, ConnectionError, RuntimeError) as e:
        return f"Error generating summary: {e}"


def _discover_eval_dirs(root: Path) -> List[Path]:
    """Discover all eval directories under the given root path.

    Args:
        root: Path to search for eval directories. Can be:
            - An eval folder (…/<item_id>/eval)
            - An item folder containing an eval subfolder (…/<item_id>)
            - An experiment folder containing multiple items (…/results/exp_xxx)

    Returns:
        List of paths to eval directories found.
    """
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


def _extract_item_id(eval_dir: Path) -> str:
    """Extract item ID from an eval directory path."""
    parent = eval_dir.parent.name
    return parent or "item"


def generate_summary_for_eval(eval_dir: Path, system_instruction_file: Path) -> bool:
    """Generate summary for a single eval directory.

    Args:
        eval_dir: Path to the eval directory containing ratings_*.json files
        system_instruction_file: Path to the system instruction file

    Returns:
        True if summary was generated successfully, False otherwise
    """
    try:
        # Extract item ID first
        item_id = _extract_item_id(eval_dir)

        # Load individual JSON files
        eval_data = load_individual_eval_files(eval_dir)

        # Generate summary with individual files
        summary = generate_summary(eval_data, system_instruction_file, item_id)

        # Write output
        output_file = eval_dir / "qualitative_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"Summary written to: {output_file} (item: {item_id})")

        return True

    except FileNotFoundError as e:
        item_id = _extract_item_id(eval_dir)
        print(
            f"Warning: No ratings files found for item '{item_id}' in {eval_dir}: {e}"
        )
        return False
    except (OSError, IOError, RuntimeError) as e:
        item_id = _extract_item_id(eval_dir)
        print(f"Error processing item '{item_id}' in {eval_dir}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate summaries from JSON evaluation data for all items in an experiment"
    )
    parser.add_argument(
        "experiment-dir",
        help="Path to experiment directory containing item subdirectories with eval folders",
    )
    args = parser.parse_args()

    experiment_dir = Path(getattr(args, "experiment-dir"))
    system_file = (
        Path(__file__).parent.parent
        / "prompts"
        / "system_instruction_report.txt"  # Fixed filename
    )

    if not experiment_dir.exists():
        print(
            f"Error: Experiment directory not found: {experiment_dir}", file=sys.stderr
        )
        return 1

    if not system_file.exists():
        print(
            f"Error: System instruction file not found: {system_file}", file=sys.stderr
        )
        return 1

    # Discover all eval directories in the experiment
    eval_dirs = _discover_eval_dirs(experiment_dir)

    if not eval_dirs:
        print(f"No eval directories found under: {experiment_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(eval_dirs)} eval directories to process")

    successful = 0
    failed = 0

    for eval_dir in eval_dirs:
        if generate_summary_for_eval(eval_dir, system_file):
            successful += 1
        else:
            failed += 1

    print("\nSummary generation complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(eval_dirs)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
