# -*- coding: utf-8 -*-
"""main module for RaA"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv
from google import genai

from benchmark_config import BenchmarkConfig
from evaluation_engine import EvaluationEngine
from loop_controller import LoopController

load_dotenv()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RaA Benchmark Runner")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Only run evaluation on existing outputs, skip generation.",
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    try:
        config = BenchmarkConfig.from_yaml(args.config)
    except (FileNotFoundError, yaml.YAMLError, AttributeError) as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(config.output_dir, exist_ok=True)

    if not args.eval:
        if config.logging.save_config_snapshot:
            raw_dict = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
            snapshot_path = os.path.join(config.output_dir, "config_snapshot.yaml")
            with open(snapshot_path, "w", encoding="utf-8") as out_f:
                yaml.safe_dump(raw_dict, out_f)

        controller = LoopController(config)
        controller.run()

        print(f"[INFO] Loop generation complete. Outputs are in `{config.output_dir}`.")
    else:
        print(
            f"[INFO] Skipping generation. Evaluating outputs in `{config.output_dir}`."
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if config.evaluation.enabled:
        engine = EvaluationEngine(
            config.output_dir,
            config=config,
            client=genai.Client(api_key=api_key),
        )
        engine.run()
        print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
