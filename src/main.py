import argparse
import os
import sys

import yaml

from loop_controller import LoopController
from benchmark_config import BenchmarkConfig
from evaluation_engine import EvaluationEngine


def parse_args():
    parser = argparse.ArgumentParser(description="RaA v0.1 Dry Loop Runner")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load & validate config
    try:
        config = BenchmarkConfig.from_yaml(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # 3) Optionally save a snapshot of the loaded YAML
    if config.logging.save_config_snapshot:
        raw_dict = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        snapshot_path = os.path.join(config.output_dir, "config_snapshot.yaml")
        with open(snapshot_path, "w", encoding="utf-8") as out_f:
            yaml.safe_dump(raw_dict, out_f)

    # 4) Instantiate & run the LoopController
    controller = LoopController(config)
    controller.run()

    if config.evaluation.enabled:
        engine = EvaluationEngine(
            config.output_dir, mode=config.evaluation.mode
        )
        engine.run()
        print("[INFO] Evaluation complete.")

    print(f"[INFO] Dry loop complete. Outputs are in `{config.output_dir}`.")


if __name__ == "__main__":
    main()
