# -*- coding: utf-8 -*-
"""main module for RaA

Orchestrates generation, evaluation, and reporting from a single entry point.
Supports skip/resume flags:
    -e / --eval   : run only evaluation on existing outputs
    -r / --report : run only reporting (charts + summaries) on existing evals
If no flags are provided, the full pipeline runs: generation -> evaluation -> reporting.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai

from benchmark_config import BenchmarkConfig
from evaluation_engine import EvaluationEngine
from loop_controller import LoopController
from graph_creator import GraphCreator
from reporting_summary import SummaryGenerator

load_dotenv()


def main():
    """Main entry point for the script."""
    args = parse_args()

    try:
        config = BenchmarkConfig.from_yaml(args.config)
    except (FileNotFoundError, yaml.YAMLError, AttributeError) as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(config.output_dir, exist_ok=True)

    # Determine run mode
    run_eval_only = bool(args.eval)
    run_report_only = bool(args.report)

    # Sanity: mutually exclusive flags should already be enforced by parser

    # Helper: save a snapshot of the config used for this run
    def _save_config_snapshot():
        if config.logging.save_config_snapshot and not (
            run_eval_only or run_report_only
        ):
            raw_dict = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
            snapshot_path = os.path.join(config.output_dir, "config_snapshot.yaml")
            with open(snapshot_path, "w", encoding="utf-8") as out_f:
                yaml.safe_dump(raw_dict, out_f)

    # Helper: evaluation step
    def _run_evaluation(always_run: bool = False):
        if not always_run and not getattr(config.evaluation, "enabled", False):
            print("[INFO] Evaluation disabled by config. Skipping.")
            return False
        api_key = os.getenv("GOOGLE_API_KEY")
        engine = EvaluationEngine(
            config.output_dir,
            config=config,
            client=genai.Client(api_key=api_key) if api_key else None,
        )
        engine.run()
        print("[INFO] Evaluation complete.")
        return True

    # Helper: reporting step (charts + qualitative summary)
    def _run_reporting(force: bool = False):
        reporting_cfg = getattr(config, "reporting", None)
        # config.reporting is a typed object (e.g. _ReportingConfig). Use getattr with defaults.
        if reporting_cfg is None:
            want_charts = False
            want_summary = False
        else:
            want_charts = bool(getattr(reporting_cfg, "charts", True))
            want_summary = bool(getattr(reporting_cfg, "summary", True))

        if not force and not (want_charts or want_summary):
            print("[INFO] Reporting disabled by config. Skipping.")
            return False

        exp_root = Path(config.output_dir)
        ran_any = False

        # Charts
        if force or want_charts:
            graph_creator = GraphCreator()
            all_charts = graph_creator.generate_charts_for_experiment(exp_root)
            total = len(all_charts)
            print(f"[INFO] Charts generated: {total}")
            ran_any = ran_any or total > 0
        else:
            print("[INFO] Charts disabled by config.")

        # Qualitative summaries
        if force or want_summary:
            api_key = os.getenv("GOOGLE_API_KEY")
            summary_generator = SummaryGenerator(
                client=genai.Client(api_key=api_key) if api_key else None
            )
            prompts_root = Path(__file__).parent.parent / "prompts"
            system_file = prompts_root / "system_instruction_report.txt"
            if not system_file.exists():
                print(f"[WARN] System instruction file not found: {system_file}")
                success = 0
            else:
                successful, _ = summary_generator.generate_summaries_for_experiment(
                    exp_root, system_file
                )
                success = successful
            print(f"[INFO] Summaries generated for {success} item(s)")
            ran_any = True
        else:
            print("[INFO] Qualitative summary disabled by config.")

        return ran_any

    # Execution orchestration
    if run_eval_only:
        print(f"[INFO] Running EVALUATION ONLY for `{config.output_dir}`.")
        _run_evaluation(always_run=True)
        return

    if run_report_only:
        print(f"[INFO] Running REPORTING ONLY for `{config.output_dir}`.")
        _run_reporting(force=True)
        return

    # Full run: generation -> evaluation -> reporting
    _save_config_snapshot()

    controller = LoopController(config)
    # Announce loop generation start for better console visibility
    loop_type = str(getattr(config.loop, "type", "") or "").upper()
    iterations_raw = getattr(config.loop, "num_iterations", None)
    try:
        iterations = int(iterations_raw) if iterations_raw is not None else 0
    except (TypeError, ValueError):
        iterations = iterations_raw if iterations_raw is not None else "?"
    print(
        f"[INFO] Loop generation starting: type={loop_type}, iterations={iterations}"
    )
    controller.run()
    print(f"[INFO] Loop generation complete. Outputs are in `{config.output_dir}`.")

    did_eval = _run_evaluation(always_run=False)
    if did_eval:
        _run_reporting(force=False)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RaA Benchmark Runner")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run only EVALUATION on existing outputs (skip generation and reporting).",
    )
    group.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Run only REPORTING (charts + qualitative summaries) using existing eval results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
