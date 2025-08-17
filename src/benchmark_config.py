# -*- coding: utf-8 -*-
"""
Configuration loader for the benchmark.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class _LoopConfig:
    type: str
    num_iterations: int


@dataclass
class _PromptsConfig:
    caption: str = "Describe this image in a single, descriptive sentence."
    image: str = "Generate a detailed image based on this text description."


@dataclass
class _LoggingConfig:
    level: str = "INFO"
    save_config_snapshot: bool = True


@dataclass
class _EvaluationConfig:
    enabled: bool


@dataclass
class _ReportingConfig:
    charts: bool = False
    summary: bool = False


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark.

    Raises:
        FileNotFoundError: If the config file is not found.
        KeyError: If a required key is missing from the config.

    Returns:
        BenchmarkConfig: An instance of the benchmark configuration.
    """

    # REQUIRED fields (no default):
    experiment_name: str
    input_dir: str
    loop: _LoopConfig
    evaluation: _EvaluationConfig

    # OPTIONAL fields (with defaults):
    output_dir: str = "results/{{experiment_name}}"
    prompts: _PromptsConfig = field(default_factory=_PromptsConfig)
    logging: _LoggingConfig = field(default_factory=_LoggingConfig)
    reporting: _ReportingConfig = field(default_factory=_ReportingConfig)

    @staticmethod
    def from_yaml(path: str) -> "BenchmarkConfig":
        """
        Load the YAML at `path`, merge with defaults, then return a BenchmarkConfig.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for key in ["experiment_name", "input_dir", "loop", "evaluation"]:
            if key not in raw:
                raise KeyError(f"Missing required key: '{key}'")

        exp_name = raw["experiment_name"]
        inp_dir = raw["input_dir"]
        output_dir = BenchmarkConfig._format_output_dir(
            raw.get("output_dir", "results/{{experiment_name}}"), exp_name
        )

        loop_cfg = BenchmarkConfig._load_loop_config(raw["loop"])
        prompts_cfg = BenchmarkConfig._load_prompts_config(raw.get("prompts", {}))
        logging_cfg = BenchmarkConfig._load_logging_config(raw.get("logging", {}))
        eval_cfg = BenchmarkConfig._load_evaluation_config(raw["evaluation"])
        rep_cfg = BenchmarkConfig._load_reporting_config(raw.get("reporting", {}))

        return BenchmarkConfig(
            experiment_name=exp_name,
            input_dir=inp_dir,
            loop=loop_cfg,
            output_dir=output_dir,
            prompts=prompts_cfg,
            logging=logging_cfg,
            evaluation=eval_cfg,
            reporting=rep_cfg,
        )

    @staticmethod
    def _load_loop_config(loop_dict: Dict[str, Any]) -> _LoopConfig:
        num = loop_dict.get("num_iterations")
        if not isinstance(num, int) or num <= 0:
            raise ValueError("num_iterations must be an integer greater than zero")
        return _LoopConfig(type=loop_dict["type"], num_iterations=num)

    @staticmethod
    def _load_prompts_config(prompts_dict: Dict[str, Any]) -> _PromptsConfig:
        return _PromptsConfig(
            caption=prompts_dict.get(
                "caption", "Describe this image in a single, descriptive sentence."
            ),
            image=prompts_dict.get(
                "image", "Generate a detailed image based on this text description."
            ),
        )

    @staticmethod
    def _load_logging_config(log_dict: Dict[str, Any]) -> _LoggingConfig:
        return _LoggingConfig(
            level=log_dict.get("level", "INFO"),
            save_config_snapshot=bool(log_dict.get("save_config_snapshot", True)),
        )

    @staticmethod
    def _load_evaluation_config(eval_dict: Dict[str, Any]) -> _EvaluationConfig:
        if "enabled" not in eval_dict:
            raise KeyError("evaluation.enabled is required")
        return _EvaluationConfig(enabled=bool(eval_dict["enabled"]))

    @staticmethod
    def _load_reporting_config(rep_dict: Dict[str, Any]) -> _ReportingConfig:
        return _ReportingConfig(
            charts=bool(rep_dict.get("charts", True)),
            summary=bool(rep_dict.get("summary", True)),
        )

    @staticmethod
    def _format_output_dir(template: str, exp_name: str) -> str:
        single_brace = template.replace("{{", "{").replace("}}", "}")
        try:
            return single_brace.format(experiment_name=exp_name)
        except KeyError as e:
            # Unknown placeholder used in template
            raise ValueError(
                "Invalid output_dir template. Only '{experiment_name}' is supported. "
                f"Got unknown key: {e} in template: {template!r}"
            ) from e
        except ValueError as e:
            # Malformed braces or format string
            raise ValueError(
                f"Malformed output_dir template: {template!r}. Use 'results/{{experiment_name}}' or similar."
            ) from e
