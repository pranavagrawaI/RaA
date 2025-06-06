"""Load the benchmark configuration from a YAML file.

Raises:
    FileNotFoundError: If the config file is not found.
    ValueError: If the config file is not valid YAML.
    KeyError: If required keys are missing from the config.
    ValueError: If there is an error rendering templates in the config.
    KeyError: If required keys are missing from nested sections.
    KeyError: If there are unknown keys in nested sections.

Returns:
    _type_: _description_
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

import yaml


@dataclass
class _LoopConfig:
    type: str
    num_iterations: int
    stateless: bool


@dataclass
class _ModelSpec:
    name: str
    params: Dict[str, Any]


@dataclass
class _ModelsConfig:
    caption_model: _ModelSpec
    image_model: _ModelSpec


@dataclass
class _PromptsConfig:
    naive: str
    raa_aware: str


@dataclass
class _LoggingConfig:
    level: str
    save_config_snapshot: bool


@dataclass
class _MetadataConfig:
    random_seed: int


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark experiment."""

    experiment_name: str
    input_dir: str
    output_dir: str
    loop: _LoopConfig
    models: _ModelsConfig
    prompts: _PromptsConfig
    logging: _LoggingConfig
    metadata: _MetadataConfig
    # evaluation and reporting can remain as dicts for now
    evaluation: Dict[str, Any]
    reporting: Dict[str, Any]

    @staticmethod
    def from_yaml(path: str) -> "BenchmarkConfig":
        """
        Load the YAML file at `path`, validate required fields, render templates,
        and return a populated BenchmarkConfig instance.
        """
        # 1) Read raw YAML
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse YAML: {e}") from e

        # 2) Basic validation of top-level keys
        required_keys = [
            "experiment_name",
            "input_dir",
            "output_dir",
            "loop",
            "models",
            "prompts",
            "logging",
            "metadata",
        ]
        missing = [k for k in required_keys if k not in raw]
        if missing:
            raise KeyError(f"Missing required keys in config: {missing}")

        # 3) Render output_dir template: replace {{experiment_name}} with actual value
        experiment_name = raw["experiment_name"]
        # e.g., "results/{{experiment_name}}".format(experiment_name=…)
        try:
            single_brace = raw["output_dir"].replace("{{", "{").replace("}}", "}")
            rendered_output_dir = single_brace.format(experiment_name=experiment_name)
        except Exception as e:
            raise ValueError(f"Failed to render output_dir template: {e}") from e
        # 4) Parse nested sections into dataclasses
        # 4a) LoopConfig
        loop_dict = raw["loop"]
        for lk in ["type", "num_iterations", "stateless"]:
            if lk not in loop_dict:
                raise KeyError(f"Missing '{lk}' in 'loop' section")
        loop_cfg = _LoopConfig(
            type=loop_dict["type"],
            num_iterations=int(loop_dict["num_iterations"]),
            stateless=bool(loop_dict["stateless"]),
        )

        # 4b) ModelsConfig
        models_dict = raw["models"]
        for model_name in ["caption_model", "image_model"]:
            if model_name not in models_dict:
                raise KeyError(f"Missing '{model_name}' in 'models' section")

        cap = models_dict["caption_model"]
        img = models_dict["image_model"]
        # Validate presence of `name` and `params`
        for mkey in ["name", "params"]:
            if mkey not in cap:
                raise KeyError(f"Missing '{mkey}' in 'models.caption_model'")
            if mkey not in img:
                raise KeyError(f"Missing '{mkey}' in 'models.image_model'")

        models_cfg = _ModelsConfig(
            caption_model=_ModelSpec(name=cap["name"], params=cap["params"] or {}),
            image_model=_ModelSpec(name=img["name"], params=img["params"] or {}),
        )

        # 4c) PromptsConfig
        prompts_dict = raw["prompts"]
        if "naive" not in prompts_dict or "raa_aware" not in prompts_dict:
            raise KeyError("Both 'naive' and 'raa_aware' keys required in 'prompts'")
        prompts_cfg = _PromptsConfig(
            naive=prompts_dict["naive"], raa_aware=prompts_dict["raa_aware"]
        )

        # 4d) LoggingConfig
        log_dict = raw["logging"]
        if "level" not in log_dict or "save_config_snapshot" not in log_dict:
            raise KeyError("Missing keys under 'logging'")
        logging_cfg = _LoggingConfig(
            level=log_dict["level"],
            save_config_snapshot=bool(log_dict["save_config_snapshot"]),
        )

        # 4e) MetadataConfig
        meta_dict = raw["metadata"]
        if "random_seed" not in meta_dict:
            raise KeyError("Missing 'random_seed' under 'metadata'")
        metadata_cfg = _MetadataConfig(random_seed=int(meta_dict["random_seed"]))

        # 5) evaluation and reporting can stay as raw dicts (empty or user-defined)
        eval_cfg = raw.get("evaluation", {})
        rep_cfg = raw.get("reporting", {})

        # 6) Instantiate the top-level BenchmarkConfig
        return BenchmarkConfig(
            experiment_name=experiment_name,
            input_dir=raw["input_dir"],
            output_dir=rendered_output_dir,
            loop=loop_cfg,
            models=models_cfg,
            prompts=prompts_cfg,
            logging=logging_cfg,
            metadata=metadata_cfg,
            evaluation=eval_cfg,
            reporting=rep_cfg,
        )
