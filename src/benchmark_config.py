import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class LoopConfig:
    type: str
    num_iterations: int


@dataclass
class ModelSpec:
    name: str = "dummy-captioner"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    caption_model: ModelSpec = field(default_factory=ModelSpec)
    image_model: ModelSpec = field(
        default_factory=lambda: ModelSpec(name="dummy-imagegen")
    )


@dataclass
class PromptsConfig:
    naive: str = "prompts/naive.json"
    raa_aware: str = "prompts/raa_aware.json"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    save_config_snapshot: bool = True


@dataclass
class MetadataConfig:
    random_seed: Optional[int] = None


@dataclass
class BenchmarkConfig:
    # REQUIRED fields (no default):
    experiment_name: str
    input_dir: str
    loop: LoopConfig

    # OPTIONAL fields (with defaults):
    output_dir: str = "results/{{experiment_name}}"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    reporting: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "BenchmarkConfig":
        """
        Load the YAML at `path`, merge with defaults, then return a BenchmarkConfig.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for key in ["experiment_name", "input_dir", "loop"]:
            if key not in raw:
                raise KeyError(f"Missing required key: '{key}'")

        exp_name = raw["experiment_name"]
        inp_dir = raw["input_dir"]

        out_dir_template = raw.get("output_dir", "results/{{experiment_name}}")
        out_dir_single = out_dir_template.replace("{{", "{").replace("}}", "}")
        output_dir = out_dir_single.format(experiment_name=exp_name)

        loop_dict = raw["loop"]
        loop_cfg = LoopConfig(
            type=loop_dict["type"],
            num_iterations=int(loop_dict["num_iterations"]),
        )

        models_dict = raw.get("models", {})
        cap_dict = models_dict.get("caption_model", {})
        img_dict = models_dict.get("image_model", {})

        cap_spec = ModelSpec(
            name=cap_dict.get("name", "dummy-captioner"),
            params=cap_dict.get("params", {}),
        )
        img_spec = ModelSpec(
            name=img_dict.get("name", "dummy-imagegen"),
            params=img_dict.get("params", {}),
        )
        models_cfg = ModelsConfig(caption_model=cap_spec, image_model=img_spec)

        prompts_dict = raw.get("prompts", {})
        prompts_cfg = PromptsConfig(
            naive=prompts_dict.get("naive", "prompts/naive.json"),
            raa_aware=prompts_dict.get("raa_aware", "prompts/raa_aware.json"),
        )

        log_dict = raw.get("logging", {})
        logging_cfg = LoggingConfig(
            level=log_dict.get("level", "INFO"),
            save_config_snapshot=bool(log_dict.get("save_config_snapshot", True)),
        )

        meta_dict = raw.get("metadata", {})
        metadata_cfg = MetadataConfig(random_seed=meta_dict.get("random_seed", None))

        eval_cfg = raw.get("evaluation", {})
        rep_cfg = raw.get("reporting", {})

        return BenchmarkConfig(
            experiment_name=exp_name,
            input_dir=inp_dir,
            loop=loop_cfg,
            output_dir=output_dir,
            models=models_cfg,
            prompts=prompts_cfg,
            logging=logging_cfg,
            metadata=metadata_cfg,
            evaluation=eval_cfg,
            reporting=rep_cfg,
        )
