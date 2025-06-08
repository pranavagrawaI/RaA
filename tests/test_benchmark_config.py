import pytest

from benchmark_config import BenchmarkConfig

VALID_YAML = """
experiment_name: "exp_test"
input_dir: "foo"
output_dir: "bar/{{experiment_name}}"
loop:
  type: "I-T-I"
  num_iterations: 3
  stateless: false
models:
  caption_model:
    name: "m1"
    params: {}
  image_model:
    name: "m2"
    params: {}
prompts:
  naive: "p1"
  raa_aware: "p2"
logging:
  level: "DEBUG"
  save_config_snapshot: false
metadata:
  random_seed: 123
evaluation: {}
reporting: {}
"""

MINIMAL_YAML = """
experiment_name: "exp_minimal"
input_dir: "data/somefolder"

loop:
  type: "I-T-I"
  num_iterations: 1
  stateless: true
"""


@pytest.fixture
def valid_cfg_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(VALID_YAML)
    return str(path)


def test_valid_config_loads(valid_cfg_file):
    cfg = BenchmarkConfig.from_yaml(valid_cfg_file)
    assert cfg.experiment_name == "exp_test"
    assert cfg.input_dir == "foo"
    # Double-brace renders to single brace, then format → "bar/exp_test"
    assert cfg.output_dir.endswith("bar/exp_test")
    assert cfg.loop.num_iterations == 3
    assert cfg.logging.level == "DEBUG"


def test_minimal_config_loads(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(MINIMAL_YAML)
    cfg = BenchmarkConfig.from_yaml(str(cfg_file))

    assert cfg.experiment_name == "exp_minimal"
    assert cfg.input_dir == "data/somefolder"
    # Defaults:
    assert cfg.output_dir.endswith("results/exp_minimal")
    assert cfg.loop.type == "I-T-I"
    assert cfg.loop.num_iterations == 1
    assert cfg.models.caption_model.name == "dummy-captioner"
    assert cfg.logging.level == "INFO"


def test_missing_key_raises(tmp_path):
    bad_yaml = VALID_YAML.replace('input_dir: "foo"', "")
    path = tmp_path / "bad.yaml"
    path.write_text(bad_yaml)
    with pytest.raises(KeyError):
        BenchmarkConfig.from_yaml(str(path))


def test_malformed_output_template(tmp_path):
    malformed = VALID_YAML.replace("bar/{{experiment_name}}", "bar/{experiment_name")
    path = tmp_path / "bad2.yaml"
    path.write_text(malformed)
    with pytest.raises(ValueError):
        BenchmarkConfig.from_yaml(str(path))
