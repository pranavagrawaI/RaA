import json
from pathlib import Path
import pytest

from benchmark_config import BenchmarkConfig
from loop_controller import LoopController


def _create_dummy_texts(directory: Path):
    """Populate *directory* with two placeholder .txt prompts."""
    directory.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        (directory / f"{stem}.txt").write_text(
            f"Initial prompt for {stem}.", encoding="utf-8"
        )


@pytest.fixture
def tit_config(tmp_path):
    """Return a BenchmarkConfig pointing to dummy .txt inputs for a T-I-T run."""
    input_dir = tmp_path / "input_txt"
    _create_dummy_texts(input_dir)

    out_dir = tmp_path / "out_results"

    yaml_content = f"""
        experiment_name: 'tit_test'
        input_dir: '{input_dir}'
        output_dir: '{out_dir}'
        loop:
          type: 'T-I-T'
          num_iterations: 2
        """

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)

    return BenchmarkConfig.from_yaml(str(cfg_file))


def test_tit_loop_generates_expected_files(tit_config):
    """Run LoopController in T-I-T mode and assert placeholder outputs exist."""
    controller = LoopController(tit_config)
    controller.run()

    out_dir = Path(tit_config.output_dir)

    # Global metadata
    meta_path = out_dir / "metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert set(meta.keys()) == {"a", "b"}

    # Each subfolder should have input.txt, image_iter{i}.jpg, text_iter{i}.txt
    for stem in ("a", "b"):
        subdir = out_dir / stem
        assert subdir.is_dir()
        assert (subdir / "input.txt").exists()
        for i in (1, 2):
            assert (subdir / f"image_iter{i}.jpg").exists()
            assert (subdir / f"text_iter{i}.txt").exists()
