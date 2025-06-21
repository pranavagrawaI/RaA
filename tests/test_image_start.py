import json
import os
from unittest.mock import patch

import pytest
from PIL import Image
from test_utils import mock_generate_caption, mock_generate_image

from benchmark_config import BenchmarkConfig
from loop_controller import LoopController

DUMMY_YAML = """
experiment_name: "exp_test"
input_dir: "{input_dir}"
output_dir: "{output_dir}"
loop:
  type: "I-T-I"
  num_iterations: {num_iter}
models:
  caption_model:
    name: "m1"
    params: {{}}
  image_model:
    name: "m2"
    params: {{}}
prompts:
  naive: "p1"
  raa_aware: "p2"
logging:
  level: "INFO"
  save_config_snapshot: false
metadata:
  random_seed: 0
evaluation:
  enabled: true
  mode: "llm"
reporting: {{}}
"""


@pytest.fixture(name="two_dummy_images")
def dummy_images_2(tmp_path):
    d = tmp_path / "input_data"
    d.mkdir()
    for name in ["a.png", "b.png"]:
        img = Image.new("RGB", (2, 2), color="white")
        img.save(d / name)
    return str(d)


def make_config(tmp_path, input_dir, num_iter=1):
    out_dir = str(tmp_path / "out_data")
    text = DUMMY_YAML.format(
        input_dir=input_dir.replace("\\", "\\\\"),  # escape backslashes on Windows
        output_dir=out_dir.replace("\\", "\\\\"),
        num_iter=num_iter,
    )
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(text)
    return str(cfg_file), out_dir


@patch("loop_controller.generate_image", side_effect=mock_generate_image)
@patch("loop_controller.generate_caption", side_effect=mock_generate_caption)
def test_single_image_one_iter(mock_caption, mock_image, tmp_path, two_dummy_images):
    img_dir = two_dummy_images
    os.remove(os.path.join(img_dir, "b.png"))

    cfg_path, out_dir = make_config(tmp_path, img_dir, num_iter=1)
    config = BenchmarkConfig.from_yaml(cfg_path)
    lc = LoopController(config)
    lc.run()

    sub = os.path.join(out_dir, "a")
    assert os.path.isdir(sub)
    assert os.path.exists(os.path.join(sub, "input.jpg"))

    assert os.path.exists(os.path.join(sub, "text_iter1.txt"))
    assert os.path.exists(os.path.join(sub, "image_iter1.jpg"))

    meta = json.loads(
        open(os.path.join(out_dir, "metadata.json"), "r", encoding="utf-8").read()
    )
    assert "a" in meta
    assert meta["a"]["iter1_text"] == "text_iter1.txt"
    assert meta["a"]["iter1_img"] == "image_iter1.jpg"

    assert mock_caption.called
    assert mock_image.called


@patch("loop_controller.generate_image", side_effect=mock_generate_image)
@patch("loop_controller.generate_caption", side_effect=mock_generate_caption)
def test_two_images_two_iters_stateful(
    mock_caption, mock_image, tmp_path, two_dummy_images
):
    cfg_path, out_dir = make_config(tmp_path, two_dummy_images, num_iter=2)
    config = BenchmarkConfig.from_yaml(cfg_path)
    lc = LoopController(config)
    lc.run()

    for img_name in ["a", "b"]:
        sub = os.path.join(out_dir, img_name)
        assert os.path.isdir(sub)
        assert os.path.exists(os.path.join(sub, "input.jpg"))

        for i in [1, 2]:
            assert os.path.exists(os.path.join(sub, f"text_iter{i}.txt"))
            assert os.path.exists(os.path.join(sub, f"image_iter{i}.jpg"))

    assert mock_caption.call_count == 4
    assert mock_image.call_count == 4
