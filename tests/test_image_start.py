import os
import errno

import pytest
from PIL import Image

from benchmark_config import BenchmarkConfig
from loop_controller import LoopController

DUMMY_YAML = """
experiment_name: "exp_test"
input_dir: "{input_dir}"
output_dir: "{output_dir}"
loop:
  type: "I-T-I"
  num_iterations: {num_iter}
  stateless: {stateless}
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
evaluation: {{}}
reporting: {{}}
"""


@pytest.fixture
def two_dummy_images(tmp_path):
    d = tmp_path / "input_data"
    d.mkdir()
    # Create two tiny PNGs
    for name in ["a.png", "b.png"]:
        img = Image.new("RGB", (2, 2), color="white")
        img.save(d / name)
    return str(d)


def make_config(tmp_path, input_dir, num_iter=1, stateless=True):
    out_dir = str(tmp_path / "out_data")
    text = DUMMY_YAML.format(
        input_dir=input_dir.replace("\\", "\\\\"),  # escape backslashes on Windows
        output_dir=out_dir.replace("\\", "\\\\"),
        num_iter=num_iter,
        stateless=str(stateless).lower(),
    )
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(text)
    return str(cfg_file), out_dir


def test_single_image_one_iter(tmp_path, two_dummy_images, monkeypatch):
    # Use only the first image for simplicity
    img_dir = two_dummy_images
    # Temporarily rename so only "a.png" is seen
    os.remove(os.path.join(img_dir, "b.png"))

    cfg_path, out_dir = make_config(tmp_path, img_dir, num_iter=1, stateless=True)
    config = BenchmarkConfig.from_yaml(cfg_path)
    lc = LoopController(config)
    lc.run()

    # Verify folder structure
    sub = os.path.join(out_dir, "a")
    assert os.path.isdir(sub)
    # input.jpg should be symlink or copy
    assert os.path.exists(os.path.join(sub, "input.jpg"))

    # text_iter1.txt and image_iter1.jpg must exist
    assert os.path.exists(os.path.join(sub, "text_iter1.txt"))
    assert os.path.exists(os.path.join(sub, "image_iter1.jpg"))

    # metadata.json should map "a" → {…}
    import json

    meta = json.loads(open(os.path.join(out_dir, "metadata.json"), 'r', encoding='utf-8').read())
    assert "a" in meta
    assert meta["a"]["iter1_text"] == "text_iter1.txt"
    assert meta["a"]["iter1_img"] == "image_iter1.jpg"


def test_two_images_two_iters_stateful(tmp_path, two_dummy_images):
    cfg_path, out_dir = make_config(
        tmp_path, two_dummy_images, num_iter=2, stateless=False
    )
    config = BenchmarkConfig.from_yaml(cfg_path)
    lc = LoopController(config)
    lc.run()

    for img_name in ["a", "b"]:
        sub = os.path.join(out_dir, img_name)
        assert os.path.isdir(sub)
        assert os.path.exists(os.path.join(sub, "input.jpg"))

        # Check both iterations exist
        for i in [1, 2]:
            assert os.path.exists(os.path.join(sub, f"text_iter{i}.txt"))
            assert os.path.exists(os.path.join(sub, f"image_iter{i}.jpg"))

    # Verify at least that the second iteration used image_iter1 as input
    # We could monkeypatch generate_caption to record calls, but for now just check files exist


def test_symlink_fallback(monkeypatch, tmp_path, two_dummy_images):
    # Force os.symlink to raise EPERM
    orig_symlink = os.symlink

    def fake_symlink(src, dst):
        raise OSError(errno.EPERM, "No perms")

    monkeypatch.setattr(os, "symlink", fake_symlink)

    cfg_path, out_dir = make_config(
        tmp_path, two_dummy_images, num_iter=1, stateless=True
    )
    config = BenchmarkConfig.from_yaml(cfg_path)
    lc = LoopController(config)
    lc.run()

    sub = os.path.join(out_dir, "a")
    # If fallback worked, input.jpg exists as a real copy
    assert os.path.isfile(os.path.join(sub, "input.jpg"))

    # Restore original symlink
    monkeypatch.setattr(os, "symlink", orig_symlink)
