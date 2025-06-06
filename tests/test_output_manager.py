import json

import yaml
from PIL import Image

from output_manager import OutputManager


def test_save_text_and_read(tmp_path):
    om = OutputManager(str(tmp_path))
    om.save_text("hello world", "greet.txt")
    content = (tmp_path / "greet.txt").read_text(encoding="utf-8")
    assert content == "hello world"


def test_save_image(tmp_path):
    om = OutputManager(str(tmp_path))
    img = Image.new("RGB", (10, 10), color="blue")
    om.save_image(img, "pic.jpg")
    saved = Image.open(tmp_path / "pic.jpg")
    assert saved.size == (10, 10)
    assert saved.mode == "RGB"


def test_subdir_isolation(tmp_path):
    om = OutputManager(str(tmp_path))
    om2 = om.subdir("foo")
    om2.save_text("X", "a.txt")
    assert (tmp_path / "foo" / "a.txt").exists()
    assert not (tmp_path / "a.txt").exists()


def test_write_and_read_json(tmp_path):
    om = OutputManager(str(tmp_path))
    data = {"a": 1, "b": "two"}
    om.write_json(data, "meta.json")
    loaded = json.loads((tmp_path / "meta.json").read_text())
    assert loaded == data


def test_save_and_load_yaml(tmp_path):
    om = OutputManager(str(tmp_path))
    data = {"foo": [1, 2, 3], "nested": {"x": "y"}}
    om.save_yaml(data, "cfg_out.yaml")
    loaded = yaml.safe_load((tmp_path / "cfg_out.yaml").read_text())
    assert loaded == data
