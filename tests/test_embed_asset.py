import os
from prompt_engine import embed_asset
from test_utils import create_mock_image


def test_embed_asset_reads_text(tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text("hello", encoding="utf-8")
    assert embed_asset(str(txt)) == "hello"


def test_embed_asset_returns_image_path(tmp_path):
    img_path = tmp_path / "img.png"
    create_mock_image().save(img_path)
    assert embed_asset(str(img_path)) == str(img_path)

