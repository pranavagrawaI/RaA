import types

from PIL import Image

import prompt_engine
from prompt_engine import generate_caption, generate_image


def test_generate_caption_contains_filename(tmp_path):
    fake_path = str(tmp_path / "some_image.jpg")
    (tmp_path / "some_image.jpg").write_text("dummy")
    caption = generate_caption(fake_path)
    assert "some_image.jpg" in caption


def test_generate_caption_with_gemini(monkeypatch, tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (2, 2)).save(img_path)

    class DummyResponse:
        text = "a test caption"

    class DummyModel:
        def generate_content(self, parts):
            return DummyResponse()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine, "genai",
        types.SimpleNamespace(
            configure=lambda api_key=None: None,
            GenerativeModel=lambda name: DummyModel(),
        ),
    )

    caption = generate_caption(str(img_path))
    assert caption == "a test caption"


def test_generate_image_dimensions():
    img = generate_image("anything", width=32, height=16)
    assert img.size == (32, 16)
    assert img.mode == "RGB"
