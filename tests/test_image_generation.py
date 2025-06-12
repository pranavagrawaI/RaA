import io
import types
import pytest
from PIL import Image
import prompt_engine
from prompt_engine import generate_image


def test_generate_image_dimensions(monkeypatch):
    """Test that generate_image returns an image with the correct mode and size (mocked)."""
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fake_bytes = buf.getvalue()

    class DummyGeneratedImage:
        image_bytes = fake_bytes

    class DummyResponse:
        generated_images = [types.SimpleNamespace(image=DummyGeneratedImage())]

    class MockModels:
        def generate_images(self, model, prompt, config):  # pylint: disable=unused-argument
            return DummyResponse()

    class MockClientInstance:
        def __init__(self, api_key):  # pylint: disable=unused-argument
            self.models = MockModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine, "genai", types.SimpleNamespace(Client=MockClientInstance)
    )

    out_img = generate_image(prompt="A test prompt", text="This is a test text")
    assert out_img.mode == "RGB"
    assert out_img.size == (64, 64)
    assert out_img.getpixel((0, 0)) == (255, 0, 0)


def test_generate_image_no_images(monkeypatch):
    """Test that generate_image raises if no images are returned."""

    class DummyResponse:
        generated_images = []

    class MockModels:
        def generate_images(self, model, prompt, config):  # pylint: disable=unused-argument
            return DummyResponse()

    class MockClientInstance:
        def __init__(self, api_key):  # pylint: disable=unused-argument
            self.models = MockModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine, "genai", types.SimpleNamespace(Client=MockClientInstance)
    )

    with pytest.raises(
        RuntimeError,
        match="Image generation failed: API response missing or no images returned.",
    ):
        generate_image(prompt="foo", text="bar")


def test_generate_image_missing_bytes(monkeypatch):
    """Test that generate_image raises if image bytes are missing."""

    class DummyGeneratedImage:
        image_bytes = None

    class DummyResponse:
        generated_images = [types.SimpleNamespace(image=DummyGeneratedImage())]

    class MockModels:
        def generate_images(self, model, prompt, config):  # pylint: disable=unused-argument
            return DummyResponse()

    class MockClientInstance:
        def __init__(self, api_key):  # pylint: disable=unused-argument
            self.models = MockModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine, "genai", types.SimpleNamespace(Client=MockClientInstance)
    )

    with pytest.raises(
        RuntimeError,
        match="Image generation failed: The 'image_bytes' data in API response is null.",
    ):
        generate_image(prompt="foo", text="bar")


def test_generate_image_different_prompt(monkeypatch):
    """Test that generate_image works with different prompt/text (mocked)."""
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fake_bytes = buf.getvalue()

    class DummyGeneratedImage:
        image_bytes = fake_bytes

    class DummyResponse:
        generated_images = [types.SimpleNamespace(image=DummyGeneratedImage())]

    class MockModels:
        def generate_images(self, model, prompt, config):  # pylint: disable=unused-argument
            return DummyResponse()

    class MockClientInstance:
        def __init__(self, api_key):  # pylint: disable=unused-argument
            self.models = MockModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine, "genai", types.SimpleNamespace(Client=MockClientInstance)
    )

    out_img = generate_image(prompt="Another prompt", text="Different text")
    assert out_img.size == (32, 32)
    assert out_img.getpixel((0, 0)) == (0, 0, 255)
