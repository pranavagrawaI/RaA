from PIL import Image
from test_utils import mock_google_api

from prompt_engine import generate_image


def test_generate_image_creation(mock_google_api):
    """Test that generate_image returns an image with expected properties."""
    result = generate_image(prompt="Generate an image", text="A test description")

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (64, 64)


def test_generate_image_no_api_key(monkeypatch):
    """Test that generate_image returns a gray image when no API key is present."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = generate_image(prompt="test prompt", text="test text")

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (32, 32)
    assert result.getpixel((0, 0)) == (128, 128, 128)
