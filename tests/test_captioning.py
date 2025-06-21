from prompt_engine import generate_caption
from test_utils import test_image, mock_google_api


def test_generate_caption_returns_nonempty_string(test_image, mock_google_api):
    """Test that generate_caption returns a non-empty string with API."""
    caption = generate_caption(str(test_image), prompt="This is a test prompt")
    assert isinstance(caption, str) and len(caption.strip()) > 0


def test_generate_caption_no_api_key(test_image, monkeypatch):
    """Test that generate_caption falls back to placeholder with no API key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    caption = generate_caption(str(test_image), prompt="This is a test prompt")
    assert "placeholder caption" in caption.lower()
    assert test_image.name in caption  # contains filename


def test_image_context_manager_used(test_image, mock_google_api, monkeypatch):
    """Ensure Image.open is used as a context manager."""

    closed = {"flag": False}

    class DummyImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            closed["flag"] = True

        def copy(self):
            return self

    def dummy_open(path):
        return DummyImage()

    monkeypatch.setattr("prompt_engine.Image.open", dummy_open)

    generate_caption(str(test_image), prompt="Prompt")
    assert closed["flag"]
