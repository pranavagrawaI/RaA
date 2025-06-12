import types
from PIL import Image
import prompt_engine
from prompt_engine import generate_caption

def test_generate_caption_returns_nonempty_string(tmp_path):
    fake_path = str(tmp_path / "some_image.jpg")
    # Create a valid image file instead of writing text
    Image.new("RGB", (2, 2)).save(fake_path)
    caption = generate_caption(fake_path, prompt="This is a test prompt")
    assert isinstance(caption, str) and len(caption.strip()) > 0


def test_generate_caption_with_gemini(monkeypatch, tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (2, 2)).save(img_path)

    class DummyResponse:
        text = "a test caption"

    class MockModels:
        def generate_content(self, model: str, contents: list):
            return DummyResponse()

    class MockClientInstance:
        def __init__(self, api_key: str):
            self.models = MockModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(
        prompt_engine,
        "genai",
        types.SimpleNamespace(Client=MockClientInstance),
    )

    caption = generate_caption(str(img_path), prompt="This is a test prompt")
    assert caption == "a test caption"
