"""Test utilities, fixtures, and mock objects for RaA tests."""

import io
import types

import pytest
from PIL import Image


def create_mock_image(size=(64, 64), color="white"):
    """Create a mock image for testing."""
    return Image.new("RGB", size, color=color)


def get_mock_image_bytes(size=(64, 64), color="white"):
    """Get bytes of a mock image."""
    img = create_mock_image(size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mock_generate_caption(image_path: str, **_) -> str:
    """Mock implementation of caption generation."""
    return f"Mock caption for {image_path}"


def mock_generate_image(prompt: str, text: str) -> Image.Image:
    """Mock implementation of image generation."""
    return create_mock_image()


# Mock Google API Classes
class MockGeneratedImage:
    """Mock for Google's GeneratedImage class."""

    def __init__(self, size=(64, 64), color="white"):
        self.image_bytes = get_mock_image_bytes(size, color)


class MockGenerateResponse:
    """Mock for Google's GenerateResponse class."""

    def __init__(self, size=(64, 64), color="white"):
        self.generated_images = [
            types.SimpleNamespace(image=MockGeneratedImage(size, color))
        ]
        self.text = "Mock generated text"


class MockModels:
    """Mock for Google API Models."""

    def generate_content(self, model: str, contents: list):
        return MockGenerateResponse()

    def generate_images(self, model: str, prompt: str, config: dict):
        return MockGenerateResponse()


class MockGoogleClient:
    """Mock for Google API Client."""

    def __init__(self, api_key: str):
        self.models = MockModels()


@pytest.fixture
def mock_google_api(monkeypatch):
    """Fixture to mock Google API client and set dummy API key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    class MockGenAI:
        """Mock for the google.generativeai module."""

        @staticmethod
        def Client(*args, **kwargs):
            return MockGoogleClient("dummy-key")

    monkeypatch.setattr("prompt_engine.genai", MockGenAI)
    return MockGoogleClient("dummy-key")


@pytest.fixture
def test_image(tmp_path):
    """Fixture to create a test image file."""
    img_path = tmp_path / "test.jpg"
    create_mock_image().save(img_path)
    return img_path


@pytest.fixture
def mock_evaluation_engine(monkeypatch):
    """Fixture to mock evaluation engine responses."""

    def mock_run_rater(self, kind, a, b):
        return {"score": 5, "reason": "Mock evaluation"}

    monkeypatch.setattr("evaluation_engine.EvaluationEngine._run_rater", mock_run_rater)
