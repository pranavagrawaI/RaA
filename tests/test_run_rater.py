from types import SimpleNamespace
from unittest.mock import MagicMock
from evaluation_engine import EvaluationEngine, DEFAULT_RATING
from benchmark_config import BenchmarkConfig


def create_mock_config():
    """Create a mock that satisfies the BenchmarkConfig interface."""
    mock_config = MagicMock(spec=BenchmarkConfig)
    mock_config.loop = MagicMock(type="I-T-I")
    return mock_config


def test_run_rater_missing_image_returns_default(tmp_path, monkeypatch):
    # Create a mock config
    engine = EvaluationEngine(str(tmp_path), config=create_mock_config())
    # Mock the client to avoid actual API calls
    monkeypatch.setattr(engine, "client", MagicMock())
    rating = engine._run_rater(
        "image-image", str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    )
    assert rating == DEFAULT_RATING


def test_run_rater_invalid_json_from_gemini(tmp_path, monkeypatch):
    text_a = tmp_path / "a.txt"
    text_a.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    # This file does not exist, which will cause an error handled by the rater

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    # Mock the response to simulate a failure in structured output
    mock_response = SimpleNamespace(parsed=None, text="Invalid JSON")

    class DummyModel:
        def generate_content(self, *args, **kwargs):
            return mock_response

    class DummyClient:
        def __init__(self, api_key=None):
            self.models = DummyModel()

    monkeypatch.setattr("evaluation_engine.genai.Client", DummyClient)

    engine = EvaluationEngine(str(tmp_path), config=create_mock_config())
    monkeypatch.setattr(
        engine, "client", DummyClient()
    )  # Use monkeypatch to set client

    rating = engine._run_rater("text-text", str(text_a), str(missing))
    assert rating == DEFAULT_RATING
