import types
from unittest.mock import MagicMock
from evaluation_engine import EvaluationEngine, DEFAULT_RATING


def test_run_rater_missing_image_returns_default(tmp_path):
    engine = EvaluationEngine(str(tmp_path))
    # Mock the client to avoid actual API calls
    engine.client = MagicMock()
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
    mock_response = types.SimpleNamespace(parsed=None, text="Invalid JSON")

    class DummyModel:
        def generate_content(self, *args, **kwargs):
            return mock_response

    class DummyClient:
        def __init__(self, api_key=None):
            self.models = DummyModel()

    monkeypatch.setattr("evaluation_engine.genai.Client", DummyClient)

    engine = EvaluationEngine(str(tmp_path))
    engine.client = DummyClient()  # Ensure the mocked client is used

    rating = engine._run_rater("text-text", str(text_a), str(missing))
    assert rating == DEFAULT_RATING


def test_run_rater_human_mode(tmp_path, monkeypatch):
    engine = EvaluationEngine(str(tmp_path), mode="human")

    # Simulate a sequence of user inputs for all criteria
    inputs = ["5", "reason1", "6", "reason2", "7", "reason3", "8", "reason4", "9", "reason5"]
    input_iterator = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _: next(input_iterator))

    rating = engine._run_rater("text-text", "a.txt", "b.txt")

    # Verify that the returned rating matches the simulated input
    expected_rating = {
        "content_correspondence": {"score": 5.0, "reason": "reason1"},
        "compositional_alignment": {"score": 6.0, "reason": "reason2"},
        "fidelity_completeness": {"score": 7.0, "reason": "reason3"},
        "stylistic_congruence": {"score": 8.0, "reason": "reason4"},
        "overall_semantic_intent": {"score": 9.0, "reason": "reason5"},
    }
    assert rating == expected_rating