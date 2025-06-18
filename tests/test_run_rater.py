import types
from evaluation_engine import EvaluationEngine


def test_run_rater_missing_image_returns_default(tmp_path):
    engine = EvaluationEngine(str(tmp_path))
    rating = engine._run_rater("image-image", str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg"))
    assert rating["score"] == 3
    assert "missing" in rating["reason"].lower()


def test_run_rater_invalid_json_from_gemini(tmp_path, monkeypatch):
    text_a = tmp_path / "a.txt"
    text_a.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    class DummyClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, *args, **kwargs):
            return types.SimpleNamespace(text="nonsense without digits")

    monkeypatch.setattr("evaluation_engine.genai.Client", DummyClient)

    engine = EvaluationEngine(str(tmp_path))
    rating = engine._run_rater("text-text", str(text_a), str(missing))
    assert rating["score"] == 3
    assert isinstance(rating.get("reason"), str) and rating["reason"]
