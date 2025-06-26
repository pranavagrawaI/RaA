import types
from evaluation_engine import EvaluationEngine


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.models = self

    def generate_content(self, *args, **kwargs):
        return types.SimpleNamespace(text="dummy response")


def test_run_rater_missing_image_returns_default(tmp_path):
    engine = EvaluationEngine(str(tmp_path), client=DummyClient())  # type: ignore
    rating = engine._run_rater(  # pylint: disable=protected-access
        "image-image", str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    )
    assert rating["score"] == -1
    assert "missing" in rating["reason"].lower()


def test_run_rater_invalid_json_from_gemini(tmp_path):
    text_a = tmp_path / "a.txt"
    text_a.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    class DummyClient2:
        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content(self, *args, **kwargs):
            return types.SimpleNamespace(text="nonsense without digits")

    engine = EvaluationEngine(str(tmp_path), client=DummyClient2())  # type: ignore
    rating = engine._run_rater("text-text", str(text_a), str(missing))  # pylint: disable=protected-access
    assert rating["score"] == -1
    assert isinstance(rating.get("reason"), str) and rating["reason"]
