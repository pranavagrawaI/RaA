import json
import types

import evaluation_engine
from evaluation_engine import EvaluationEngine


def test_engine_creates_ratings(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    item = exp / "item1"
    item.mkdir(parents=True)

    meta = {
        "item1": {
            "input": "input.jpg",
            "iter1_img": "image_iter1.jpg",
            "iter1_text": "text_iter1.txt",
        }
    }
    (exp / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (item / "input.jpg").write_text("x", encoding="utf-8")
    (item / "image_iter1.jpg").write_text("y", encoding="utf-8")
    (item / "text_iter1.txt").write_text("z", encoding="utf-8")

    # Ensure the EvaluationEngine does not attempt real API calls
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    monkeypatch.setattr(
        EvaluationEngine,
        "_run_rater",
        lambda self, kind, a, b: {
            "content_correspondence": {"score": 5, "reason": "test"},
            "compositional_alignment": {"score": 5, "reason": "test"},
            "fidelity_completeness": {"score": 5, "reason": "test"},
            "stylistic_congruence": {"score": 5, "reason": "test"},
            "overall_semantic_intent": {"score": 5, "reason": "test"},
        },
    )

    class DummyClient(evaluation_engine.genai.Client):
        pass

    engine = EvaluationEngine(str(exp), client=DummyClient())
    assert isinstance(engine.client, DummyClient)
    engine.run()

    ratings_path = item / "eval" / "ratings.json"
    assert ratings_path.is_file()
    data = json.loads(ratings_path.read_text())
    assert data
    assert "content_correspondence" in data[0]
    assert data[0]["content_correspondence"]["score"] == 5
    assert "compositional_alignment" in data[0]
    assert data[0]["compositional_alignment"]["score"] == 5
    assert "fidelity_completeness" in data[0]
    assert data[0]["fidelity_completeness"]["score"] == 5
    assert "stylistic_congruence" in data[0]
    assert data[0]["stylistic_congruence"]["score"] == 5
    assert "overall_semantic_intent" in data[0]
    assert data[0]["overall_semantic_intent"]["score"] == 5


def test_run_rater_uses_structured_output(monkeypatch, tmp_path):
    # This test replaces the old `test_run_rater_uses_extractor`
    eng = EvaluationEngine(str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    b.write_text("B")

    # The expected data structure that the model should return
    mock_rating_data = {
        "content_correspondence": {"score": 4.0, "reason": "works"},
        "compositional_alignment": {"score": 4.0, "reason": "works"},
        "fidelity_completeness": {"score": 4.0, "reason": "works"},
        "stylistic_congruence": {"score": 4.0, "reason": "works"},
        "overall_semantic_intent": {"score": 4.0, "reason": "works"},
    }

    # Create a mock _RatingModel instance from the data
    mock_rating_model = evaluation_engine._RatingModel(**mock_rating_data)

    # Create a mock response object that mimics the Gemini API response
    mock_response = types.SimpleNamespace(parsed=mock_rating_model)

    class DummyModel:
        def generate_content(self, *args, **kwargs):
            # Check if the response schema is correctly requested
            config = kwargs.get("config", {})
            assert config.response_mime_type == "application/json"
            assert config.response_schema == evaluation_engine._RatingModel
            return mock_response

    class DummyClient:
        def __init__(self, api_key=None):
            self.models = DummyModel()

    monkeypatch.setattr(evaluation_engine.genai, "Client", DummyClient)
    eng.client = DummyClient()  # Re-initialize client on the instance

    rating = eng._run_rater("text-text", str(a), str(b))
    assert rating == mock_rating_data