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

    monkeypatch.setattr(
        EvaluationEngine,
        "_run_rater",
        lambda self, kind, a, b: {
            "content": {"score": 5, "reason": "test"},
            "missing": {"score": 5, "reason": "test"},
            "style": {"score": 5, "reason": "test"},
            "meaning": {"score": 5, "reason": "test"},
            "overall": {"score": 5, "reason": "test"},
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
    assert "content" in data[0]
    assert data[0]["content"]["score"] == 5
    assert "missing" in data[0]
    assert data[0]["missing"]["score"] == 5
    assert "style" in data[0]
    assert data[0]["style"]["score"] == 5
    assert "meaning" in data[0]
    assert data[0]["meaning"]["score"] == 5
    assert "overall" in data[0]
    assert data[0]["overall"]["score"] == 5


def test_extract_response_text_candidates():
    eng = EvaluationEngine("foo")
    resp = types.SimpleNamespace(
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(text="hello")]
                )
            )
        ]
    )
    assert eng._extract_response_text(resp) == "hello"


def test_run_rater_uses_extractor(monkeypatch, tmp_path):
    eng = EvaluationEngine(str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    b.write_text("B")

    dummy_resp = object()

    class DummyModel:
        def generate_content(self, model, contents):
            return dummy_resp

    class DummyClient:
        def __init__(self, api_key):
            self.models = DummyModel()

    monkeypatch.setattr(evaluation_engine.genai, "Client", DummyClient)

    called = {}

    def fake_extract(self, resp):
        called["resp"] = resp
        return '{"content": {"score": 4, "reason": "works"}, "missing": {"score": 4, "reason": "works"}, "style": {"score": 4, "reason": "works"}, "meaning": {"score": 4, "reason": "works"}, "overall": {"score": 4, "reason": "works"}}'

    monkeypatch.setattr(EvaluationEngine, "_extract_response_text", fake_extract)

    rating = eng._run_rater("text-text", str(a), str(b))
    assert rating == {
        "content": {"score": 4, "reason": "works"},
        "missing": {"score": 4, "reason": "works"},
        "style": {"score": 4, "reason": "works"},
        "meaning": {"score": 4, "reason": "works"},
        "overall": {"score": 4, "reason": "works"},
    }
