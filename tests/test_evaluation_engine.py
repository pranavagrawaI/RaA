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
        lambda self, kind, a, b: {"score": 5, "reason": "ok"},
    )

    class DummyClient:
        pass

    engine = EvaluationEngine(str(exp), client=DummyClient())
    assert isinstance(engine.client, DummyClient)
    engine.run()

    ratings_path = item / "eval" / "ratings.json"
    assert ratings_path.is_file()
    data = json.loads(ratings_path.read_text())
    assert data and data[0]["score"] == 5


def test_extract_response_text_candidates():
    eng = EvaluationEngine("foo")
    resp = types.SimpleNamespace(
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="hello")])
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
        return '{"score": 4, "reason": "works"}'

    monkeypatch.setattr(EvaluationEngine, "_extract_response_text", fake_extract)

    rating = eng._run_rater("text-text", str(a), str(b))
    assert called.get("resp") is dummy_resp
    assert rating == {"score": 4, "reason": "works"}
