import json
import types

from evaluation_engine import EvaluationEngine


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.models = self

    def generate_content(self, *args, **kwargs):
        return types.SimpleNamespace(text="dummy response")


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

    engine = EvaluationEngine(str(exp), client=DummyClient())  # type: ignore
    assert isinstance(engine.client, DummyClient)
    engine.run()

    ratings_path = item / "eval" / "ratings.json"
    assert ratings_path.is_file()
    data = json.loads(ratings_path.read_text())
    assert data and data[0]["score"] == 5


def test_extract_response_text_candidates():
    eng = EvaluationEngine("foo", mode="human")
    resp = types.SimpleNamespace(
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(text="hello")]
                )
            )
        ]
    )
    assert eng._extract_response_text(resp) == "hello"  # pylint: disable=protected-access


def test_run_rater_uses_extractor(tmp_path):
    class DummyClient2:
        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content(self, *args, **kwargs):
            return types.SimpleNamespace(text='{"score": 4, "reason": "ok"}')

    eng = EvaluationEngine(str(tmp_path), client=DummyClient2())  # type: ignore
    rating = eng._run_rater(  # pylint: disable=protected-access
        "text-text", str(tmp_path / "a.txt"), str(tmp_path / "b.txt")
    )
    assert rating["score"] == 4
    assert rating["reason"] == "ok"
