import json

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
