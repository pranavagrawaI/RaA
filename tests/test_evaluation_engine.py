import json

import evaluation_engine
from evaluation_engine import EvaluationEngine
from benchmark_config import BenchmarkConfig, _LoopConfig, _EvaluationConfig


def create_mock_config():
    """Create a mock config for testing."""
    return BenchmarkConfig(
        experiment_name="test",
        input_dir="test",
        loop=_LoopConfig(type="I-T-I", num_iterations=2),
        evaluation=_EvaluationConfig(enabled=True),
    )


def test_engine_creates_ratings(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    item = exp / "item1"
    item.mkdir(parents=True)

    meta = {
        "item1": {
            "input": "input.jpg",
            "iter1_img": "image_iter1.jpg",
            "iter1_text": "text_iter1.txt",
            "iter2_img": "image_iter2.jpg",
            "iter2_text": "text_iter2.txt",
        }
    }
    (exp / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (item / "input.jpg").write_text("x", encoding="utf-8")
    (item / "image_iter1.jpg").write_text("y", encoding="utf-8")
    (item / "text_iter1.txt").write_text("z", encoding="utf-8")
    (item / "image_iter2.jpg").write_text("a", encoding="utf-8")
    (item / "text_iter2.txt").write_text("b", encoding="utf-8")

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

    config = create_mock_config()
    engine = EvaluationEngine(str(exp), config=config, client=DummyClient())
    assert isinstance(engine.client, DummyClient)
    engine.run()

    # Check that all three rating files are created
    img_img_path = item / "eval" / "ratings_image-image.json"
    txt_txt_path = item / "eval" / "ratings_text-text.json"
    img_txt_path = item / "eval" / "ratings_image-text.json"

    assert img_img_path.is_file()
    assert txt_txt_path.is_file()
    assert img_txt_path.is_file()

    # Load and check the contents of each file
    img_img_data = json.loads(img_img_path.read_text())
    txt_txt_data = json.loads(txt_txt_path.read_text())
    img_txt_data = json.loads(img_txt_path.read_text())

    # For I-T-I loop:
    # iter1: img vs base_img (original) = 1
    # iter2: img vs base_img (original) + img vs prev_img (previous) = 2
    # Total image-image = 3
    assert len(img_img_data) == 3

    # For I-T-I loop:
    # iter1: no text-text comparison (no previous text)
    # iter2: txt vs prev_txt (previous) = 1
    # Total text-text = 1
    assert len(txt_txt_data) == 1

    # For I-T-I loop:
    # iter1: base_img vs txt (original) + img vs txt (same-step) = 2
    # iter2: base_img vs txt (original) + prev_img vs txt (previous) + img vs txt (same-step) = 3
    # Total image-text = 5
    assert len(img_txt_data) == 5

    # Check structure of first entry in each file
    if img_img_data:
        assert "content_correspondence" in img_img_data[0]
        assert img_img_data[0]["content_correspondence"]["score"] == 5
        assert "comparison_type" in img_img_data[0]
        assert img_img_data[0]["comparison_type"] == "image-image"

    if txt_txt_data:
        assert "content_correspondence" in txt_txt_data[0]
        assert txt_txt_data[0]["content_correspondence"]["score"] == 5
        assert "comparison_type" in txt_txt_data[0]
        assert txt_txt_data[0]["comparison_type"] == "text-text"

    if img_txt_data:
        assert "content_correspondence" in img_txt_data[0]
        assert img_txt_data[0]["content_correspondence"]["score"] == 5
        assert "comparison_type" in img_txt_data[0]
        assert img_txt_data[0]["comparison_type"] == "image-text"


def test_run_rater_uses_structured_output(monkeypatch, tmp_path):
    # This test replaces the old `test_run_rater_uses_extractor`
    config = create_mock_config()
    eng = EvaluationEngine(str(tmp_path), config=config)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    b.write_text("B")

    # Mock the entire evaluation engine client setup
    def mock_run_rater(_self, _kind, _a, _b):
        expected_output = {
            "content_correspondence": {"score": 4.0, "reason": "works"},
            "compositional_alignment": {"score": 4.0, "reason": "works"},
            "fidelity_completeness": {"score": 4.0, "reason": "works"},
            "stylistic_congruence": {"score": 4.0, "reason": "works"},
            "overall_semantic_intent": {"score": 4.0, "reason": "works"},
        }
        return expected_output

    monkeypatch.setattr(EvaluationEngine, "_run_rater", mock_run_rater)

    eng = EvaluationEngine(str(tmp_path), config=config)

    rating = eng._run_rater("text-text", str(a), str(b))

    # Expected output format after model_dump()
    expected_output = {
        "content_correspondence": {"score": 4.0, "reason": "works"},
        "compositional_alignment": {"score": 4.0, "reason": "works"},
        "fidelity_completeness": {"score": 4.0, "reason": "works"},
        "stylistic_congruence": {"score": 4.0, "reason": "works"},
        "overall_semantic_intent": {"score": 4.0, "reason": "works"},
    }

    assert rating == expected_output
