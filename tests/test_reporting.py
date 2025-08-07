import json
from pathlib import Path

from reporting import generate_line_overview


def _write_ratings(path: Path) -> None:
    data = [
        {
            "item_id": "item1",
            "step": 1,
            "anchor": "original",
            "comparison_type": "image-image",
            "comparison_items": ["a", "b"],
            "content_correspondence": {"score": 8.0, "reason": ""},
            "compositional_alignment": {"score": 7.0, "reason": ""},
            "fidelity_completeness": {"score": 6.0, "reason": ""},
            "style_consistency": {"score": 5.0, "reason": ""},
            "overall": {"score": 4.0, "reason": ""},
        },
        {
            "item_id": "item1",
            "step": 2,
            "anchor": "previous",
            "comparison_type": "image-image",
            "comparison_items": ["c", "d"],
            "content_correspondence": {"score": 7.5, "reason": ""},
            "compositional_alignment": {"score": 6.5, "reason": ""},
            "fidelity_completeness": {"score": 5.5, "reason": ""},
            "style_consistency": {"score": 4.5, "reason": ""},
            "overall": {"score": 3.5, "reason": ""},
        },
    ]
    path.write_text(json.dumps(data))


def test_generate_line_overview(tmp_path):
    eval_dir = tmp_path / "item1" / "eval"
    eval_dir.mkdir(parents=True)
    ratings_file = eval_dir / "ratings.json"
    _write_ratings(ratings_file)

    generate_line_overview(tmp_path)

    out_file = eval_dir / "line_overview.html"
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
