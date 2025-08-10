import json
from pathlib import Path

from autorater_visualizer import AutoRaterVisualizer


def test_plot_score_trajectories(tmp_path):
    eval_dir = tmp_path / "item1" / "eval"
    eval_dir.mkdir(parents=True)
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
            "stylistic_congruence": {"score": 5.0, "reason": ""},
            "overall_semantic_intent": {"score": 4.0, "reason": ""},
        },
        {
            "item_id": "item1",
            "step": 2,
            "anchor": "original",
            "comparison_type": "image-image",
            "comparison_items": ["c", "d"],
            "content_correspondence": {"score": 7.5, "reason": ""},
            "compositional_alignment": {"score": 6.5, "reason": ""},
            "fidelity_completeness": {"score": 5.5, "reason": ""},
            "stylistic_congruence": {"score": 4.5, "reason": ""},
            "overall_semantic_intent": {"score": 3.5, "reason": ""},
        },
    ]
    ratings_file = eval_dir / "ratings.json"
    ratings_file.write_text(json.dumps(data))

    viz = AutoRaterVisualizer(ratings_file)
    out_path = viz.plot_score_trajectories()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
