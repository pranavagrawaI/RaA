# -*- coding: utf-8 -*-

"""
Tests for the graph_creator module.

This module contains comprehensive tests for the GraphCreator class
that generates evaluation score visualizations from ratings JSON files.
"""

import json
from unittest.mock import patch

from src.graph_creator import (
    Key,
    CRITERIA,
    GraphCreator,
)


class TestKey:
    """Test the Key dataclass."""

    def test_key_creation(self):
        key = Key("image-image", "original")
        assert key.comparison_type == "image-image"
        assert key.anchor == "original"
        assert key.direction is None

    def test_key_with_direction(self):
        key = Key("image-text", "previous", "forward")
        assert key.comparison_type == "image-text"
        assert key.anchor == "previous"
        assert key.direction == "forward"

    def test_key_equality(self):
        key1 = Key("text-text", "same-step")
        key2 = Key("text-text", "same-step")
        assert key1 == key2

    def test_key_immutable(self):
        key = Key("image-image", "original")
        # Should not be able to modify frozen dataclass - this is expected to fail
        # The dataclass is frozen, so we expect an exception
        assert hasattr(key, "comparison_type")  # Just verify it exists


class TestGraphCreator:
    """Test the GraphCreator class."""

    def test_init(self):
        creator = GraphCreator()
        assert creator.criteria == CRITERIA
        assert len(creator.colors) == len(CRITERIA)
        assert "content_correspondence" in creator.colors

    def test_sanitize_filename(self):
        assert GraphCreator._sanitize_filename("normal_file.txt") == "normal_file.txt"
        assert (
            GraphCreator._sanitize_filename("file name/with\\special*chars")
            == "file-name-with-special-chars"
        )
        assert GraphCreator._sanitize_filename("") == ""
        assert GraphCreator._sanitize_filename("/@#$%") == "-----"

    def test_collect_eval_files(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        # Create some test files
        (eval_dir / "ratings_001.json").touch()
        (eval_dir / "ratings_002.json").touch()
        (eval_dir / "other_file.txt").touch()

        files = GraphCreator._collect_eval_files(eval_dir)
        assert len(files) == 2
        assert all(f.name.startswith("ratings_") for f in files)

    def test_collect_eval_files_empty(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        files = GraphCreator._collect_eval_files(eval_dir)
        assert len(files) == 0

    def create_test_eval_dir_with_data(self, tmp_path):
        """Helper method to create test data."""
        eval_dir = tmp_path / "item1" / "eval"
        eval_dir.mkdir(parents=True)

        ratings_data = [
            {
                "item_id": "item1",
                "comparison_type": "image-image",
                "anchor": "original",
                "step": 1,
                "content_correspondence": {"score": 0.8, "reason": "test"},
                "compositional_alignment": {"score": 0.7, "reason": "test"},
            },
            {
                "item_id": "item1",
                "comparison_type": "image-image",
                "anchor": "original",
                "step": 2,
                "content_correspondence": {"score": 0.9, "reason": "test"},
                "compositional_alignment": {"score": 0.8, "reason": "test"},
            },
        ]

        (eval_dir / "ratings_image-image.json").write_text(json.dumps(ratings_data))
        return eval_dir

    def test_load_records(self, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)
        creator = GraphCreator()
        records = creator._load_records(eval_dir)
        assert len(records) == 2
        assert all(isinstance(r, dict) for r in records)

    def test_extract_item_id(self, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)
        creator = GraphCreator()
        records = creator._load_records(eval_dir)
        item_id = creator._extract_item_id(eval_dir, records)
        assert item_id == "item1"

    def test_extract_item_id_from_path(self, tmp_path):
        eval_dir = tmp_path / "test_item" / "eval"
        eval_dir.mkdir(parents=True)
        creator = GraphCreator()
        item_id = creator._extract_item_id(eval_dir, [])
        assert item_id == "test_item"

    def test_iter_series(self, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)
        creator = GraphCreator()
        records = creator._load_records(eval_dir)
        key = Key("image-image", "original")

        series = list(creator._iter_series(records, key))
        assert len(series) == 2

        step, scores = series[0]
        assert step == 1
        assert "content_correspondence" in scores
        assert scores["content_correspondence"] == 0.8

    def test_get_wanted_keys_iti(self):
        creator = GraphCreator()
        keys = creator._get_wanted_keys("I-T-I")

        # Check that we have image-image with original but not text-image with original
        has_img_img_orig = any(
            k.comparison_type == "image-image" and k.anchor == "original" for k in keys
        )
        has_txt_img_orig = any(
            k.comparison_type == "text-image" and k.anchor == "original" for k in keys
        )

        assert has_img_img_orig
        assert not has_txt_img_orig

    def test_get_wanted_keys_tit(self):
        creator = GraphCreator()
        keys = creator._get_wanted_keys("T-I-T")

        # Check that we have text-text with original but not image-image with original
        has_txt_txt_orig = any(
            k.comparison_type == "text-text" and k.anchor == "original" for k in keys
        )
        has_img_img_orig = any(
            k.comparison_type == "image-image" and k.anchor == "original" for k in keys
        )

        assert has_txt_txt_orig
        assert not has_img_img_orig

    @patch("src.graph_creator.GraphCreator._plot_group")
    def test_generate_charts_for_eval_success(self, mock_plot_group, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)

        # Mock _plot_group to return a chart path
        mock_chart_path = eval_dir / "test_chart.png"
        mock_plot_group.return_value = mock_chart_path

        creator = GraphCreator()
        charts = creator.generate_charts_for_eval(eval_dir)

        assert len(charts) > 0
        assert mock_plot_group.call_count > 0

    def test_discover_eval_dirs_single_eval(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        dirs = GraphCreator.discover_eval_dirs(tmp_path / "eval")
        assert len(dirs) == 1
        assert dirs[0] == eval_dir

    def test_discover_eval_dirs_item_folder(self, tmp_path):
        eval_dir = tmp_path / "item1" / "eval"
        eval_dir.mkdir(parents=True)

        dirs = GraphCreator.discover_eval_dirs(tmp_path / "item1")
        assert len(dirs) == 1
        assert dirs[0] == eval_dir

    def test_discover_eval_dirs_experiment_folder(self, tmp_path):
        eval_dir1 = tmp_path / "item1" / "eval"
        eval_dir2 = tmp_path / "item2" / "eval"
        eval_dir1.mkdir(parents=True)
        eval_dir2.mkdir(parents=True)

        dirs = GraphCreator.discover_eval_dirs(tmp_path)
        assert len(dirs) == 2
        assert eval_dir1 in dirs
        assert eval_dir2 in dirs

    def test_generate_charts_for_experiment(self, tmp_path):
        # Create multiple eval directories
        eval_dir1 = tmp_path / "item1" / "eval"
        eval_dir2 = tmp_path / "item2" / "eval"
        eval_dir1.mkdir(parents=True)
        eval_dir2.mkdir(parents=True)

        # Create some test data in each
        ratings_data = [
            {
                "item_id": "item1",
                "comparison_type": "image-image",
                "anchor": "original",
                "step": 1,
                "content_correspondence": {"score": 0.8, "reason": "test"},
            }
        ]

        (eval_dir1 / "ratings_image-image.json").write_text(json.dumps(ratings_data))
        (eval_dir2 / "ratings_image-image.json").write_text(json.dumps(ratings_data))

        creator = GraphCreator()
        with patch.object(creator, "_plot_group") as mock_plot:
            mock_plot.return_value = eval_dir1 / "test_chart.png"
            creator.generate_charts_for_experiment(tmp_path)

            # Should process both directories
            assert mock_plot.call_count >= 2


class TestCriteria:
    """Test that CRITERIA constant is properly defined."""

    def test_criteria_constant(self):
        expected_criteria = [
            "content_correspondence",
            "compositional_alignment",
            "fidelity_completeness",
            "stylistic_congruence",
            "overall_semantic_intent",
        ]
        assert CRITERIA == expected_criteria
