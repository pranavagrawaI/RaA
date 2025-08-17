# -*- coding: utf-8 -*-

"""
Tests for the graph_creator module.

This module contains comprehensive tests for the reporting/charting functionality
that generates evaluation score visualizations from ratings JSON files.

The tests cover:
- Key dataclass functionality
- File and directory operations
- JSON data loading and parsing
- Chart generation with matplotlib
- Main CLI function behavior
- Error handling and edge cases

Test Structure:
- TestKey: Tests for the Key dataclass used to group evaluation data
- TestSanitizeFilename: Tests for filename sanitization utility
- TestCollectEvalFiles: Tests for finding ratings JSON files
- TestLoadRecords: Tests for loading and parsing JSON data
- TestExtractItemId: Tests for extracting item identifiers
- TestIterSeries: Tests for filtering and iterating evaluation series
- TestPlotGroup: Tests for matplotlib chart generation (with mocking)
- TestGenerateChartsForEval: Tests for the main chart generation workflow
- TestDiscoverEvalDirs: Tests for finding evaluation directories
- TestMain: Tests for CLI entry point functionality
- TestCriteria: Tests for the evaluation criteria constants
"""

import json
from unittest.mock import patch

from src.graph_creator import (
    Key,
    _sanitize_filename,
    _collect_eval_files,
    _load_records,
    _extract_item_id,
    _iter_series,
    _plot_group,
    generate_charts_for_eval,
    _discover_eval_dirs,
    main,
    CRITERIA,
)


class TestKey:
    """Test the Key dataclass."""

    def test_key_creation_minimal(self):
        key = Key("image-image", "original")
        assert key.comparison_type == "image-image"
        assert key.anchor == "original"
        assert key.direction is None

    def test_key_creation_with_direction(self):
        key = Key("image-text", "same-step", "forward")
        assert key.comparison_type == "image-text"
        assert key.anchor == "same-step"
        assert key.direction == "forward"

    def test_key_creation_text_image(self):
        key = Key("text-image", "previous")
        assert key.comparison_type == "text-image"
        assert key.anchor == "previous"
        assert key.direction is None

    def test_key_equality(self):
        key1 = Key("text-text", "previous")
        key2 = Key("text-text", "previous")
        key3 = Key("text-text", "original")
        assert key1 == key2
        assert key1 != key3


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_normal_name(self):
        assert _sanitize_filename("normal_file.txt") == "normal_file.txt"

    def test_sanitize_special_chars(self):
        assert (
            _sanitize_filename("file name/with\\special*chars")
            == "file-name-with-special-chars"
        )

    def test_sanitize_empty_string(self):
        assert _sanitize_filename("") == ""

    def test_sanitize_only_special_chars(self):
        assert _sanitize_filename("/@#$%") == "-----"


class TestCollectEvalFiles:
    """Test collection of evaluation files."""

    def test_collect_eval_files_empty_dir(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        files = _collect_eval_files(eval_dir)
        assert files == []

    def test_collect_eval_files_with_ratings(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        # Create some ratings files
        (eval_dir / "ratings_001.json").write_text("{}")
        (eval_dir / "ratings_002.json").write_text("{}")
        (eval_dir / "other_file.json").write_text("{}")

        files = _collect_eval_files(eval_dir)
        assert len(files) == 2
        assert all("ratings_" in f.name for f in files)
        assert files == sorted(files)  # Should be sorted


class TestLoadRecords:
    """Test loading records from JSON files."""

    def test_load_records_empty_dir(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        records = _load_records(eval_dir)
        assert records == []

    def test_load_records_valid_json_list(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        test_data = [{"step": 1, "score": 0.8}, {"step": 2, "score": 0.9}]
        (eval_dir / "ratings_001.json").write_text(json.dumps(test_data))

        records = _load_records(eval_dir)
        assert len(records) == 2
        assert records[0]["step"] == 1
        assert records[1]["step"] == 2

    def test_load_records_valid_json_dict(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        test_data = {"step": 1, "score": 0.8}
        (eval_dir / "ratings_001.json").write_text(json.dumps(test_data))

        records = _load_records(eval_dir)
        assert len(records) == 1
        assert records[0]["step"] == 1

    def test_load_records_invalid_json(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        # Create invalid JSON
        (eval_dir / "ratings_001.json").write_text("invalid json")

        with patch("builtins.print") as mock_print:
            records = _load_records(eval_dir)
            assert records == []
            mock_print.assert_called()

    def test_load_records_multiple_files(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        data1 = [{"step": 1, "item": "A"}]
        data2 = [{"step": 2, "item": "B"}, {"step": 3, "item": "C"}]

        (eval_dir / "ratings_001.json").write_text(json.dumps(data1))
        (eval_dir / "ratings_002.json").write_text(json.dumps(data2))

        records = _load_records(eval_dir)
        assert len(records) == 3


class TestExtractItemId:
    """Test item ID extraction."""

    def test_extract_item_id_from_records(self, tmp_path):
        eval_dir = tmp_path / "eval"
        records = [{"item_id": "test_item", "step": 1}]
        item_id = _extract_item_id(eval_dir, records)
        assert item_id == "test_item"

    def test_extract_item_id_from_parent_dir(self, tmp_path):
        item_dir = tmp_path / "my_item"
        eval_dir = item_dir / "eval"
        records = []
        item_id = _extract_item_id(eval_dir, records)
        assert item_id == "my_item"

    def test_extract_item_id_fallback(self, tmp_path):
        eval_dir = tmp_path / "eval"
        records = []
        item_id = _extract_item_id(eval_dir, records)
        # The function falls back to the parent directory name, which is the temp directory name in tests
        assert item_id != ""  # Just ensure it's not empty


class TestIterSeries:
    """Test series iteration for specific keys."""

    def create_sample_record(
        self, comparison_type="image-image", anchor="original", step=1, scores=None
    ):
        """Helper to create sample records."""
        if scores is None:
            scores = {"content_correspondence": 0.8}

        record = {
            "comparison_type": comparison_type,
            "anchor": anchor,
            "step": step,
        }

        # Add criterion scores in the expected format
        for criterion, score in scores.items():
            record[criterion] = {"score": score, "reason": "test"}

        return record

    def test_iter_series_matching_key(self):
        records = [
            self.create_sample_record(
                "image-image", "original", 1, {"content_correspondence": 0.8}
            ),
            self.create_sample_record(
                "image-image", "original", 2, {"content_correspondence": 0.9}
            ),
        ]

        key = Key("image-image", "original")
        series = list(_iter_series(records, key))

        assert len(series) == 2
        assert series[0][0] == 1  # step
        assert series[0][1]["content_correspondence"] == 0.8
        assert series[1][0] == 2
        assert series[1][1]["content_correspondence"] == 0.9

    def test_iter_series_no_matching_key(self):
        records = [
            self.create_sample_record("image-image", "original", 1),
        ]

        key = Key("text-text", "previous")
        series = list(_iter_series(records, key))

        assert len(series) == 0

    def test_iter_series_filters_negative_scores(self):
        records = [
            self.create_sample_record(
                "image-image", "original", 1, {"content_correspondence": -1.0}
            ),
            self.create_sample_record(
                "image-image", "original", 2, {"content_correspondence": 0.8}
            ),
        ]

        key = Key("image-image", "original")
        series = list(_iter_series(records, key))

        # Should only have one valid series point (step 2)
        assert len(series) == 1
        assert series[0][0] == 2

    def test_iter_series_invalid_step(self):
        records = [
            {
                "comparison_type": "image-image",
                "anchor": "original",
                "step": "invalid",
                "content_correspondence": {"score": 0.8, "reason": "test"},
            }
        ]

        key = Key("image-image", "original")
        with patch("builtins.print") as mock_print:
            series = list(_iter_series(records, key))
            assert len(series) == 0
            mock_print.assert_called()

    def test_iter_series_text_image_matching_key(self):
        """Test that text-image comparison type works correctly."""
        records = [
            self.create_sample_record(
                "text-image", "previous", 1, {"content_correspondence": 0.8}
            ),
            self.create_sample_record(
                "text-image", "previous", 2, {"content_correspondence": 0.9}
            ),
        ]

        key = Key("text-image", "previous")
        series = list(_iter_series(records, key))

        assert len(series) == 2
        assert series[0][0] == 1  # step
        assert series[0][1]["content_correspondence"] == 0.8
        assert series[1][0] == 2
        assert series[1][1]["content_correspondence"] == 0.9


class TestPlotGroup:
    """Test plotting functionality."""

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_group_creates_chart(
        self, mock_close, mock_savefig, mock_title, mock_plot, mock_figure, tmp_path
    ):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        key = Key("image-image", "original")
        series = [
            (
                1,
                {
                    "content_correspondence": 0.8,
                    "compositional_alignment": 0.7,
                    "fidelity_completeness": None,
                    "stylistic_congruence": None,
                    "overall_semantic_intent": None,
                },
            ),
            (
                2,
                {
                    "content_correspondence": 0.9,
                    "compositional_alignment": 0.8,
                    "fidelity_completeness": None,
                    "stylistic_congruence": None,
                    "overall_semantic_intent": None,
                },
            ),
        ]

        result = _plot_group("test_item", eval_dir, key, series)

        assert result is not None
        assert result.name == "chart_image-image_original.png"
        # Check that matplotlib functions were called, but don't be strict about call counts
        assert mock_figure.called
        mock_savefig.assert_called_once_with(result)
        mock_close.assert_called_once()

    def test_plot_group_empty_series(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        key = Key("image-image", "original")
        series = []

        result = _plot_group("test_item", eval_dir, key, series)

        assert result is None

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_group_with_direction(
        self, mock_close, mock_savefig, mock_figure, tmp_path
    ):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        key = Key("image-text", "same-step", "forward")
        series = [
            (
                1,
                {
                    "content_correspondence": 0.8,
                    "compositional_alignment": None,
                    "fidelity_completeness": None,
                    "stylistic_congruence": None,
                    "overall_semantic_intent": None,
                },
            )
        ]

        result = _plot_group("test_item", eval_dir, key, series)

        assert result is not None
        assert "forward" in result.name

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_group_text_image(
        self, mock_close, mock_savefig, mock_figure, tmp_path
    ):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        key = Key("text-image", "previous")
        series = [
            (
                1,
                {
                    "content_correspondence": 0.8,
                    "compositional_alignment": 0.7,
                    "fidelity_completeness": None,
                    "stylistic_congruence": None,
                    "overall_semantic_intent": None,
                },
            )
        ]

        result = _plot_group("test_item", eval_dir, key, series)

        assert result is not None
        assert result.name == "chart_text-image_previous.png"


class TestGenerateChartsForEval:
    """Test the main chart generation function."""

    def create_test_eval_dir_with_data(self, tmp_path):
        """Helper to create test evaluation directory with sample data."""
        eval_dir = tmp_path / "item1" / "eval"
        eval_dir.mkdir(parents=True)

        # Create sample ratings data
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

        (eval_dir / "ratings_001.json").write_text(json.dumps(ratings_data))
        return eval_dir

    @patch("src.graph_creator._plot_group")
    def test_generate_charts_for_eval_success(self, mock_plot_group, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)

        # Mock _plot_group to return a chart path
        mock_chart_path = eval_dir / "test_chart.png"
        mock_plot_group.return_value = mock_chart_path

        charts = generate_charts_for_eval(eval_dir)

        assert len(charts) > 0
        assert mock_plot_group.call_count > 0

    @patch("src.graph_creator._plot_group")
    def test_generate_charts_includes_text_image_types(self, mock_plot_group, tmp_path):
        """Test that text-image comparison types are included in chart generation."""
        eval_dir = tmp_path / "item1" / "eval"
        eval_dir.mkdir(parents=True)

        # Create sample data with text-image comparisons
        ratings_data = [
            {
                "item_id": "item1",
                "comparison_type": "text-image",
                "anchor": "previous",
                "step": 2,
                "content_correspondence": {"score": 0.8, "reason": "test"},
            },
            {
                "item_id": "item1",
                "comparison_type": "text-image",
                "anchor": "original",
                "step": 1,
                "content_correspondence": {"score": 0.7, "reason": "test"},
            },
        ]

        (eval_dir / "ratings_text-image.json").write_text(json.dumps(ratings_data))

        # Mock _plot_group to return a chart path
        mock_chart_path = eval_dir / "test_chart.png"
        mock_plot_group.return_value = mock_chart_path

        generate_charts_for_eval(eval_dir)

        # Verify that _plot_group was called with text-image keys
        called_keys = [
            call[0][2] for call in mock_plot_group.call_args_list
        ]  # Third argument is the Key
        text_image_keys = [
            key
            for key in called_keys
            if hasattr(key, "comparison_type") and key.comparison_type == "text-image"
        ]

        assert len(text_image_keys) >= 2  # Should have both original and previous

    def test_generate_charts_for_eval_no_data(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        charts = generate_charts_for_eval(eval_dir)

        assert charts == []

    @patch("src.graph_creator._plot_group")
    def test_generate_charts_creates_index(self, mock_plot_group, tmp_path):
        eval_dir = self.create_test_eval_dir_with_data(tmp_path)

        mock_chart_path = eval_dir / "test_chart.png"
        mock_plot_group.return_value = mock_chart_path

        generate_charts_for_eval(eval_dir)

        index_file = eval_dir / "charts_index.json"
        assert index_file.exists()

        index_data = json.loads(index_file.read_text())
        assert "item_id" in index_data
        assert "charts" in index_data
        assert index_data["item_id"] == "item1"


class TestDiscoverEvalDirs:
    """Test evaluation directory discovery."""

    def test_discover_eval_dirs_direct_eval(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        dirs = _discover_eval_dirs(eval_dir)
        assert len(dirs) == 1
        assert dirs[0] == eval_dir

    def test_discover_eval_dirs_item_with_eval(self, tmp_path):
        item_dir = tmp_path / "item1"
        eval_dir = item_dir / "eval"
        eval_dir.mkdir(parents=True)

        dirs = _discover_eval_dirs(item_dir)
        assert len(dirs) == 1
        assert dirs[0] == eval_dir

    def test_discover_eval_dirs_experiment_with_items(self, tmp_path):
        exp_dir = tmp_path / "exp_001"
        exp_dir.mkdir()

        # Create multiple items with eval subdirs
        for i in range(3):
            item_eval = exp_dir / f"item{i}" / "eval"
            item_eval.mkdir(parents=True)

        dirs = _discover_eval_dirs(exp_dir)
        assert len(dirs) == 3
        assert all(d.name == "eval" for d in dirs)

    def test_discover_eval_dirs_no_eval_folders(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        dirs = _discover_eval_dirs(empty_dir)
        assert dirs == []


class TestMain:
    """Test the main function."""

    def test_main_nonexistent_path(self):
        result = main(["nonexistent_path"])
        assert result == 2

    @patch("src.graph_creator._discover_eval_dirs")
    def test_main_no_eval_dirs(self, mock_discover, tmp_path):
        mock_discover.return_value = []

        test_dir = tmp_path / "test"
        test_dir.mkdir()

        result = main([str(test_dir)])
        assert result == 1

    @patch("src.graph_creator.generate_charts_for_eval")
    @patch("src.graph_creator._discover_eval_dirs")
    def test_main_success(self, mock_discover, mock_generate, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        mock_discover.return_value = [eval_dir]
        mock_generate.return_value = [eval_dir / "chart1.png", eval_dir / "chart2.png"]

        with patch("builtins.print") as mock_print:
            result = main([str(tmp_path)])
            assert result == 0
            mock_print.assert_called()


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
        assert len(CRITERIA) == 5
        assert all(isinstance(c, str) for c in CRITERIA)
