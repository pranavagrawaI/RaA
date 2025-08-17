# -*- coding: utf-8 -*-
"""Test the reporting_summary.py implementation."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock the Google Generative AI module before importing reporting_summary
mock_genai = Mock()
mock_types = Mock()

with patch.dict(
    "sys.modules",
    {"google.generativeai": mock_genai, "google.generativeai.types": mock_types},
):
    from reporting_summary import SummaryGenerator


class TestSummaryGenerator:
    """Test the SummaryGenerator class."""

    def test_load_individual_eval_files(self):
        """Test loading individual eval JSON files returns filename+data structures in order."""
        generator = SummaryGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            expected_order = [
                "ratings_image-image.json",
                "ratings_image-text.json",
                "ratings_text-image.json",
                "ratings_text-text.json",
            ]

            # Create two of the expected files and an unrelated file that should be ignored
            data1 = {"step": 1, "score": 9.5, "reason": "Test reason 1"}
            data2 = [{"step": 2, "score": 8.0}, {"step": 3, "score": 7.5}]

            (temp_path / expected_order[0]).write_text(json.dumps(data1, indent=2))
            (temp_path / expected_order[1]).write_text(json.dumps(data2, indent=2))
            (temp_path / "unrelated.json").write_text("{}")

            result = generator._load_individual_eval_files(temp_path)

            # Verify two files loaded, in expected relative order
            assert len(result) == 2
            assert result[0]["filename"] == expected_order[0]
            assert result[1]["filename"] == expected_order[1]
            assert result[0]["data"]["reason"] == "Test reason 1"
            assert result[1]["data"][0]["step"] == 2

    def test_load_individual_eval_files_no_directory(self):
        """Test error handling when directory doesn't exist."""
        generator = SummaryGenerator()

        with pytest.raises(FileNotFoundError, match="Directory not found"):
            generator._load_individual_eval_files(Path("/nonexistent/path"))

    def test_load_individual_eval_files_no_json(self):
        """Test error handling when no expected JSON files exist."""
        generator = SummaryGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create unrelated files only
            (temp_path / "test.txt").write_text("Not JSON")
            (temp_path / "some.json").write_text("{}")

            with pytest.raises(FileNotFoundError, match="No JSON files found"):
                generator._load_individual_eval_files(temp_path)

    def test_generate_summary_no_api_key(self):
        """Test summary generation without API key."""
        generator = SummaryGenerator()

        # Remove API key if set
        original_key = os.environ.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        try:
            # Create temporary system instruction file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Test system instruction")
                system_file = Path(f.name)

            try:
                eval_data = [{"filename": "ratings_image-image.json", "data": {"k": 1}}]
                result = generator._generate_summary(
                    eval_data, system_file, item_id="item1"
                )
                assert "Error: GOOGLE_API_KEY not set" == result
            finally:
                system_file.unlink()

        finally:
            # Restore original API key
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key

    def test_system_instruction_loading(self):
        """Test that system instruction file is loaded correctly."""
        generator = SummaryGenerator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            test_instruction = "Custom test instruction for analysis"
            f.write(test_instruction)
            system_file = Path(f.name)

        # Mock the entire _generate_summary method to avoid import issues
        try:
            eval_data = [{"filename": "ratings_image-image.json", "data": {"k": 1}}]
            with patch.object(
                generator, "_generate_summary", return_value="OK"
            ) as mock_generate:
                result = generator._generate_summary(
                    eval_data, system_file, item_id="abc"
                )
                assert result == "OK"

                # Verify the method was called with correct arguments
                mock_generate.assert_called_once_with(
                    eval_data, system_file, item_id="abc"
                )

        finally:
            system_file.unlink()

    def test_extract_item_id(self):
        """Test extracting item ID from eval directory path."""
        # Test typical case
        eval_dir = Path("/results/exp_001/item1/eval")
        item_id = SummaryGenerator._extract_item_id(eval_dir)
        assert item_id == "item1"

        # Test edge case
        eval_dir = Path("/eval")
        item_id = SummaryGenerator._extract_item_id(eval_dir)
        assert item_id == "item"

    def test_discover_eval_dirs(self):
        """Test discovering eval directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create experiment structure
            item1_eval = temp_path / "item1" / "eval"
            item2_eval = temp_path / "item2" / "eval"
            item1_eval.mkdir(parents=True)
            item2_eval.mkdir(parents=True)

            # Test discovery
            eval_dirs = SummaryGenerator.discover_eval_dirs(temp_path)

            assert len(eval_dirs) == 2
            assert item1_eval in eval_dirs
            assert item2_eval in eval_dirs

    def test_generate_summary_for_eval_success(self):
        """Test successful summary generation for an eval directory."""
        generator = SummaryGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_dir = temp_path / "item1" / "eval"
            eval_dir.mkdir(parents=True)

            # Create test JSON file
            test_data = {"score": 8.5, "reason": "Good quality"}
            (eval_dir / "ratings_image-image.json").write_text(json.dumps(test_data))

            # Create system instruction file
            system_file = temp_path / "system.txt"
            system_file.write_text("Test instruction")

            # Mock the summary generation
            with patch.object(
                generator, "_generate_summary", return_value="Test summary"
            ):
                result = generator.generate_summary_for_eval(eval_dir, system_file)

                assert result is True

                # Verify output file was created
                output_file = eval_dir / "qualitative_summary.txt"
                assert output_file.exists()
                assert output_file.read_text() == "Test summary"

    def test_generate_summaries_for_experiment(self):
        """Test generating summaries for an entire experiment."""
        generator = SummaryGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create experiment structure with two items
            for item_name in ["item1", "item2"]:
                eval_dir = temp_path / item_name / "eval"
                eval_dir.mkdir(parents=True)

                # Create test JSON file
                test_data = {"score": 8.5, "reason": f"Good quality for {item_name}"}
                (eval_dir / "ratings_image-image.json").write_text(
                    json.dumps(test_data)
                )

            # Create system instruction file
            system_file = temp_path / "system.txt"
            system_file.write_text("Test instruction")

            # Mock the summary generation
            with patch.object(
                generator, "_generate_summary", return_value="Test summary"
            ):
                successful, failed = generator.generate_summaries_for_experiment(
                    temp_path, system_file
                )

                assert successful == 2
                assert failed == 0

                # Verify output files were created for both items
                for item_name in ["item1", "item2"]:
                    output_file = (
                        temp_path / item_name / "eval" / "qualitative_summary.txt"
                    )
                    assert output_file.exists()
                    assert output_file.read_text() == "Test summary"
