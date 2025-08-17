# -*- coding: utf-8 -*-
"""Test the reporting_summary.py implementation."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from reporting_summary import generate_summary, load_individual_eval_files

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



def test_load_individual_eval_files():
    """Test loading individual eval JSON files returns filename+data structures in order."""
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

        result = load_individual_eval_files(temp_path)

        # Verify two files loaded, in expected relative order
        assert len(result) == 2
        assert result[0]["filename"] == expected_order[0]
        assert result[1]["filename"] == expected_order[1]
        assert result[0]["data"]["reason"] == "Test reason 1"
        assert result[1]["data"][0]["step"] == 2


def test_load_individual_eval_files_no_directory():
    """Test error handling when directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        load_individual_eval_files(Path("/nonexistent/path"))


def test_load_individual_eval_files_no_json():
    """Test error handling when no expected JSON files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create unrelated files only
        (temp_path / "test.txt").write_text("Not JSON")
        (temp_path / "some.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="No JSON files found"):
            load_individual_eval_files(temp_path)


def test_generate_summary_no_api_key():
    """Test summary generation without API key."""
    # Remove API key if set
    original_key = os.environ.get("GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]

    try:
        # Create temporary system instruction file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test system instruction")
            system_file = Path(f.name)

        try:
            eval_data = [{"filename": "ratings_image-image.json", "data": {"k": 1}}]
            result = generate_summary(eval_data, system_file, item_id="item1")
            assert "Error: GOOGLE_API_KEY not set" == result
        finally:
            system_file.unlink()

    finally:
        # Restore original API key
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key


def test_system_instruction_loading(monkeypatch):
    """Test that system instruction file is loaded correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_instruction = "Custom test instruction for analysis"
        f.write(test_instruction)
        system_file = Path(f.name)

    # Stub the genai client to avoid network and validate system instruction is passed
    class FakeModels:
        def generate_content(self, model, contents, config):
            # Ensure our system instruction file was read and passed through
            assert getattr(config, "system_instruction", None) == test_instruction

            class Resp:
                text = "OK"

            return Resp()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        models = FakeModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key_for_test")
    monkeypatch.setattr("reporting_summary.genai.Client", FakeClient)

    try:
        eval_data = [{"filename": "ratings_image-image.json", "data": {"k": 1}}]
        result = generate_summary(eval_data, system_file, item_id="abc")
        assert result == "OK"
    finally:
        system_file.unlink()
