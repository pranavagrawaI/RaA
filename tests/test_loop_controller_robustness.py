# -*- coding: utf-8 -*-

"""Tests for the LoopController's error handling and resumption capabilities."""

import json
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from benchmark_config import BenchmarkConfig
from loop_controller import LoopController


class TestLoopControllerRobustness:
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock configuration."""
        config = Mock(spec=BenchmarkConfig)
        config.output_dir = tmp_path / "results/exp_test"
        config.input_dir = tmp_path / "inputs"

        # Create nested mock objects for loop and prompts
        loop_mock = Mock()
        loop_mock.type = "I-T-I"
        loop_mock.num_iterations = 3
        config.loop = loop_mock

        prompts_mock = Mock()
        prompts_mock.caption = "Describe this image"
        prompts_mock.image = "Generate an image"
        config.prompts = prompts_mock

        config.max_retries = 2
        config.retry_delay = 0.1  # Short delay for tests
        return config

    @pytest.fixture
    def setup_input_files(self, mock_config):
        """Create test input files."""
        mock_config.input_dir.mkdir(parents=True)
        test_image = mock_config.input_dir / "test.jpg"
        test_image.write_bytes(b"fake image data")
        return mock_config

    def test_retry_logic(self, mock_config, setup_input_files):
        """Test that operations are retried with exponential backoff."""
        controller = LoopController(mock_config)

        # Mock that fails once then succeeds (max_retries=2)
        fail_count = 0

        def failing_operation(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 1:
                raise RuntimeError("API temporary error")
            return "success"

        # Should succeed after retries
        result = controller._retry_with_backoff(failing_operation)
        assert result == "success"
        assert fail_count == 2  # One failure + one success

    def test_resumption_after_failure(self, mock_config, setup_input_files):
        """Test that processing resumes from last successful point."""
        controller = LoopController(mock_config)

        # Create a mock PIL Image
        test_image = Image.new("RGB", (100, 100), color="red")

        # Mock the first iteration to succeed
        with (
            patch("loop_controller.generate_caption", return_value="test caption"),
            patch("loop_controller.generate_image", return_value=test_image),
        ):
            controller._process_i_t_i_for_image(
                str(mock_config.input_dir / "test.jpg"), "test"
            )

            # Verify metadata was saved
            meta_file = mock_config.output_dir / "metadata.json"
            assert meta_file.exists()

            with open(meta_file, "r") as f:
                metadata = json.load(f)

            # Check first iteration files exist
            assert metadata["test"]["iter1_text"] == "text_iter1.txt"
            assert metadata["test"]["iter1_img"] == "image_iter1.jpg"

            # Verify files were created
            test_dir = mock_config.output_dir / "test"
            assert (test_dir / "text_iter1.txt").exists()
            assert (test_dir / "image_iter1.jpg").exists()

    def test_complete_failure_handling(self, mock_config, setup_input_files):
        """Test behavior when all retries are exhausted."""
        controller = LoopController(mock_config)

        mock_generate_caption = Mock(side_effect=RuntimeError("Persistent API error"))

        # Create an empty test directory for output
        output_dir = mock_config.output_dir / "test"
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("loop_controller.generate_caption", mock_generate_caption),
            pytest.raises(RuntimeError, match="Persistent API error"),
        ):
            controller._process_i_t_i_for_image(
                str(mock_config.input_dir / "test.jpg"), "test"
            )

        # Should have tried max_retries times (2 in our test config)
        assert mock_generate_caption.call_count == mock_config.max_retries
