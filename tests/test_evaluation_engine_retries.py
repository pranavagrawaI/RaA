from unittest.mock import Mock, patch

import pytest

from src.benchmark_config import BenchmarkConfig
from src.evaluation_engine import DEFAULT_RATING, EvaluationEngine


@pytest.fixture
def engine():
    # Create a mock for the loop configuration
    mock_loop = Mock()
    mock_loop.type = "I-T-I"

    # Create the main config mock and set up the loop attribute
    mock_config = Mock(spec=BenchmarkConfig)
    mock_config.loop = mock_loop

    # Create a properly structured client mock
    mock_client = Mock()
    mock_client.models = Mock()
    mock_client.models.generate_content = Mock()

    return EvaluationEngine("dummy_root", mock_config, client=mock_client)


def test_successful_retry_after_failure(engine):
    """Test that the rater succeeds after initial failures."""
    mock_response = Mock()
    mock_response.parsed = {"some": "data"}

    # First two calls raise exception, third one succeeds
    engine.client.models.generate_content.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        mock_response,
    ]

    # Mock _prepare_contents to prevent file system access
    with (
        patch.object(engine, "_prepare_contents", return_value=["mock_content"]),
        patch("time.sleep") as mock_sleep,
    ):
        rating = engine._run_rater("image-image", "test_a.jpg", "test_b.jpg")

    # Verify we made exactly 3 attempts
    assert engine.client.models.generate_content.call_count == 3
    # Verify we got actual data instead of DEFAULT_RATING
    assert rating == {"some": "data"}
    # Verify sleep was called with exponential backoff
    mock_sleep.assert_any_call(1)  # First retry
    mock_sleep.assert_any_call(2)  # Second retry


def test_exceeds_max_retries(engine):
    """Test that the rater returns DEFAULT_RATING after max retries."""
    # All attempts raise exception
    engine.client.models.generate_content.side_effect = Exception("API Error")

    # Mock _prepare_contents to prevent file system access
    with (
        patch.object(engine, "_prepare_contents", return_value=["mock_content"]),
        patch("time.sleep"),
    ):
        rating = engine._run_rater("image-image", "test_a.jpg", "test_b.jpg")

    # Verify we made exactly 3 attempts
    assert engine.client.models.generate_content.call_count == 3
    # Verify we got DEFAULT_RATING after all retries failed
    assert rating == DEFAULT_RATING


def test_invalid_response_retry(engine):
    """Test that invalid responses trigger retries."""
    mock_response1 = Mock()
    mock_response1.parsed = None  # Invalid response

    mock_response2 = Mock()
    mock_response2.parsed = {"valid": "data"}  # Valid response

    engine.client.models.generate_content.side_effect = [mock_response1, mock_response2]

    with patch("time.sleep"):  # Don't actually sleep in tests
        rating = engine._run_rater("text-text", "test_a.txt", "test_b.txt")

    # Verify we made exactly 2 attempts (first failed, second succeeded)
    assert engine.client.models.generate_content.call_count == 2
    # Verify we got the valid data from the second attempt
    assert rating == {"valid": "data"}


def test_backoff_timing(engine):
    """Test that the backoff timing is correct between retries."""
    current_time = 0

    def mock_sleep(seconds):
        nonlocal current_time
        current_time += seconds

    # Make all attempts fail
    engine.client.models.generate_content.side_effect = Exception("API Error")

    with patch("time.sleep", side_effect=mock_sleep):
        engine._run_rater("image-image", "test_a.jpg", "test_b.jpg")

    # Verify total time waited matches expected backoff (1 + 2 = 3 seconds)
    assert current_time == 3
