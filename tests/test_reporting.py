import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.reporting import (
    discover_ratings_files,
    load_and_normalize_data,
    calculate_aggregates,
    generate_vega_specs,
    generate_static_charts,
)


class TestReporting(unittest.TestCase):
    def setUp(self):
        """Set up a synthetic dataset for testing."""
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.eval_dir = self.test_dir / "eval"
        self.eval_dir.mkdir(exist_ok=True)
        self.ratings_file = self.eval_dir / "ratings.json"

        self.synthetic_data = [
            {
                "item_id": "test_01",
                "step": 1,
                "comparison_type": "image-image",
                "anchor": "original",
                "overall": {"score": 8, "reason": "Good start"},
                "style_consistency": {"score": 7},
            },
            {
                "item_id": "test_01",
                "step": 2,
                "comparison_type": "image-image",
                "anchor": "original",
                "overall": {"score": 9, "reason": "Improved"},
                "style_consistency": {"score": 6, "reason": "Slightly off"},
            },
            {
                "item_id": "test_01",
                "step": 1,
                "comparison_type": "image-text",
                "anchor": "previous",
                "overall": {"score": -1, "reason": "Not applicable"},
                "fidelity_completeness": {"score": 5},
            },
        ]
        with self.ratings_file.open("w") as f:
            json.dump(self.synthetic_data, f)

    def tearDown(self):
        """Clean up test data."""
        import shutil

        shutil.rmtree(self.test_dir)

    def test_discover_ratings_files(self):
        """Test discovery of ratings.json files."""
        found_files = discover_ratings_files(self.test_dir)
        self.assertEqual(len(found_files), 1)
        self.assertEqual(found_files[0], self.ratings_file)

        found_files = discover_ratings_files(self.ratings_file)
        self.assertEqual(len(found_files), 1)
        self.assertEqual(found_files[0], self.ratings_file)

    def test_load_and_normalize_data(self):
        """Test data loading and metric canonicalization."""
        df, _ = load_and_normalize_data(self.ratings_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("valid", df.columns)
        self.assertEqual(df["valid"].sum(), 5)
        self.assertEqual(df[df["metric"] == "style_consistency"].shape[0], 2)
        self.assertEqual(df.shape[0], 6)

    def test_calculate_aggregates(self):
        """Test correctness of aggregation calculations."""
        df, _ = load_and_normalize_data(self.ratings_file)
        aggregates = calculate_aggregates(df)

        self.assertAlmostEqual(aggregates["summary_cards"]["coverage"], 5 / 6)

        deltas_df = aggregates["deltas"]
        self.assertIn("delta", deltas_df.columns)
        self.assertEqual(
            deltas_df[(deltas_df["metric"] == "overall") & (deltas_df["step"] == 2)][
                "delta"
            ].iloc[0],
            1.0,
        )

        stability_df = aggregates["stability"]
        self.assertIn("mean", stability_df.columns)
        self.assertIn("std", stability_df.columns)

    def test_generate_vega_specs(self):
        """Test that Vega-Lite specs are generated without errors."""
        df, _ = load_and_normalize_data(self.ratings_file)
        aggregates = calculate_aggregates(df)
        specs = generate_vega_specs(df, aggregates)
        self.assertIn("trends", specs)
        self.assertIn("$schema", specs["trends"])

    @patch("matplotlib.pyplot.show")
    def test_generate_static_charts(self, mock_show):
        """Test that static charts are generated without errors."""
        df, _ = load_and_normalize_data(self.ratings_file)
        aggregates = calculate_aggregates(df)
        charts = generate_static_charts(df, aggregates, "light")
        self.assertIn("trends", charts)
        self.assertTrue(charts["trends"].startswith("iVBOR"))


if __name__ == "__main__":
    unittest.main()
