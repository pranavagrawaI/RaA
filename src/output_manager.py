# -*- coding: utf-8 -*-
"""
Utility for writing text, images, and metadata inside the experiment folder.
Supports creating sub-managers rooted at results/exp_name/<image_stem>/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PIL import Image


class OutputManager:
    """Utility for writing text, images, and metadata inside the experiment folder."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _full(self, fname: str) -> Path:
        return self.root_dir / fname

    def save_text(self, text: str, fname: str) -> Path:
        """Save text to a file."""
        path = self._full(fname)
        path.write_text(text, encoding="utf-8")
        return path

    def save_image(self, image: Image.Image, fname: str) -> Path:
        """Save image to a file."""
        path = self._full(fname)
        image.save(path)
        return path

    def write_json(
        self, obj: Dict[str, Any] | List[Any], fname: str = "metadata.json"
    ) -> None:
        """Save a dictionary or list to a JSON file."""
        with open(self._full(fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def save_yaml(self, data: Dict[str, Any], fname: str) -> None:
        """Save a dictionary to a YAML file."""
        with open(self._full(fname), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    def subdir(self, subfolder: str) -> "OutputManager":
        """Return a new OutputManager rooted at <root_dir>/<subfolder>."""
        return OutputManager(self.root_dir / subfolder)
