# -*- coding: utf-8 -*-
"""
Utility for writing text, images, and metadata inside the experiment folder.
Supports creating sub-managers rooted at results/exp_name/<image_stem>/.
"""

import json
import os
from typing import Any, Dict

import yaml
from PIL import Image


class OutputManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir  # e.g. results/exp_001  OR  results/exp_001/input_0
        os.makedirs(self.root_dir, exist_ok=True)

    def _full(self, fname: str) -> str:
        return os.path.join(self.root_dir, fname)

    def save_text(self, text: str, fname: str) -> str:
        path = self._full(fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def save_image(self, image: Image.Image, fname: str) -> str:
        path = self._full(fname)
        image.save(path)
        return path

    def write_json(self, obj: Dict[str, Any], fname: str = "metadata.json") -> None:
        with open(self._full(fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def save_yaml(self, data: Dict[str, Any], fname: str) -> None:
        with open(self._full(fname), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    def subdir(self, subfolder: str) -> "OutputManager":
        """Return a new OutputManager rooted at <root_dir>/<subfolder>."""
        return OutputManager(os.path.join(self.root_dir, subfolder))
