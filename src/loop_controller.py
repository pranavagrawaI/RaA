# -*- coding: utf-8 -*-
"""
Dry-run I→T→I loop that produces placeholder captions & blank images.
Stores outputs in results/exp_name/<image_stem>/
"""

import errno
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict

from benchmark_config import BenchmarkConfig
from output_manager import OutputManager
from prompt_engine import generate_caption, generate_image


class LoopController:
    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.root_om = OutputManager(config.output_dir)  # manager for exp root
        self.meta: Dict[str, Any] = {}

    def run(self) -> None:
        images = sorted(glob(os.path.join(self.cfg.input_dir, "*.[jp][pn]g")))
        if not images:
            raise RuntimeError(f"No .jpg/.png found in {self.cfg.input_dir}")

        for path in images:
            self._process_single(path)

        self.root_om.write_json(self.meta, "metadata.json")

    def _process_single(self, img_path: str) -> None:
        stem = Path(img_path).stem  # e.g. "input_0"
        om = self.root_om.subdir(stem)  # results/exp_name/input_0
        record: Dict[str, str] = {}

        dest_input = os.path.join(om.root_dir, "input.jpg")
        self._link_file(img_path, dest_input)
        record["input"] = "input.jpg"

        current_img_path = dest_input
        for i in range(1, self.cfg.loop.num_iterations + 1):
            caption = generate_caption(current_img_path)
            txt_name = f"text_iter{i}.txt"
            om.save_text(caption, txt_name)
            record[f"iter{i}_text"] = txt_name

            blank = generate_image(caption)
            img_name = f"image_iter{i}.jpg"
            om.save_image(blank, img_name)
            record[f"iter{i}_img"] = img_name

            if not self.cfg.loop.stateless:
                current_img_path = os.path.join(om.root_dir, img_name)

        self.meta[stem] = record

    def _link_file(self, src: str, dst: str) -> None:
        """Create a symlink; fall back to copy if symlink fails."""
        try:
            os.symlink(os.path.abspath(src), dst)
        except (AttributeError, NotImplementedError, OSError) as e:
            # Windows without admin rights often hits EPERM
            print("[WARNING] Symlink failed, falling back to copy:", e)
            if isinstance(e, (OSError,)) and e.errno not in (errno.EEXIST, errno.EPERM):
                raise
            shutil.copy2(src, dst)
