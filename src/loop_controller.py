# -*- coding: utf-8 -*-
"""
Recursive loop controller.
Stores outputs in results/<exp_name>/<identifier>/
"""

import os
import shutil
import errno
import time
from glob import glob
from pathlib import Path
from typing import Dict, Any

from benchmark_config import BenchmarkConfig
from output_manager import OutputManager
from prompt_engine import generate_caption, generate_image


class LoopController:
    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.rootOM = OutputManager(config.output_dir)
        self.meta: Dict[str, Any] = {}

    def run(self) -> None:
        loop_type = self.cfg.loop.type.upper()
        if loop_type == "I-T-I":
            self._run_i_t_i()
        elif loop_type == "T-I-T":
            self._run_t_i_t()
        else:
            raise ValueError(f"Unsupported loop type: {loop_type}")

        self.rootOM.write_json(self.meta, "metadata.json")

    def _run_i_t_i(self) -> None:
        images = sorted(Path(self.cfg.input_dir).glob("*.[jp][pn]g"))
        if not images:
            raise RuntimeError(f"No .jpg/.png found in {self.cfg.input_dir}")

        for path in images:
            stem = Path(path).stem
            self._process_i_t_i_for_image(str(path), stem)

    def _process_i_t_i_for_image(self, img_path: str, stem: str) -> None:
        om = self.rootOM.subdir(stem)
        record: Dict[str, str] = {}

        dest_input = om.root_dir / "input.jpg"
        self._link_file(img_path, str(dest_input))
        record["input"] = "input.jpg"

        current_img_path = str(dest_input)
        for i in range(1, self.cfg.loop.num_iterations + 1):
            caption = generate_caption(
                current_img_path, prompt=self.cfg.prompts.caption
            )
            txt_name = f"text_iter{i}.txt"
            om.save_text(caption, txt_name)
            record[f"iter{i}_text"] = txt_name

            generated_img = generate_image(self.cfg.prompts.image, caption)
            img_name = f"image_iter{i}.jpg"
            om.save_image(generated_img, img_name)
            record[f"iter{i}_img"] = img_name

            current_img_path = str(om.root_dir / img_name)

        self.meta[stem] = record

    def _run_t_i_t(self) -> None:
        texts = sorted(Path(self.cfg.input_dir).glob("*.txt"))
        if not texts:
            raise RuntimeError(f"No .txt found in {self.cfg.input_dir}")

        for path in texts:
            stem = Path(path).stem
            self._process_t_i_t_for_text(str(path), stem)

    def _process_t_i_t_for_text(self, txt_path: str, stem: str) -> None:
        om = self.rootOM.subdir(stem)
        record: Dict[str, str] = {}

        dest_input = om.root_dir / "input.txt"
        self._link_file(txt_path, str(dest_input))
        record["input"] = "input.txt"

        current_text_path = str(dest_input)
        for i in range(1, self.cfg.loop.num_iterations + 1):
            with open(current_text_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            generated_img = generate_image(self.cfg.prompts.image, text_content)
            img_name = f"image_iter{i}.jpg"
            om.save_image(generated_img, img_name)
            record[f"iter{i}_img"] = img_name

            img_path = str(om.root_dir / img_name)
            caption = generate_caption(img_path, prompt=self.cfg.prompts.caption)
            txt_name = f"text_iter{i}.txt"
            om.save_text(caption, txt_name)
            record[f"iter{i}_text"] = txt_name

            current_text_path = str(om.root_dir / txt_name)

        self.meta[stem] = record

    def _link_file(self, src: str, dst: str) -> None:
        """Create a symlink; fall back to copy if symlink fails."""
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            try:
                dst_path.unlink()
            except (PermissionError, OSError):
                time.sleep(1)
                try:
                    dst_path.unlink()
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not remove existing file {dst}: {e}")
                    return

        try:
            os.symlink(os.path.abspath(src), dst_path)
            return
        except (AttributeError, NotImplementedError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in (errno.EEXIST, errno.EPERM):
                raise

        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst_path)
                return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"Warning: Could not copy file {src} to {dst}: {e}")
                return
