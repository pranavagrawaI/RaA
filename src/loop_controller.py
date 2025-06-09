# -*- coding: utf-8 -*-
"""
Dry-run recursive loop controller (I-T-I, T-I-T, etc.) that produces placeholders.
Stores outputs in results/exp_name/<identifier>/
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

        # Write global metadata
        self.rootOM.write_json(self.meta, "metadata.json")

    def _run_i_t_i(self) -> None:
        # Image → Text → Image loop for each input image
        images = sorted(glob(os.path.join(self.cfg.input_dir, "*.[jp][pn]g")))
        if not images:
            raise RuntimeError(f"No .jpg/.png found in {self.cfg.input_dir}")

        for path in images:
            stem = Path(path).stem  # e.g. "input_0"
            self._process_i_t_i_for_image(path, stem)

    def _process_i_t_i_for_image(self, img_path: str, stem: str) -> None:
        # Create per-image subdirectory
        om = self.rootOM.subdir(stem)
        record: Dict[str, str] = {}

        # Copy or symlink original image
        dest_input = os.path.join(om.root_dir, "input.jpg")
        self._link_file(img_path, dest_input)
        record["input"] = "input.jpg"

        current_img_path = dest_input
        for i in range(1, self.cfg.loop.num_iterations + 1):
            # I → T: image to placeholder text
            caption = generate_caption(current_img_path, prompt="Describe this image.")
            txt_name = f"text_iter{i}.txt"
            om.save_text(caption, txt_name)
            record[f"iter{i}_text"] = txt_name

            # T → I: placeholder text to blank image
            blank = generate_image(caption)
            img_name = f"image_iter{i}.jpg"
            om.save_image(blank, img_name)
            record[f"iter{i}_img"] = img_name

            # Always update current_img_path to the newly generated image for the next iteration
            current_img_path = os.path.join(om.root_dir, img_name)

        self.meta[stem] = record

    def _run_t_i_t(self) -> None:
        # Text → Image → Text loop for each input text file
        texts = sorted(glob(os.path.join(self.cfg.input_dir, "*.txt")))
        if not texts:
            raise RuntimeError(f"No .txt found in {self.cfg.input_dir}")

        for path in texts:
            stem = Path(path).stem  # e.g. "prompt_0"
            self._process_t_i_t_for_text(path, stem)

    def _process_t_i_t_for_text(self, txt_path: str, stem: str) -> None:
        # Create per-text subdirectory
        om = self.rootOM.subdir(stem)
        record: Dict[str, str] = {}

        # Copy or symlink original text file
        dest_input = os.path.join(om.root_dir, "input.txt")
        self._link_file(txt_path, dest_input)
        record["input"] = "input.txt"

        current_text_path = dest_input
        for i in range(1, self.cfg.loop.num_iterations + 1):
            # T → I: read text to placeholder image
            with open(current_text_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            blank = generate_image(text_content)
            img_name = f"image_iter{i}.jpg"
            om.save_image(blank, img_name)
            record[f"iter{i}_img"] = img_name

            # I → T: placeholder image back to placeholder text
            img_path = os.path.join(om.root_dir, img_name)
            caption = generate_caption(
                img_path, prompt="This is a placeholder caption."
            )
            txt_name = f"text_iter{i}.txt"
            om.save_text(caption, txt_name)
            record[f"iter{i}_text"] = txt_name

            # Always update current_text_path to the newly generated text for the next iteration
            current_text_path = os.path.join(om.root_dir, txt_name)

        self.meta[stem] = record

    def _link_file(self, src: str, dst: str) -> None:
        """Create a symlink; fall back to copy if symlink fails."""
        # First, ensure the destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Remove the destination file if it exists
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except (PermissionError, OSError):
                # If we can't remove it, wait a bit and try again
                time.sleep(1)
                try:
                    os.remove(dst)
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not remove existing file {dst}: {e}")
                    return

        # Try to create symlink first
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except (AttributeError, NotImplementedError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in (errno.EEXIST, errno.EPERM):
                raise

        # If symlink fails, try to copy with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst)
                return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait a second before retrying
                    continue
                print(f"Warning: Could not copy file {src} to {dst}: {e}")
                return
