# -*- coding: utf-8 -*-
"""
Recursive loop controller.
Stores outputs in results/<exp_name>/<identifier>/
"""

import errno
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from benchmark_config import BenchmarkConfig
from output_manager import OutputManager
from prompt_engine import generate_caption, generate_image

T = TypeVar("T")


class LoopController:
    """Controller for running recursive loops in a benchmark experiment."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.root_om = OutputManager(config.output_dir)
        self.meta: Dict[str, Any] = {}
        # Default retry settings
        self.max_retries = getattr(config, "max_retries", 3)
        self.retry_delay = getattr(config, "retry_delay", 5)

    def _save_progress(self, stem: str, record: Dict[str, str]) -> None:
        """Save current progress to metadata file."""
        self.meta[stem] = record
        self.root_om.write_json(self.meta, "metadata.json")

    def _load_progress(self, stem: str) -> Dict[str, str]:
        """Load existing progress from metadata file."""
        try:
            with open(
                self.root_om.root_dir / "metadata.json", "r", encoding="utf-8"
            ) as f:
                meta = json.load(f)
            return meta.get(stem, {})
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            return {}

    def _retry_with_backoff(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute operation with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (OSError, ValueError, RuntimeError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logging.warning(
                        "Operation failed with error: %s. Retrying in %d seconds (attempt %d/%d)",
                        str(e),
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue

        if last_exception is not None:
            logging.error(
                "Operation failed after %d attempts. Last error: %s",
                self.max_retries,
                str(last_exception),
            )
            raise last_exception
        raise RuntimeError("Operation failed with no exception captured")

    def run(self) -> None:
        """Run the loop controller."""
        loop_type = self.cfg.loop.type.upper()
        if loop_type == "I-T-I":
            self._run_i_t_i()
        elif loop_type == "T-I-T":
            self._run_t_i_t()
        else:
            raise ValueError(f"Unsupported loop type: {loop_type}")

        self.root_om.write_json(self.meta, "metadata.json")

    def _run_i_t_i(self) -> None:
        """Run the Image-Text-Image loop."""
        images = sorted(Path(self.cfg.input_dir).glob("*.[jp][pn]g"))
        if not images:
            raise RuntimeError(f"No .jpg/.png found in {self.cfg.input_dir}")

        for path in images:
            stem = Path(path).stem
            self._process_i_t_i_for_image(str(path), stem)

    def _process_i_t_i_for_image(self, img_path: str, stem: str) -> None:
        om = self.root_om.subdir(stem)
        record = self._load_progress(stem)

        # If no progress exists, initialize with input image
        if not record:
            dest_input = om.root_dir / "input.jpg"
            self._link_file(img_path, str(dest_input))
            record["input"] = "input.jpg"
            self._save_progress(stem, record)

        # Determine the starting point
        last_img = record.get(f"iter{self.cfg.loop.num_iterations}_img")
        if last_img:  # All iterations completed
            return

        # Find last successful iteration
        current_iter = 1
        while f"iter{current_iter}_img" in record:
            current_iter += 1

        # Get path of last successful image
        if current_iter == 1:
            current_img_path = str(om.root_dir / record["input"])
        else:
            current_img_path = str(om.root_dir / record[f"iter{current_iter - 1}_img"])

        # Continue from last successful iteration
        for i in range(current_iter, self.cfg.loop.num_iterations + 1):
            try:
                # Generate caption with retry
                caption = self._retry_with_backoff(
                    generate_caption, current_img_path, prompt=self.cfg.prompts.caption
                )
                txt_name = f"text_iter{i}.txt"
                om.save_text(caption, txt_name)
                record[f"iter{i}_text"] = txt_name
                self._save_progress(stem, record)

                # Generate image with retry
                generated_img = self._retry_with_backoff(
                    generate_image, self.cfg.prompts.image, caption
                )
                img_name = f"image_iter{i}.jpg"
                om.save_image(generated_img, img_name)
                record[f"iter{i}_img"] = img_name
                self._save_progress(stem, record)

                current_img_path = str(om.root_dir / img_name)

            except Exception as e:
                logging.error(
                    "Failed to complete iteration %d for %s: %s", i, stem, str(e)
                )
                raise  # Re-raise after logging

    def _run_t_i_t(self) -> None:
        """Run the Text-Image-Text loop."""
        texts = sorted(Path(self.cfg.input_dir).glob("*.txt"))
        if not texts:
            raise RuntimeError(f"No .txt found in {self.cfg.input_dir}")

        for path in texts:
            stem = Path(path).stem
            self._process_t_i_t_for_text(str(path), stem)

    def _process_t_i_t_for_text(self, txt_path: str, stem: str) -> None:
        om = self.root_om.subdir(stem)
        record = self._load_progress(stem)

        # If no progress exists, initialize with input text
        if not record:
            dest_input = om.root_dir / "input.txt"
            self._link_file(txt_path, str(dest_input))
            record["input"] = "input.txt"
            self._save_progress(stem, record)

        # Determine the starting point
        last_text = record.get(f"iter{self.cfg.loop.num_iterations}_text")
        if last_text:  # All iterations completed
            return

        # Find last successful iteration
        current_iter = 1
        while f"iter{current_iter}_text" in record:
            current_iter += 1

        # Get path of last successful text
        if current_iter == 1:
            current_text_path = str(om.root_dir / record["input"])
        else:
            current_text_path = str(
                om.root_dir / record[f"iter{current_iter - 1}_text"]
            )

        # Continue from last successful iteration
        for i in range(current_iter, self.cfg.loop.num_iterations + 1):
            try:
                with open(current_text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

                # Generate image with retry
                generated_img = self._retry_with_backoff(
                    generate_image, self.cfg.prompts.image, text_content
                )
                img_name = f"image_iter{i}.jpg"
                om.save_image(generated_img, img_name)
                record[f"iter{i}_img"] = img_name
                self._save_progress(stem, record)

                # Generate caption with retry
                img_path = str(om.root_dir / img_name)
                caption = self._retry_with_backoff(
                    generate_caption, img_path, prompt=self.cfg.prompts.caption
                )
                txt_name = f"text_iter{i}.txt"
                om.save_text(caption, txt_name)
                record[f"iter{i}_text"] = txt_name
                self._save_progress(stem, record)

                current_text_path = str(om.root_dir / txt_name)

            except Exception as e:
                logging.error(
                    "Failed to complete iteration %d for %s: %s", i, stem, str(e)
                )
                raise  # Re-raise after logging

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

        max_retries = 2
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst)
                return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logging.warning(
                        "Retrying copy %s to %s after error: %s (attempt %d/%d)",
                        src,
                        dst,
                        e,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                logging.warning("Could not copy file %s to %s: %s", src, dst, e)
                return
