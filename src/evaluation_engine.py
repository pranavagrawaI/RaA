# -*- coding: utf-8 -*-
"""Evaluation Engine for RaA.

This module loads metadata produced by :class:`LoopController`, performs
pairwise comparisons, obtains ratings from either a human or Gemini LLM
backend, and persists the results under each item's ``eval`` folder.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from benchmark_config import BenchmarkConfig
from output_manager import OutputManager


class _Criterion(BaseModel):
    score: float
    reason: str


class _RatingModel(BaseModel):
    content_correspondence: _Criterion
    compositional_alignment: _Criterion
    fidelity_completeness: _Criterion
    stylistic_congruence: _Criterion
    overall_semantic_intent: _Criterion


# The system instruction is stored in the prompts directory and loaded at runtime.
# See prompts/system_instruction_eval.txt

MODEL_NAME = "gemini-2.5-flash-lite"

Rating = Dict[str, Dict[str, Any]]

DEFAULT_RATING = {
    "content_correspondence": {"score": -1.0, "reason": "Rating failed"},
    "compositional_alignment": {"score": -1.0, "reason": "Rating failed"},
    "fidelity_completeness": {"score": -1.0, "reason": "Rating failed"},
    "stylistic_congruence": {"score": -1.0, "reason": "Rating failed"},
    "overall_semantic_intent": {"score": -1.0, "reason": "Rating failed"},
}


class EvaluationEngine:
    """Evaluate semantic drift across loop iterations."""

    def __init__(
        self,
        exp_root: str,
        config: BenchmarkConfig,
        client: genai.Client | None = None,
    ) -> None:
        self.exp_root = Path(exp_root)
        self.loop_type = config.loop.type.upper() if config else ""
        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key) if api_key else None
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompts = {}
        for prompt_file in prompts_dir.glob("*.txt"):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts[prompt_file.stem] = f.read()
        return prompts

    def run(self) -> None:
        """Run the evaluation process."""
        meta_path = self.exp_root / "metadata.json"
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        for item_id, record in meta.items():
            self._eval_single_item(item_id, record)

    def _eval_single_item(self, item_id: str, rec: Dict[str, str]) -> None:
        om = OutputManager(self.exp_root / item_id / "eval")
        # separate rating lists per comparison type
        img_img_ratings: List[Dict[str, Any]] = []
        txt_txt_ratings: List[Dict[str, Any]] = []
        img_txt_ratings: List[Dict[str, Any]] = []
        txt_img_ratings: List[Dict[str, Any]] = []

        def _path(rel: str) -> str:
            return str(self.exp_root / item_id / rel)

        iters = [
            k.split("_")[0] for k in rec if k.startswith("iter") and k.endswith("_img")
        ]
        iters = sorted({int(idx.replace("iter", "")) for idx in iters})

        start_with_image = "input.jpg" in rec or self.loop_type == "I-T-I"
        base_img = (
            _path("input.jpg") if start_with_image else _path(rec.get("iter1_img", ""))
        )
        base_txt = (
            _path("input.txt")
            if not start_with_image
            else _path(rec.get("iter1_text", ""))
        )

        for i in iters:
            curr_img = _path(rec[f"iter{i}_img"])
            curr_txt = _path(rec[f"iter{i}_text"])

            # Compare with original (base) content
            if self.loop_type == "I-T-I":
                # For I-T-I: Always compare current image with original
                img_img_ratings += self._compare_images(
                    item_id, i, curr_img, base_img, "original"
                )
            elif self.loop_type == "T-I-T":
                # For T-I-T: Always compare current text with original
                txt_txt_ratings += self._compare_texts(
                    item_id, i, curr_txt, base_txt, "original"
                )

            # Cross-modal comparison with original
            if self.loop_type == "I-T-I":
                # For I-T-I: image-text with original
                img_txt_ratings += self._compare_cross(
                    item_id, i, base_img, curr_txt, "original"
                )
                # text-image: not for original in I-T-I
            elif self.loop_type == "T-I-T":
                # For T-I-T: text-image with original
                txt_img_ratings += self._compare_text_image(
                    item_id, i, base_txt, curr_img, "original"
                )
            else:
                # Unknown loop type: do both directions for original
                img_txt_ratings += self._compare_cross(
                    item_id, i, base_img, curr_txt, "original"
                )
                txt_img_ratings += self._compare_text_image(
                    item_id, i, base_txt, curr_img, "original"
                )

            # Compare with previous iteration (only for iterations after the first)
            if i > 1:
                prev_img = _path(rec[f"iter{i - 1}_img"])
                prev_txt = _path(rec[f"iter{i - 1}_text"])

                if self.loop_type == "I-T-I":
                    img_img_ratings += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                    txt_txt_ratings += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )
                elif self.loop_type == "T-I-T":
                    txt_txt_ratings += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )
                    img_img_ratings += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                else:
                    # If loop type unknown, do both comparisons
                    img_img_ratings += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                    txt_txt_ratings += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )

                # Cross-modal comparison with previous
                if self.loop_type == "I-T-I":
                    # For I-T-I: image-text (prev_img vs curr_txt) and text-image (prev_txt vs curr_img)
                    img_txt_ratings += self._compare_cross(
                        item_id, i, prev_img, curr_txt, "previous"
                    )
                    txt_img_ratings += self._compare_text_image(
                        item_id, i, prev_txt, curr_img, "previous"
                    )
                elif self.loop_type == "T-I-T":
                    # For T-I-T: image-text only for previous (curr_img vs prev_txt)
                    img_txt_ratings += self._compare_cross(
                        item_id, i, curr_img, prev_txt, "previous"
                    )
                    # And text-image also happens for previous
                    txt_img_ratings += self._compare_text_image(
                        item_id, i, prev_txt, curr_img, "previous"
                    )
                else:
                    # Unknown loop type: do both directions for previous
                    img_txt_ratings += self._compare_cross(
                        item_id, i, prev_img, curr_txt, "previous"
                    )
                    img_txt_ratings += self._compare_cross(
                        item_id, i, curr_img, prev_txt, "previous"
                    )
                    txt_img_ratings += self._compare_text_image(
                        item_id, i, prev_txt, curr_img, "previous"
                    )

            # Same-step cross-modal comparison
            if self.loop_type == "I-T-I":
                # Keep same-step image-text in I-T-I
                img_txt_ratings += self._compare_cross(
                    item_id, i, curr_img, curr_txt, "same-step"
                )
            elif self.loop_type == "T-I-T":
                # In T-I-T, image-text happens only for previous per requirements
                pass
            else:
                # Unknown loop type: keep same-step image-text for backward compatibility
                img_txt_ratings += self._compare_cross(
                    item_id, i, curr_img, curr_txt, "same-step"
                )

        # write separate ratings files per comparison type
        om.write_json(img_img_ratings, "ratings_image-image.json")
        om.write_json(txt_txt_ratings, "ratings_text-text.json")
        om.write_json(img_txt_ratings, "ratings_image-text.json")
        om.write_json(txt_img_ratings, "ratings_text-image.json")

    def _compare_images(
        self, item: str, step: int, img_a: str, img_b: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("image-image", img_a, img_b)
        return [
            self._package("image-image", item, step, anchor, rating, [img_a, img_b])
        ]

    def _compare_texts(
        self, item: str, step: int, txt_a: str, txt_b: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("text-text", txt_a, txt_b)
        return [self._package("text-text", item, step, anchor, rating, [txt_a, txt_b])]

    def _compare_cross(
        self, item: str, step: int, img: str, txt: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("image-text", img, txt)
        return [self._package("image-text", item, step, anchor, rating, [img, txt])]

    def _compare_text_image(
        self, item: str, step: int, txt: str, img: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("text-image", txt, img)
        return [self._package("text-image", item, step, anchor, rating, [txt, img])]

    def _prepare_contents(self, kind: str, a: str, b: str) -> List[Any]:
        """Prepare the contents list for the Gemini API call based on comparison kind."""
        if kind == "image-image":
            # Accept symlinks as valid entries; we'll error clearly on open if target is missing.
            for p in (a, b):
                if not (os.path.lexists(p) or Path(p).is_symlink() or Path(p).exists()):
                    dir_p = os.path.dirname(os.path.abspath(p))
                    print(f"[DEBUG] Listing files in directory of {p}: {dir_p}")
                    try:
                        print(os.listdir(dir_p))
                    except (OSError, PermissionError) as e:
                        print(f"[DEBUG] Could not list directory {dir_p}: {e}")
                    print(
                        f"[DEBUG] Path entry missing: {p} | exists={Path(p).exists()} is_symlink={Path(p).is_symlink()} lexists={os.path.lexists(p)}"
                    )
                    raise FileNotFoundError(f"Missing image file entry: {p}")
            try:
                img1 = Image.open(a)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                target = None
                if Path(a).is_symlink():
                    try:
                        target = os.readlink(a)
                    except OSError:
                        target = "<unreadable symlink target>"
                raise FileNotFoundError(
                    f"Cannot open image A: {a}. Symlink target: {target}. Error: {e}"
                ) from e
            try:
                img2 = Image.open(b)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                target = None
                if Path(b).is_symlink():
                    try:
                        target = os.readlink(b)
                    except OSError:
                        target = "<unreadable symlink target>"
                raise FileNotFoundError(
                    f"Cannot open image B: {b}. Symlink target: {target}. Error: {e}"
                ) from e
            if img1.mode != "RGB":
                img1 = img1.convert("RGB")
            if img2.mode != "RGB":
                img2 = img2.convert("RGB")
            return [self.prompts["image_image_prompt"], img1, img2]

        if kind == "text-text":
            text1 = (
                Path(a).read_text(encoding="utf-8").strip()
                if Path(a).exists()
                else "No text available"
            )
            text2 = (
                Path(b).read_text(encoding="utf-8").strip()
                if Path(b).exists()
                else "No text available"
            )
            prompt_text = f"Compare these two texts:\nText 1: {text1}\nText 2: {text2}"
            return [self.prompts["text_text_prompt"], prompt_text]

        if kind == "image-text":
            img_path, txt_path = (
                (a, b) if a.lower().endswith((".jpg", ".jpeg", ".png")) else (b, a)
            )
            # Accept symlinked images; error clearly if they cannot be opened
            if not (
                os.path.lexists(img_path)
                or Path(img_path).is_symlink()
                or Path(img_path).exists()
            ):
                raise FileNotFoundError(f"Missing image file entry: {img_path}")
            text = (
                Path(txt_path).read_text(encoding="utf-8").strip()
                if Path(txt_path).exists()
                else "No text available"
            )
            try:
                img = Image.open(img_path)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                target = None
                if Path(img_path).is_symlink():
                    try:
                        target = os.readlink(img_path)
                    except OSError:
                        target = "<unreadable symlink target>"
                raise FileNotFoundError(
                    f"Cannot open image: {img_path}. Symlink target: {target}. Error: {e}"
                ) from e
            if img.mode != "RGB":
                img = img.convert("RGB")
            return [self.prompts["image_text_prompt"], img, f"Text: {text}"]

        if kind == "text-image":
            # Determine which input is text and which is image regardless of order
            if a.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path, txt_path = a, b
            elif b.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path, txt_path = b, a
            else:
                # Neither looks like an image; treat as missing image
                raise FileNotFoundError(
                    "No image file provided for text-image comparison"
                )

            # Accept symlinked images; error clearly if they cannot be opened
            if not (
                os.path.lexists(img_path)
                or Path(img_path).is_symlink()
                or Path(img_path).exists()
            ):
                raise FileNotFoundError(f"Missing image file entry: {img_path}")

            text = (
                Path(txt_path).read_text(encoding="utf-8").strip()
                if Path(txt_path).exists()
                else "No text available"
            )
            try:
                img = Image.open(img_path)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                target = None
                if Path(img_path).is_symlink():
                    try:
                        target = os.readlink(img_path)
                    except OSError:
                        target = "<unreadable symlink target>"
                raise FileNotFoundError(
                    f"Cannot open image: {img_path}. Symlink target: {target}. Error: {e}"
                ) from e
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Reuse same prompt; order the modalities as text then image
            return [self.prompts["image_text_prompt"], f"Text: {text}", img]

        raise ValueError(f"Unknown comparison type: {kind}")

    def _run_rater(self, kind: str, a: str, b: str, max_retries: int = 3) -> Rating:
        if not self.client:
            return DEFAULT_RATING

        for attempt in range(max_retries):
            try:
                contents = self._prepare_contents(kind, a, b)

                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompts.get(
                            "system_instruction_eval", ""
                        ),
                        response_mime_type="application/json",
                        response_schema=_RatingModel,
                    ),
                    contents=contents,
                )

                if hasattr(response, "parsed") and response.parsed:
                    parsed_data = response.parsed
                    # We expect a _RatingModel, but cast to be safe
                    rating_model = cast(_RatingModel, parsed_data)
                    if hasattr(rating_model, "model_dump"):
                        return rating_model.model_dump()
                    if isinstance(rating_model, dict):
                        return rating_model

                # If we reach here, structured output failed.
                print(
                    f"Error: No valid structured output from Gemini for {kind} {a} vs {b} (Attempt {attempt + 1}/{max_retries})"
                )
                if hasattr(response, "text"):
                    print("Raw response:", response.text)

                # Only sleep if we're going to retry
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s...
                    continue

            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"Error during {kind} comparison for '{a}' vs '{b}': {e} (Attempt {attempt + 1}/{max_retries})"
                )

                # Only sleep if we're going to retry
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s...
                    continue

        # If all retries failed, return default rating
        return DEFAULT_RATING

    def _package(
        self,
        typ: str,
        item: str,
        step: int,
        anchor: str,
        rating: Rating,
        items: List[str],
    ) -> Dict[str, Any]:
        rel_items = [Path(item).name for item in items]
        return {
            "item_id": item,
            "step": step,
            "anchor": anchor,
            "comparison_type": typ,
            "comparison_items": rel_items,
            **rating,
        }
