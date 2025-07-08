# -*- coding: utf-8 -*-
"""Evaluation Engine for RaA.

This module loads metadata produced by :class:`LoopController`, performs
pairwise comparisons, obtains ratings from either a human or Gemini LLM
backend, and persists the results under each item's ``eval`` folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

from google import genai
from google.genai import types
from pydantic import BaseModel
from PIL import Image

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


SYSTEM_INSTRUCTION = """
You are an Expert Multimodal Analyst.Your task is to compare two inputs (A and B) with the nuanced judgment of a human specialist. You will evaluate them across five universal criteria and provide a single, consistent JSON output.

Adhere to these four Guiding Principles in all evaluations:

1.  **Deconstruct, Then Compare:** Before scoring, mentally break down each input into its core components: the subjects, the actions, the relationships between them, and the overall mood or style. Do not make a holistic judgment until you have analyzed the parts.

2.  **Prioritize Core Meaning:** The most important question is "Do A and B tell the same fundamental story or convey the same core message?" A stylistic change (e.g., photo vs. painting) is less significant than a change in the central action or relationship (e.g., a cat chasing a mouse vs. a mouse chasing a cat).

3.  **Score the *Difference*, Not the Quality:** Your task is to measure similarity, not to judge which input is "better." A low-quality photo of a subject is a near-perfect match to a high-quality photo of the *same subject*. Your scores must reflect the degree of transformation between A and B, not their intrinsic quality.

4.  **Justify with Specifics:** Your "reason" for each score is critical. It must be concise and point to concrete evidence in the inputs. Avoid vague justifications. Instead of "The style is different," write "A is a photorealistic image, while B is an abstract watercolor painting."
"""

MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"

Rating = Dict[str, Dict[str, Any]]  # Updated to match new schema

DEFAULT_RATING = {
    "content_correspondence": {"score": -1.0, "reason": "Rating unavailable"},
    "compositional_alignment": {"score": -1.0, "reason": "Rating unavailable"},
    "fidelity_completeness": {"score": -1.0, "reason": "Rating unavailable"},
    "stylistic_congruence": {"score": -1.0, "reason": "Rating unavailable"},
    "overall_semantic_intent": {"score": -1.0, "reason": "Rating unavailable"},
}


class EvaluationEngine:
    """Evaluate semantic drift across loop iterations."""

    def __init__(
        self,
        exp_root: str,
        mode: Literal["llm", "human"] = "llm",
        config=None,
        client: genai.Client | None = None,
    ) -> None:
        self.exp_root = Path(exp_root)
        self.mode = mode
        self.loop_type = (
            config.loop.type.upper() if config else ""
        )  # Will do all comparisons if config not provided
        if client is not None:
            self.client = client
        elif self.mode == "llm":
            api_key = os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key) if api_key else None
        else:
            self.client = None
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompts = {}
        for prompt_file in prompts_dir.glob("*.txt"):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts[prompt_file.stem] = f.read()
        return prompts

    def run(self) -> None:
        meta_path = self.exp_root / "metadata.json"
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        for item_id, record in meta.items():
            self._eval_single_item(item_id, record)

    def _eval_single_item(self, item_id: str, rec: Dict[str, str]) -> None:
        om = OutputManager(self.exp_root / item_id / "eval")
        evals: List[Dict[str, Any]] = []

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
                evals += self._compare_images(
                    item_id, i, curr_img, base_img, "original"
                )
            elif self.loop_type == "T-I-T":
                # For T-I-T: Always compare current text with original
                evals += self._compare_texts(item_id, i, curr_txt, base_txt, "original")
            else:
                # If loop type unknown, do both comparisons
                evals += self._compare_images(
                    item_id, i, curr_img, base_img, "original"
                )
                evals += self._compare_texts(item_id, i, curr_txt, base_txt, "original")

            # Compare with previous iteration (only for iterations after the first)
            if i > 1:
                prev_img = _path(rec[f"iter{i - 1}_img"])
                prev_txt = _path(rec[f"iter{i - 1}_text"])

                if self.loop_type == "I-T-I":
                    # For I-T-I: Compare both images and texts with previous iteration
                    evals += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                    evals += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )
                elif self.loop_type == "T-I-T":
                    # For T-I-T: Compare both texts and images with previous iteration
                    evals += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )
                    evals += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                else:
                    # If loop type unknown, do both comparisons
                    evals += self._compare_images(
                        item_id, i, curr_img, prev_img, "previous"
                    )
                    evals += self._compare_texts(
                        item_id, i, curr_txt, prev_txt, "previous"
                    )

            # Always do image-text comparison for current iteration
            evals += self._compare_cross(item_id, i, curr_img, curr_txt, "same-step")

        om.write_json(evals, "ratings.json")

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

    def _prepare_contents(self, kind: str, a: str, b: str) -> List[Any]:
        """Prepare the contents list for the Gemini API call based on comparison kind."""
        if kind == "image-image":
            if not (Path(a).exists() and Path(b).exists()):
                raise FileNotFoundError(f"Missing image file(s): {a} or {b}")
            img1 = Image.open(a)
            img2 = Image.open(b)
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
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Missing image file: {img_path}")
            text = (
                Path(txt_path).read_text(encoding="utf-8").strip()
                if Path(txt_path).exists()
                else "No text available"
            )
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return [self.prompts["image_text_prompt"], img, f"Text: {text}"]

        raise ValueError(f"Unknown comparison type: {kind}")

    def _run_rater(self, kind: str, a: str, b: str) -> Rating:
        if self.mode == "human":
            print(f"--- Human Evaluation: {kind} ---")
            print(f"A: {a}")
            print(f"B: {b}")

            rating = {}
            # Dynamically get keys from the Pydantic model for robustness
            for criterion in list(_RatingModel.model_fields.keys()):
                print(f"\nEnter rating for {criterion}:")
                score = float(input("Score 1-10? "))
                reason = input("Reason? ")[:280]
                rating[criterion] = {"score": score, "reason": reason}
            return rating

        if not self.client:
            return DEFAULT_RATING

        try:
            contents = self._prepare_contents(kind, a, b)

            response = self.client.models.generate_content(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
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
                f"Error: No valid structured output from Gemini for {kind} {a} vs {b}"
            )
            if hasattr(response, "text"):
                print("Raw response:", response.text)
            return DEFAULT_RATING

        except Exception as e:
            print(f"Error during {kind} comparison for '{a}' vs '{b}': {e}")
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
        # Convert absolute paths to relative paths for cleaner output
        rel_items = [Path(item).name for item in items]
        return {
            "item_id": item,
            "step": step,
            "anchor": anchor,
            "comparison_type": typ,
            "comparison_items": rel_items,
            **rating,
        }
