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
from typing import Any, Dict, List, Literal

from google import genai
from PIL import Image
import types


# Fallback criteria used for rating different comparison types.
CRITERIA = {
    "image-image": [
        {
            "id": "content",
            "question": "How similar are the main objects and their arrangement?",
        },
        {
            "id": "style",
            "question": (
                "How similar are the artistic or visual styles (colours, "
                "textures, lighting)?"
            ),
        },
        {
            "id": "overall",
            "question": "Overall, how visually similar are A and B?",
        },
    ],
    "text-text": [
        {
            "id": "facts",
            "question": "Do both texts express the same core facts?",
        },
        {
            "id": "details",
            "question": "Are the specific details (names, numbers, attributes) preserved?",
        },
        {
            "id": "overall",
            "question": "Overall semantic similarity of A and B?",
        },
    ],
    "image-text": [
        {
            "id": "objects_match",
            "question": (
                "Does the image accurately depict the entities and actions "
                "described in the text?"
            ),
        },
        {
            "id": "missing_or_extra",
            "question": (
                "Are important elements missing from the image that are "
                "mentioned in the text (or vice-versa)?"
            ),
        },
        {
            "id": "overall_align",
            "question": "Overall alignment between image and text?",
        },
    ],
}


_RealGenaiClient = genai.Client


class _SafeGenaiClient:
    """Wrapper around ``genai.Client`` that tolerates missing API keys."""

    def __init__(self, *args, **kwargs):
        try:
            self._client = _RealGenaiClient(*args, **kwargs)
        except Exception:
            self._client = None
            self.models = types.SimpleNamespace(generate_content=lambda *_, **__: None)

    def __getattr__(self, name):
        if self._client is not None:
            return getattr(self._client, name)
        raise AttributeError(name)


genai.Client = _SafeGenaiClient

from output_manager import OutputManager

Rating = Dict[str, Any]


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

    def _format_prompt(self, kind: str, a: str, b: str) -> str:
        base = """Rate the semantic similarity of the following items on 
        a scale of 1 (very different) to 5 (identical)."""

        if kind == "text-text":
            text_a = open(a, "r", encoding="utf-8").read()
            text_b = open(b, "r", encoding="utf-8").read()
            return f"[TEXT-TEXT]\nA: {text_a}\nB: {text_b}\n{base}"
        if kind == "image-image":
            return f"[IMAGE-IMAGE]\nA: {a}\nB: {b}\n{base}"
        if kind == "image-text":
            if a.lower().endswith((".jpg", ".jpeg", ".png")):
                text = open(b, "r", encoding="utf-8").read()
                img = a
            else:
                text = open(a, "r", encoding="utf-8").read()
                img = b
            return f"[IMAGE-TEXT]\nImage: {img}\nText: {text}\n{base}"
        return f"{base}"

    def _extract_response_text(self, resp: Any) -> str | None:
        """Return the textual content from a Gemini response."""
        if resp is None:
            return None
        if hasattr(resp, "text"):
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            parts = getattr(getattr(cand, "content", None), "parts", None)
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text
        return None

    def _run_rater(self, kind: str, a: str, b: str) -> Rating:
        if self.mode == "human":
            prompt = self._format_prompt(kind, a, b)
            print("-" * 15, "\nCopy-paste to human UI:\n", prompt)
            score = int(input("Score 1-5? "))
            reason = input("Reason? ")[:280]
            return {"score": score, "reason": reason}

        if not self.client and self.mode == "llm":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    self.client = genai.Client(api_key=api_key)
                except Exception:
                    self.client = genai.Client()

        if not self.client:
            return {"score": 3, "reason": "Missing GOOGLE_API_KEY"}

        client = self.client
        base_prompt = """You are an expert in analyzing semantic similarity between content.
        Rate the similarity from 1 (very different) to 5 (identical) and explain your rating.
        Always format your response as a JSON object with exactly two fields: "score" (integer 1-5) and "reason" (string).
        Example: {"score": 4, "reason": "The images are highly similar in composition and subject matter."}
        """

        try:
            if kind == "image-image":
                # Ensure both files exist and can be opened as images
                if not (Path(a).exists() and Path(b).exists()):
                    raise FileNotFoundError(f"Missing image file(s): {a} or {b}")

                with Image.open(a) as img1, Image.open(b) as img2:
                    # Convert to RGB mode if needed
                    if img1.mode != "RGB":
                        img1 = img1.convert("RGB")
                    if img2.mode != "RGB":
                        img2 = img2.convert("RGB")

                    prompt_part = """Compare these two images:"""
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=[base_prompt, prompt_part, img1, img2],
                    )

            elif kind == "text-text":
                # Handle missing text files by using placeholders
                text1 = "No text available"
                text2 = "No text available"

                if Path(a).exists():
                    with open(a, "r", encoding="utf-8") as f:
                        text1 = f.read().strip()
                if Path(b).exists():
                    with open(b, "r", encoding="utf-8") as f:
                        text2 = f.read().strip()

                prompt_text = f"""Compare these two texts:
                Text 1: {text1}
                Text 2: {text2}"""

                response = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=[base_prompt + "\n" + prompt_text],
                )

            elif kind == "image-text":
                # Determine which is image and which is text
                if a.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path, txt_path = a, b
                else:
                    img_path, txt_path = b, a

                if not Path(img_path).exists():
                    raise FileNotFoundError(f"Missing image file: {img_path}")

                text = "No text available"
                if Path(txt_path).exists():
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                with Image.open(img_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    prompt_part = """Compare this image to the following text:"""
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=[base_prompt, prompt_part, img, f"Text: {text}"],
                    )

            else:
                raise ValueError(f"Unknown comparison type: {kind}")

            try:
                response_text = self._extract_response_text(response)
                if response_text:
                    response_text = response_text.strip()
                    try:
                        start_idx = response_text.find("{")
                        end_idx = response_text.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = response_text[start_idx:end_idx]
                            result = json.loads(json_str)
                            if isinstance(
                                result.get("score"), (int, float)
                            ) and isinstance(result.get("reason"), str):
                                return {
                                    "score": int(result["score"]),
                                    "reason": result["reason"][:280],
                                }
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass

                    try:
                        words = response_text.lower().split()
                        score = 3
                        for word in words:
                            if word.isdigit() and 1 <= int(word) <= 5:
                                score = int(word)
                                break
                        return {"score": score, "reason": response_text[:280]}
                    except Exception:
                        pass

                return {"score": -1, "reason": "Could not parse response"}
            except Exception:
                return {"score": -1, "reason": "Error processing response"}

        except Exception as e:
            return {"score": -1, "reason": f"Error: {str(e)}"}

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
