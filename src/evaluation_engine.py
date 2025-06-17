# -*- coding: utf-8 -*-
"""Evaluation Engine for RaA.

This module loads metadata produced by :class:`LoopController`, performs
pairwise comparisons, obtains ratings from either a human or Gemini LLM
backend, and persists the results under each item's ``eval`` folder.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any, Dict, List, Literal

from google import genai
from PIL import Image

from output_manager import OutputManager

Rating = Dict[str, Any]


class EvaluationEngine:
    """Evaluate semantic drift across loop iterations."""

    def __init__(self, exp_root: str, mode: Literal["llm", "human"] = "llm") -> None:
        self.exp_root = pathlib.Path(exp_root)
        self.mode = mode

    def run(self) -> None:
        meta_path = self.exp_root / "metadata.json"
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        for item_id, record in meta.items():
            self._eval_single_item(item_id, record)

    def _eval_single_item(self, item_id: str, rec: Dict[str, str]) -> None:
        om = OutputManager(os.path.join(self.exp_root, item_id, "eval"))
        evals: List[Dict[str, Any]] = []

        def _path(rel: str) -> str:
            return os.path.join(self.exp_root, item_id, rel)

        iters = [
            k.split("_")[0] for k in rec if k.startswith("iter") and k.endswith("_img")
        ]
        iters = sorted({int(idx.replace("iter", "")) for idx in iters})

        start_with_image = "input.jpg" in rec
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
            prev_img = _path(rec[f"iter{i - 1}_img"]) if i > 1 else base_img
            prev_txt = _path(rec[f"iter{i - 1}_text"]) if i > 1 else base_txt

            evals += self._compare_images(item_id, i, curr_img, base_img, "original")
            evals += self._compare_images(item_id, i, curr_img, prev_img, "previous")
            evals += self._compare_texts(item_id, i, curr_txt, base_txt, "original")
            evals += self._compare_texts(item_id, i, curr_txt, prev_txt, "previous")
            evals += self._compare_cross(item_id, i, curr_img, curr_txt, "same-step")

        om.write_json(evals, "ratings.json")

    # ------------------------------------------------------------------
    def _compare_images(
        self, item: str, step: int, img_a: str, img_b: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("image-image", img_a, img_b)
        return [self._package("image-image", item, step, anchor, rating)]

    def _compare_texts(
        self, item: str, step: int, txt_a: str, txt_b: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("text-text", txt_a, txt_b)
        return [self._package("text-text", item, step, anchor, rating)]

    def _compare_cross(
        self, item: str, step: int, img: str, txt: str, anchor: str
    ) -> List[Dict[str, Any]]:
        rating = self._run_rater("image-text", img, txt)
        return [self._package("image-text", item, step, anchor, rating)]

    # ------------------------------------------------------------------
    def _format_prompt(self, kind: str, a: str, b: str) -> str:
        base = "Rate the semantic similarity of the following items on a scale of 1 (very different) to 5 (identical)."
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

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"score": 3, "reason": "Missing GOOGLE_API_KEY"}

        client = genai.Client(api_key=api_key)
        prompt = (
            'Rate the semantic similarity from 1 (very different) to 5 '
            '(identical) and justify. Respond in JSON {"score": int, "reason": str}.'
        )

        try:
            if kind == "image-image":
                img1 = Image.open(a)
                img2 = Image.open(b)
                contents = [img1, img2, prompt]
            elif kind == "text-text":
                text1 = open(a, "r", encoding="utf-8").read()
                text2 = open(b, "r", encoding="utf-8").read()
                contents = [text1, text2, prompt]
            elif kind == "image-text":
                if a.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(a)
                    text = open(b, "r", encoding="utf-8").read()
                else:
                    img = Image.open(b)
                    text = open(a, "r", encoding="utf-8").read()
                contents = [img, text, prompt]
            else:
                raise ValueError(f"Unknown comparison type: {kind}")

            resp = client.models.generate_content(
                model="gemini-2.0-flash-lite", contents=contents
            )
            text = self._extract_response_text(resp)
            if not text:
                raise ValueError("Empty response")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                raise
        except Exception as e:  # noqa: BLE001
            return {"score": 3, "reason": f"LLM error: {e}"}

    def _package(
        self, typ: str, item: str, step: int, anchor: str, rating: Rating
    ) -> Dict[str, Any]:
        return {
            "item_id": item,
            "step": step,
            "anchor": anchor,
            "comparison_type": typ,
            **rating,
        }
