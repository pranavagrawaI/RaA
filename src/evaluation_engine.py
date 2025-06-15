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
from typing import Any, Dict, List, Literal

from output_manager import OutputManager
from prompt_engine import generate_caption  # noqa: F401  # optional reuse
from google import genai

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

        iters = [k.split("_")[0] for k in rec if k.startswith("iter") and k.endswith("_img")]
        iters = sorted({int(idx.replace("iter", "")) for idx in iters})

        for i in iters:
            original_img = _path("input.jpg" if "input.jpg" in rec else "input.txt")
            prev_img = _path(rec[f"iter{i-1}_img"]) if i > 1 else original_img
            curr_img = _path(rec[f"iter{i}_img"])
            original_txt = _path("input.txt" if "input.txt" in rec else "input.jpg")
            prev_txt = _path(rec[f"iter{i-1}_text"]) if i > 1 else original_txt
            curr_txt = _path(rec[f"iter{i}_text"])

            evals += self._compare_images(item_id, i, curr_img, original_img, "original")
            evals += self._compare_images(item_id, i, curr_img, prev_img, "previous")
            evals += self._compare_texts(item_id, i, curr_txt, original_txt, "original")
            evals += self._compare_texts(item_id, i, curr_txt, prev_txt, "previous")
            evals += self._compare_cross(item_id, i, curr_img, curr_txt, "same-step")

        om.write_json(evals, "ratings.json")

    # ------------------------------------------------------------------
    def _compare_images(self, item: str, step: int, img_a: str, img_b: str, anchor: str) -> List[Dict[str, Any]]:
        question = self._format_prompt("image-image", img_a, img_b)
        rating = self._run_rater(question)
        return [self._package("image-image", item, step, anchor, rating)]

    def _compare_texts(self, item: str, step: int, txt_a: str, txt_b: str, anchor: str) -> List[Dict[str, Any]]:
        question = self._format_prompt("text-text", txt_a, txt_b)
        rating = self._run_rater(question)
        return [self._package("text-text", item, step, anchor, rating)]

    def _compare_cross(self, item: str, step: int, img: str, txt: str, anchor: str) -> List[Dict[str, Any]]:
        question = self._format_prompt("image-text", img, txt)
        rating = self._run_rater(question)
        return [self._package("image-text", item, step, anchor, rating)]

    # ------------------------------------------------------------------
    def _format_prompt(self, kind: str, a: str, b: str) -> str:
        return (
            f"[{kind.upper()}]\nItem A: {a}\nItem B: {b}\n"
            "Rate similarity 1-5 and justify. Respond in JSON {\"score\": int, \"reason\": str}."
        )

    def _run_rater(self, prompt: str) -> Rating:
        if self.mode == "human":
            print("-" * 15, "\nCopy-paste to human UI:\n", prompt)
            score = int(input("Score 1-5? "))
            reason = input("Reason? ")[:280]
            return {"score": score, "reason": reason}

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"score": 3, "reason": "Missing GOOGLE_API_KEY"}

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.0-pro", contents=[prompt])
        try:
            return json.loads(response.text)
        except Exception as e:  # noqa: BLE001
            return {"score": 3, "reason": f"LLM parse error: {e}"}

    def _package(self, typ: str, item: str, step: int, anchor: str, rating: Rating) -> Dict[str, Any]:
        return {
            "item_id": item,
            "step": step,
            "anchor": anchor,
            "comparison_type": typ,
            **rating,
        }
