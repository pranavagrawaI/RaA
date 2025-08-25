# Introduction

Reproducibility as Accuracy (RaA) is a benchmark for measuring how well information survives as it loops between modalities like text and images. Rather than checking a single transformation, RaA focuses on cumulative **semantic drift**: the gradual loss of meaning when outputs are fed back into generative models over multiple iterations.

## A Game of Telephone

RaA operates like a high‑tech game of "telephone." An initial image or text is handed to a model which produces a translation in the opposite modality. That result becomes the input for the next step, and the cycle repeats. As the loop continues, subtle changes compound until the final artifact may differ drastically from the starting point—just as whispered messages distort in a playground game.

## Loop Workflow

The loop is coordinated by the `LoopController`, which reads configuration settings and orchestrates each iteration. For every step it uses `prompt_engine.py` to generate the next image or caption, then hands the accumulated outputs to `evaluation_engine.py` once the loop completes. The evaluation stage performs pairwise comparisons to quantify how much meaning drifted during the process.

## Status

RaA is still a work in progress and subject to rapid change.
