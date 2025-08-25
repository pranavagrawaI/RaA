# Architecture Overview

## Module Summaries

- **`main.py`** – Entry point that loads a benchmark configuration, runs the selected loop, and optionally performs evaluation and reporting.
- **`loop_controller.py`** – Executes recursive I‑T‑I or T‑I‑T loops, saving each iteration’s outputs with retry logic and progress tracking.
- **`evaluation_engine.py`** – Reads loop metadata, performs intra‑ and cross‑modal comparisons, and records ratings from a Gemini model.
- **`graph_creator.py`** – Converts evaluation JSON files into per‑criterion charts and infers loop type to group results.
- **`reporting_summary.py`** – Aggregates evaluation records and uses Gemini to produce a qualitative summary for each item.
- **`benchmark_config.py`** – Dataclass helpers that load YAML settings for loops, prompts, logging, evaluation, and reporting.

## Pipeline

1. **Seed Input** – Images or texts from the configured `input_dir`.
2. **Generation** – `LoopController` alternates captioning and image generation for the configured iterations, storing files and metadata.
3. **Evaluation** – `EvaluationEngine` compares each step against the original and previous outputs, producing structured rating files.
4. **Reporting** – `GraphCreator` plots rating trends while `SummaryGenerator` writes qualitative summaries.

```text
Seed Inputs
    |
    v
LoopController (generation)
    |
    v
EvaluationEngine -- ratings_*.json
    |\
    | \-- GraphCreator -> charts
    \---- SummaryGenerator -> qualitative_summary.txt
```
