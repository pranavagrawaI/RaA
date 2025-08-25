# Reproducibility as Accuracy (RaA) Benchmark

> ⚠️ this is a work in progress.

RaA is a framework to evaluate information fidelity in iterative multimodal transformations. RaA assesses how well AI models preserve semantic content as data "loops" between different modalities, quantifying drift and degeneration.

**[➡️ View the Full Documentation in `/docs`](./docs/01_introduction.md)**

### Overview

The **Reproducibility as Accuracy (RaA) Benchmark** is a framework for measuring how well AI models preserve information when translating it between different formats, like text and images. At its core, RaA quantifies how much "meaning" is lost in translation.

#### The Analogy: A High-Tech Game of "Telephone" 📞

Imagine playing the game "Telephone," but with AI. You start with an image—say, a photo of a *red car on a book*.

1.  **Whisper to the AI (Image → Text)**: The first AI model looks at the image and "whispers" a description of it: `"A photorealistic image of a small, red toy car on top of a large, open book."`
2.  **Whisper Back (Text → Image)**: This text description is then given to a second AI model, which tries to draw the image based *only* on that description.
3.  **Repeat**: The newly generated image is then shown to the first model, which generates a new description, and the cycle repeats.

After several rounds, how much does the final image resemble the original? This change, or **"semantic drift,"** is precisely what RaA is designed to measure.

#### Technical Workflow

The benchmark operationalizes this "game" through a configurable, automated pipeline:

1.  **The Loop (`I-T-I` or `T-I-T`)**: The core of the benchmark is the `LoopController`, which manages the iterative process. It starts with a "seed" (an image or text) and passes it through a loop for a specified number of iterations.

2.  **Generative Models (`prompt_engine.py`)**: At each step, the `prompt_engine` calls a generative model (like Google's Gemini) to perform the transformation.

3.  **Automated Evaluation (`evaluation_engine.py`)**: Once the loop is complete, the `EvaluationEngine` assesses the drift by performing pairwise comparisons using a multimodal LLM guided by a detailed set of criteria.

By the end of the process, RaA provides a clear report on the model's ability to maintain information fidelity through repeated transformations.

---

### Getting Started

_For a more detailed guide, see [**Getting Started**](./docs/02_getting_started.md) in the docs._

#### Step 1: Installation

1.  Clone the repository.
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```

#### Step 2: API Key Configuration

1.  Create a `.env` file by copying the example: `cp .env.example .env`.
2.  Open `.env` and add your Google API Key.

#### Step 3: Configure the Benchmark

All experiment parameters are defined in a YAML file. See `configs/benchmark_config.yaml` for a complete example.

#### Step 4: Run the Benchmark

The entire pipeline is orchestrated by `src/main.py`.

* **Full Run**: `python src/main.py --config configs/benchmark_config.yaml`
* **Evaluation Only**: `python src/main.py --config configs/benchmark_config.yaml --eval`
* **Reporting Only**: `python src/main.py --config configs/benchmark_config.yaml --report`

---

### Evaluation Criteria

_For a detailed breakdown, see [**Evaluation Criteria**](./docs/04_evaluation.md) in the docs._

The `EvaluationEngine` uses detailed prompts to score similarity across five key criteria:

* **Content Correspondence** (The "What")
* **Compositional Alignment** (The "How")
* **Fidelity & Completeness** (The "Detail")
* **Stylistic Congruence** (The "Feel")
* **Overall Semantic Intent** (The "Message")

---

### Project Structure

_For a deeper dive, see [**Project Architecture**](./docs/03_architecture.md) in the docs._

| File | Description |
| :--- | :--- |
| `main.py` | The main entry point that orchestrates the entire pipeline. |
| `loop_controller.py` | Manages the core recursive loop (I-T-I or T-I-T). |
| `evaluation_engine.py` | Performs automated, pairwise comparisons between artifacts. |
| `graph_creator.py` | Generates plots and charts from the evaluation ratings. |
| `reporting_summary.py` | Generates qualitative, narrative summaries of the evaluation results. |
| `benchmark_config.py` | Defines and loads the YAML configuration. |
