# Getting Started

This guide walks through setting up the RaA benchmark and running the pipeline.

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/RaA.git
cd RaA
```

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file and provide your Google API key:

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY="your-key"
```

## 4. Configure Parameters

Benchmark parameters are defined in a YAML configuration. A sample is provided at `configs/benchmark_config.yaml`.

## 5. Run the Benchmark

The benchmark is orchestrated by `src/main.py` and supports multiple modes:

- **Full Run** – generation, evaluation, and reporting:
  ```bash
  python src/main.py --config configs/benchmark_config.yaml
  ```
- **Evaluation Only** – run evaluation on existing outputs:
  ```bash
  python src/main.py --config configs/benchmark_config.yaml --eval
  ```
- **Report Only** – generate charts and summaries from existing evaluations:
  ```bash
  python src/main.py --config configs/benchmark_config.yaml --report
  ```

You're ready to experiment with RaA.
