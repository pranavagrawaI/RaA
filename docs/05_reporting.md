# Reporting

The final step in the RaA pipeline is to generate reports that provide a clear and concise overview of the benchmark results. These reports are designed to help you quickly understand the semantic drift that occurred during the loop.

There are two types of reports that are generated:

## 1. Quantitative Charts

The `graph_creator.py` module generates a series of charts that visualize the evaluation scores over each iteration of the loop. These charts are saved as PNG files in the `eval` directory for each item.

The charts help you to quickly identify:

* **Trends**: Are the scores generally stable, or do they degrade over time?
* **Significant Drops**: At which iteration does the most significant drop in fidelity occur?
* **Areas of Failure**: Which of the five criteria are most affected by the semantic drift?

## 2. Qualitative Summaries

The `reporting_summary.py` module uses a generative model to create a narrative summary of the evaluation results. This summary is saved as a `qualitative_summary.txt` file in the `eval` directory for each item.

The qualitative summary provides a human-readable story of the semantic drift, including:

* **The Core Narrative**: A high-level overview of the trend (e.g., "the information was generally stable," "the meaning degraded quickly after the third iteration").
* **The "Why"**: An explanation of *why* the degradation is happening, based on the scores and reasons from the evaluation.
* **Cause and Effect**: A synthesis of the data from the different comparison types (e.g., how a drop in `image-image` similarity was preceded by a change in the intermediate `text-text` comparison).

These two reports provide a comprehensive view of the benchmark results, allowing you to quickly and easily understand the performance of the models being tested.
