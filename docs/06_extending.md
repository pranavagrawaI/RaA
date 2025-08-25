# Extending the RaA Benchmark

This guide explains how to customize RaA for new loops, models, metrics, prompts, and reporting outputs.

## Updating `benchmark_config.yaml`

Configuration lives in `configs/benchmark_config.yaml`. To change the loop pattern or number of iterations adjust the `loop` block:

```yaml
loop:
  type: "I-T-I"
  num_iterations: 5
```

Set `type` to the desired modality order (e.g. `T-I-T` or `I-T-T`) and `num_iterations` to the number of passes. If you introduce a new loop pattern, implement a corresponding handler in `src/loop_controller.py` and branch it in `LoopController.run`. When models require extra parameters, add them under new keys in the YAML and mirror the structure in `src/benchmark_config.py` so they are parsed into the dataclass.

## Adding transformation models

Model calls reside in `src/prompt_engine.py`. Implement a new function similar to `generate_caption` or `generate_image` and route the loop controller to it. Keep the Gemini-based implementations as reference for error handling and fallbacks, or swap in other APIs or local models.

## Adding evaluation metrics

Evaluation scores are defined by `_RatingModel` and `DEFAULT_RATING` in `src/evaluation_engine.py`, while `CRITERIA` in `src/graph_creator.py` controls chart generation. To introduce a new metric:

1. Add the metric to `_RatingModel` and `DEFAULT_RATING`.
2. Include the metric name in `CRITERIA` so charts render it.
3. Update any reporting logic that expects the previous schema.

## Contributing prompts and reporting formats

*Prompts* — Add plain text files under `prompts/`. The evaluation engine loads all `*.txt` files automatically and they can be referenced by name in code or configuration.

*Reporting* — New report types can extend `src/reporting_summary.py` or add modules alongside `GraphCreator` for bespoke visualisations. Ensure any additional output is wired into the `reporting` section of the configuration so it can be toggled on or off.
