# Evaluation

This document outlines the rubric used to assess semantic drift between iterative outputs in RaA. All comparisons are scored on a 1.0–10.0 scale with at least one decimal place of precision.

## Scoring Dimensions

### Content Correspondence ("What")
Do the two artifacts feature the same primary subjects, topics, and entities?

### Compositional Alignment ("How")
Is the logical structure, flow, or relationship between elements consistent between the artifacts?

### Fidelity & Completeness ("Detail")
Do both artifacts contain comparable levels of information, or is one a summary, elaboration, or omission of the other?

### Stylistic Congruence ("Feel")
Is the style, tone, and presentation similar across the artifacts?

### Overall Semantic Intent ("Message")
Considering all the above, do the artifacts serve the same purpose or convey the same core message?

## Prompt Templates & Scoring Rules

Evaluation prompts and scoring examples reside in the [prompts](../prompts) directory:

- [system_instruction_eval.txt](../prompts/system_instruction_eval.txt) – overarching evaluation guidance and JSON schema.
- [text_text_prompt.txt](../prompts/text_text_prompt.txt) – text vs. text comparison template with scoring examples.
- [image_image_prompt.txt](../prompts/image_image_prompt.txt) – image vs. image evaluation instructions.
- [image_text_prompt.txt](../prompts/image_text_prompt.txt) – cross-modal image–text evaluation template.

These templates define scoring conventions and should be referenced when implementing or extending the evaluation pipeline.

