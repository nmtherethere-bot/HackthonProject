# Title: Teaching Gemma to Show Its Work with Tunix (SFT + GRPO)

## Subtitle
Transparent step-by-step reasoning on math with a TPU-native post-training stack.

## Track
Reasoning Traces / Show-Your-Work (Gemma2 2B or Gemma3 1B)

## Overview (Goal and Approach)
- We fine-tune Gemma with Tunix to produce explicit reasoning traces and final answers.
- Pipeline: SFT on reasoning traces, then GRPO with a composite reward (correctness, format, length, step consistency, stop bonus).

## Model and Training
- Base: `google/gemma-2-2b-it` (or `google/gemma-3-1b`)
- Hardware: TPU v4-8 on GCP
- Algorithms: SFT → GRPO (Tunix)
- Key settings: bf16, max seq len 2k, small batch per device, KL control.

## Data
- Primary: GSM8K (train split) with parsing to `<reasoning>...</reasoning>` and `<final>...</final>` tags.
- Preprocessing script provided under `src/tunix_reasoning/data.py`.

## Reward Function Composition
`reward = w_correct*EM + w_format*Format + w_length*LengthWindow + w_step*Consistency + w_stop*StopBonus`.
- Encourages correct, formatted, and concise step-by-step reasoning.

## Results
- Report pass@1 exact match on a held-out subset (show table).
- Include format adherence and average trace length.

## Reproduction
- Public notebook attached.
- Configs in `configs/*.yaml`.
- Steps: prepare data → run SFT → run GRPO → evaluate.

## Limitations and Future Work
- Explore better rewards (equivalence checkers, unit tests), curriculum, and multi-task reasoning datasets.

## Media
- Cover image: high-level diagram of the pipeline.
- 3-minute video: walkthrough of configs, reward design, and demo inference.