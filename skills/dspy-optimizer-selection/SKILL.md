---
name: dspy-optimizer-selection
version: "2.0.0"
dspy-compatibility: "3.2.1"
description: This skill should be used when the user asks to "choose a DSPy optimizer", "compare DSPy optimizers", "which teleprompter should I use", "optimize prompts or weights", mentions LabeledFewShot, BootstrapFewShotWithRandomSearch, KNNFewShot, COPRO, MIPROv2, SIMBA, GEPA, BootstrapFinetune, Ensemble, or BetterTogether, or needs a cost-aware DSPy optimization plan.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Optimizer Selection

## Goal

Choose the smallest DSPy optimizer that matches the data, budget, and artifact being tuned. Establish a baseline before compiling anything.

## Selection Matrix

| Need | Start with | Notes |
|------|------------|-------|
| Include a few labeled examples | `dspy.LabeledFewShot` | Random labeled demos; useful as a baseline |
| About 10 examples | `dspy.BootstrapFewShot` | Teacher-generated demos with metric filtering |
| 50+ examples and stronger demo search | `dspy.BootstrapFewShotWithRandomSearch` | Searches multiple demo sets; alias: `dspy.BootstrapRS` |
| Per-input nearest demos | `dspy.KNNFewShot` | Retrieves nearby examples before bootstrapping |
| Instruction-only hill climbing | `dspy.COPRO` | Coordinate ascent over instructions |
| Instruction and demo search | `dspy.MIPROv2` | Bayesian search; install `dspy[optuna]` |
| Mini-batch introspective rules or demos | `dspy.SIMBA` | Uses output variability and self-reflection |
| Rich textual feedback and trace reflection | `dspy.GEPA` | Metric must accept five arguments |
| Distill prompts into model weights | `dspy.BootstrapFinetune` | Requires a fine-tunable LM and `set_lm()` |
| Combine candidate programs | `dspy.Ensemble` | Trades inference cost for robustness |
| Sequence prompt and weight optimization | `dspy.BetterTogether` | Meta-optimizer for configurable optimizer chains |

## Workflow

1. Split data into train and validation sets.
2. Evaluate the uncompiled program with [dspy-evaluation-suite](../dspy-evaluation-suite/SKILL.md).
3. Start with the least expensive optimizer that matches the need.
4. Save the compiled program and compare it against the baseline.
5. Escalate only when the measured gain justifies extra LM calls, fine-tuning, or inference cost.

## Common Paths

### Fast Demo Optimization

Use [dspy-bootstrap-fewshot](../dspy-bootstrap-fewshot/SKILL.md) for the first optimization pass. Move to `BootstrapFewShotWithRandomSearch` when enough examples are available to search multiple demo sets.

### Prompt Search

Use [dspy-miprov2-optimizer](../dspy-miprov2-optimizer/SKILL.md) for instruction and demonstration search. Install its optional dependency first:

```bash
pip install -U "dspy[optuna]>=3.2.1,<3.3"
```

### Reflective Optimization

Use [dspy-gepa-reflective](../dspy-gepa-reflective/SKILL.md) when failures can be described with actionable text. Use [dspy-simba-optimizer](../dspy-simba-optimizer/SKILL.md) for a smaller mini-batch introspective loop with numeric metrics.

### Prompt Plus Weight Optimization

Use [dspy-better-together](../dspy-better-together/SKILL.md) when a fine-tunable LM is available and prompt optimization alone has plateaued.

## Best Practices

1. Keep a held-out validation set.
2. Track optimization cost and inference cost separately.
3. Use reproducible seeds where supported.
4. Avoid claiming one optimizer is universally best; compare measured results.
5. Save intermediate candidates for expensive runs.

## Official Documentation

- **Optimizer guide**: https://dspy.ai/learn/optimization/optimizers/
- **Optimizer API index**: https://dspy.ai/api/optimizers/
- **DSPy releases**: https://github.com/stanfordnlp/dspy/releases
