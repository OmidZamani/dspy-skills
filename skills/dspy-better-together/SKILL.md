---
name: dspy-better-together
version: "1.0.0"
dspy-compatibility: "3.2.1"
tags: ["optimizer"]
requires-extras: []
description: Use for BetterTogether, prompt plus weight optimization, fine-tuning sequences, and strategy chains like p -> w -> p.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy BetterTogether

## Goal

Sequence prompt and weight optimizers, evaluate intermediate programs, and return the best candidate.

## Prerequisites

- Use DSPy `3.2.1` or later in the stable `3.2.x` series.
- Assign an LM directly to every predictor with `student.set_lm(lm)`.
- Keep a validation set, or allow `BetterTogether` to hold out part of the trainset.
- Confirm the LM provider supports fine-tuning before including `BootstrapFinetune`.

## Basic Pattern

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

student = dspy.ChainOfThought("question -> answer")
student.set_lm(lm)

def metric(example, pred, trace=None):
    return float(example.answer.lower() == pred.answer.lower())

optimizer = dspy.BetterTogether(
    metric=metric,
    p=dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None:
            dspy.Prediction(score=metric(gold, pred), feedback="Check answer correctness."),
        reflection_lm=dspy.LM("openai/gpt-4o"),
        auto="light",
    ),
    w=dspy.BootstrapFinetune(metric=metric),
)

compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="p -> w -> p",
)
```

## Strategy Choices

| Strategy | Use it when |
|----------|-------------|
| `"p -> w"` | Start with a simple prompt-then-weight pass |
| `"p -> w -> p"` | Re-optimize prompts after fine-tuning |
| `"w -> p"` | Fine-tuning data is already strong |
| Custom chains | Comparing prompt optimizers or conducting controlled experiments |

Optimizer names come from constructor keyword arguments. For example, `mipro=...` and `gepa=...` make `"mipro -> gepa"` valid.

## Per-Optimizer Compile Arguments

Pass optimizer-specific arguments through `optimizer_compile_args`:

```python
compiled = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset,
    strategy="p -> w",
    optimizer_compile_args={
        "p": {"max_metric_calls": 150},
    },
)
```

Do not pass `student` inside `optimizer_compile_args`; `BetterTogether` manages the current program.

## Inspect Results

The returned program exposes:

- `candidate_programs`: evaluated candidates with score and strategy
- `flag_compilation_error_occurred`: whether a step failed before completion

## Related Skills

- Pick optimizers: [dspy-optimizer-selection](../dspy-optimizer-selection/SKILL.md)
- Fine-tune weights: [dspy-finetune-bootstrap](../dspy-finetune-bootstrap/SKILL.md)
- Reflect with GEPA: [dspy-gepa-reflective](../dspy-gepa-reflective/SKILL.md)

## Official Documentation

- **BetterTogether API**: https://dspy.ai/api/optimizers/BetterTogether/
- **Optimizer guide**: https://dspy.ai/learn/optimization/optimizers/
