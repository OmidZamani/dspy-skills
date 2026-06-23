---
name: dspy-reasoning-modules
version: "1.0.0"
dspy-compatibility: "3.2.1"
tags: ["reasoning"]
requires-extras: []
description: Use for DSPy reasoning modules including RLM, ProgramOfThought, CodeAct, Parallel, sandboxed execution, and long-context workflows.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Reasoning Modules

## Goal

Choose the appropriate DSPy reasoning module for long-context exploration, code-assisted reasoning, or parallel execution.

## Module Selection

| Module | Use it for | Important constraint |
|--------|------------|----------------------|
| `dspy.RLM` | Exploring very large contexts with iterative REPL code and recursive sub-LM calls | Experimental; requires Deno by default |
| `dspy.ProgramOfThought` | Solving tasks by generating and executing Python | Requires Deno by default |
| `dspy.CodeAct` | Combining generated Python with predefined tool functions | Functions only; requires Deno |
| `dspy.Parallel` | Running `(module, example)` pairs concurrently | Tune threads and error handling |

## RLM for Large Contexts

`RLM` treats long inputs as external data in a sandbox rather than placing the full context in each LM prompt.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

rlm = dspy.RLM(
    "document, question -> answer",
    max_iterations=12,
    max_llm_calls=30,
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)

result = rlm(
    document=very_long_document,
    question="What were the main revenue drivers?",
)
print(result.answer)
```

Use `max_iterations`, `max_llm_calls`, and `max_output_chars` as explicit cost and output bounds.

## Sandboxed Execution

The default `dspy.PythonInterpreter` uses Deno and Pyodide. It denies host filesystem, environment, and network access unless explicitly enabled.

```python
from pathlib import Path
import dspy

with dspy.PythonInterpreter(
    enable_read_paths=[Path("./inputs")],
    enable_network_access=["api.example.com"],
) as interpreter:
    print(interpreter.execute("print('ready')"))
```

Grant only the minimum paths, environment variables, and network hosts needed by the task.

## ProgramOfThought and CodeAct

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

math = dspy.ProgramOfThought("question -> answer")
print(math(question="What is the sum of the first 100 integers?").answer)
```

Use `CodeAct` when generated code also needs curated host-side tools:

```python
def lookup_rate(currency: str) -> float:
    """Return a trusted exchange rate from the application service."""
    return rates[currency]

agent = dspy.CodeAct("amount, currency -> converted", tools=[lookup_rate])
```

## Parallel Execution

```python
parallel = dspy.Parallel(num_threads=8, return_failed_examples=True)
results, failed_examples, exceptions = parallel(
    [(program, {"question": question}) for question in questions]
)
```

## Best Practices

1. Prefer `Predict` or `ChainOfThought` until code execution or long-context exploration is justified.
2. Treat `RLM` as experimental and load-test before production deployment.
3. Bound loops and sub-LM calls.
4. Keep sandbox permissions narrow.
5. Create separate interpreters for concurrent custom-interpreter use.

## Official Documentation

- **RLM API**: https://dspy.ai/api/modules/RLM/
- **ProgramOfThought API**: https://dspy.ai/api/modules/ProgramOfThought/
- **CodeAct API**: https://dspy.ai/api/modules/CodeAct/
- **Parallel API**: https://dspy.ai/api/modules/Parallel/
