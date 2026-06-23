---
name: dspy-production-deployment
version: "1.0.0"
dspy-compatibility: "3.2.1"
tags: ["production"]
requires-extras: []
description: Use for deploying DSPy with save/load, configure_cache, restrict_pickle, track_usage, async execution, streaming, and production runtime controls.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Production Deployment

## Goal

Prepare a DSPy program for repeatable, observable, scalable, and safer production execution.

## Cache Hardening

DSPy enables memory and disk caches by default. Disk cache deserialization uses pickle unless restricted. Enable the allowlist mode in production:

```python
import dspy

dspy.configure_cache(restrict_pickle=True)
```

Register trusted custom cache types only when needed:

```python
dspy.configure_cache(
    restrict_pickle=True,
    safe_types=[MyResult, Metadata],
)
```

Disable a cache layer explicitly when a deployment cannot persist data or requires fresh model responses:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=True,
)
```

## Save and Load

Prefer state-only JSON for readable, safer artifacts:

```python
compiled.save("./artifacts/program.json", save_program=False)

loaded = MyProgram()
loaded.load("./artifacts/program.json")
```

Use whole-program save only for trusted artifacts. It uses cloudpickle:

```python
compiled.save("./artifacts/program/", save_program=True)
loaded = dspy.load("./artifacts/program/")
```

Keep the DSPy major version compatible when loading saved programs.

## Usage Tracking

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    track_usage=True,
)

prediction = program(question="What is DSPy?")
print(prediction.get_lm_usage())
```

Cached calls return no new token usage.

## Async Execution

Most built-in modules support `acall()`:

```python
import asyncio

async def main():
    prediction = await program.acall(question="What is DSPy?")
    print(prediction.answer)

asyncio.run(main())
```

Implement `aforward()` for custom async modules. Use `dspy.asyncify(program)` only when adapting a synchronous callable is the right boundary.

## Streaming

```python
import asyncio
import dspy

stream_program = dspy.streamify(
    dspy.Predict("question -> answer"),
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="answer"),
    ],
)

async def main():
    async for chunk in stream_program(question="Explain DSPy briefly."):
        print(chunk)

asyncio.run(main())
```

For looped modules such as ReAct, set `allow_reuse=True` on listeners for repeated fields. Cache hits yield the final `Prediction` without replaying token chunks.

## Production Checklist

1. Pin the stable DSPy series.
2. Use state-only JSON unless whole-program pickle is necessary and trusted.
3. Enable `restrict_pickle=True`.
4. Record usage, latency, errors, and traces.
5. Load-test async and streaming paths separately.
6. Use [dspy-debugging-observability](../dspy-debugging-observability/SKILL.md) for MLflow and callbacks.

## Official Documentation

- **Production guide**: https://dspy.ai/production/
- **Cache tutorial**: https://dspy.ai/tutorials/cache/
- **Saving tutorial**: https://dspy.ai/tutorials/saving/
- **Async tutorial**: https://dspy.ai/tutorials/async/
- **Streaming tutorial**: https://dspy.ai/tutorials/streaming/
