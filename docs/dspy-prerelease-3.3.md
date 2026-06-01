# DSPy 3.3 Prerelease Notes

The stable baseline for this repository is DSPy `3.2.1`.

DSPy `3.3.0b1` was published on May 28, 2026. Treat it as opt-in until a stable `3.3.x` release is available.

## Notable Preview Changes

- New `dspy.ReAct` v2 behavior is available on an opt-in path.
- `dspy.RLM` continues to evolve rapidly; verify sandbox and concurrency behavior before production use.
- Optimizer and adapter behavior may change between beta and stable releases.

## Install the Preview Explicitly

```bash
pip install -U "dspy==3.3.0b1"
```

Do not widen production requirements to include prereleases accidentally. Keep stable deployments pinned to:

```bash
pip install -U "dspy>=3.2.1,<3.3"
```

## Sources

- [DSPy releases](https://github.com/stanfordnlp/dspy/releases)
- [DSPy release tags](https://github.com/stanfordnlp/dspy/tags)
