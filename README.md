# DSPy Skills Collection

A Claude Code plugin containing 22 focused skills for programming, optimizing, evaluating, and deploying LLM applications with [DSPy](https://dspy.ai/).

**Stable DSPy baseline:** `3.2.1`, released May 5, 2026. Install the stable series with:

```bash
pip install -U "dspy>=3.2.1,<3.3"
```

DSPy `3.3.0b1` is a prerelease published May 28, 2026. See [prerelease notes](docs/dspy-prerelease-3.3.md) before opting in.

## Skills Index

### Start Here

| Skill | Use it for |
|-------|------------|
| [dspy-signature-designer](skills/dspy-signature-designer/SKILL.md) | Typed signatures and structured outputs |
| [dspy-custom-module-design](skills/dspy-custom-module-design/SKILL.md) | Reusable `dspy.Module` architecture |
| [dspy-evaluation-suite](skills/dspy-evaluation-suite/SKILL.md) | Metrics, baselines, and comparisons |
| [dspy-optimizer-selection](skills/dspy-optimizer-selection/SKILL.md) | Choosing among DSPy's optimizer families |

### Retrieval and Data

| Skill | Use it for |
|-------|------------|
| [dspy-rag-pipeline](skills/dspy-rag-pipeline/SKILL.md) | RAG and multi-hop retrieval pipelines |
| [dspy-embedding-retrieval](skills/dspy-embedding-retrieval/SKILL.md) | `Embedder`, `Embeddings`, FAISS, and local corpora |
| [dspy-haystack-integration](skills/dspy-haystack-integration/SKILL.md) | Optimizing Haystack-backed pipelines |

### Agents and Reasoning

| Skill | Use it for |
|-------|------------|
| [dspy-react-agent-builder](skills/dspy-react-agent-builder/SKILL.md) | ReAct tool agents |
| [dspy-mcp-tool-integration](skills/dspy-mcp-tool-integration/SKILL.md) | MCP tools in DSPy agents |
| [dspy-reasoning-modules](skills/dspy-reasoning-modules/SKILL.md) | `RLM`, `ProgramOfThought`, `CodeAct`, and `Parallel` |
| [dspy-advanced-module-composition](skills/dspy-advanced-module-composition/SKILL.md) | Ensembles and composed modules |

### Optimizers

| Skill | Use it for |
|-------|------------|
| [dspy-bootstrap-fewshot](skills/dspy-bootstrap-fewshot/SKILL.md) | Fast demo bootstrapping |
| [dspy-miprov2-optimizer](skills/dspy-miprov2-optimizer/SKILL.md) | Instruction and demo search |
| [dspy-gepa-reflective](skills/dspy-gepa-reflective/SKILL.md) | Reflective instruction optimization |
| [dspy-simba-optimizer](skills/dspy-simba-optimizer/SKILL.md) | Mini-batch introspective optimization |
| [dspy-finetune-bootstrap](skills/dspy-finetune-bootstrap/SKILL.md) | Weight optimization and distillation |
| [dspy-better-together](skills/dspy-better-together/SKILL.md) | Sequencing prompt and weight optimizers |
| [dspy-optimize-anything](skills/dspy-optimize-anything/SKILL.md) | GEPA optimization for text artifacts outside DSPy |

### Production and Interfaces

| Skill | Use it for |
|-------|------------|
| [dspy-output-refinement-constraints](skills/dspy-output-refinement-constraints/SKILL.md) | `Refine` and `BestOfN` constraints |
| [dspy-adapters-multimodal](skills/dspy-adapters-multimodal/SKILL.md) | Adapters, native tools, image, audio, and file inputs |
| [dspy-production-deployment](skills/dspy-production-deployment/SKILL.md) | Cache hardening, save/load, usage tracking, async, and streaming |
| [dspy-debugging-observability](skills/dspy-debugging-observability/SKILL.md) | Inspection, callbacks, and MLflow tracing |

## Typical Workflow

1. Define typed inputs and outputs with [dspy-signature-designer](skills/dspy-signature-designer/SKILL.md).
2. Build a module, RAG pipeline, or agent.
3. Establish a baseline with [dspy-evaluation-suite](skills/dspy-evaluation-suite/SKILL.md).
4. Choose an optimizer with [dspy-optimizer-selection](skills/dspy-optimizer-selection/SKILL.md).
5. Harden the runtime with [dspy-production-deployment](skills/dspy-production-deployment/SKILL.md).

## Optional Dependencies

```bash
# Bayesian search for MIPROv2
pip install -U "dspy[optuna]>=3.2.1,<3.3"

# MCP tool integration
pip install -U "dspy[mcp]>=3.2.1,<3.3"

# Large local embedding corpora
pip install faiss-cpu

# optimize_anything
pip install -U "gepa>=0.1.1,<0.2"
```

## Install as a Claude Code Plugin

```text
/plugin marketplace add OmidZamani/dspy-skills
/plugin install dspy-skills@dspy-skills-marketplace
```

## Documentation

- [DSPy framework guide](docs/dspy-framework.md)
- [DSPy official documentation](https://dspy.ai/)
- [DSPy releases](https://github.com/stanfordnlp/dspy/releases)

## Validate Changes

```bash
python3 scripts/validate_repo.py
python3 -m compileall -q examples skills/dspy-haystack-integration/examples
```

## License

MIT License. See [LICENSE](LICENSE).
