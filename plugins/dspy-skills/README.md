# DSPy Skills Collection

A comprehensive collection of AI-powered skills for programming and optimizing LLM applications using the **DSPy** framework. These skills enable you to move from manual prompt engineering to systematic, programmatic LLM development.

**Version Compatibility**: All skills target **DSPy 3.1.2** (released January 19, 2026). All code examples and APIs have been verified against this version.

## 🎯 What is DSPy?

DSPy is a declarative framework that lets you *program* language models instead of *prompting* them. It provides:
- **Modular architecture** - Compose LLM programs from reusable components
- **Automatic optimization** - Tune prompts, examples, and weights algorithmically
- **Self-improving pipelines** - Systems that get better with data

## 📚 Skills Index

### Optimizers
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-bootstrap-fewshot](skills/dspy-bootstrap-fewshot/SKILL.md) | Auto-generate few-shot examples | Quick optimization with ~10 examples |
| [dspy-miprov2-optimizer](skills/dspy-miprov2-optimizer/SKILL.md) | Bayesian instruction+demo optimization | 200+ examples, comprehensive tuning |
| [dspy-gepa-reflective](skills/dspy-gepa-reflective/SKILL.md) | LLM reflection on execution traces | Agentic systems, complex workflows |
| [dspy-simba-optimizer](skills/dspy-simba-optimizer/SKILL.md) | Mini-batch Bayesian optimization | Custom feedback, budget-friendly |
| [dspy-finetune-bootstrap](skills/dspy-finetune-bootstrap/SKILL.md) | Fine-tune model weights | Production deployment, efficiency |

### Pipelines & Components
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-rag-pipeline](skills/dspy-rag-pipeline/SKILL.md) | RAG with ColBERTv2 retrieval | Knowledge-grounded generation |
| [dspy-signature-designer](skills/dspy-signature-designer/SKILL.md) | Design type-safe I/O specs | Clean, validated outputs |
| [dspy-evaluation-suite](skills/dspy-evaluation-suite/SKILL.md) | Metrics and evaluation | Quality assessment |
| [dspy-haystack-integration](skills/dspy-haystack-integration/SKILL.md) | DSPy + Haystack pipelines | Existing Haystack projects |

### Agent Development
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-react-agent-builder](skills/dspy-react-agent-builder/SKILL.md) | Build ReAct agents with tools | Multi-step reasoning tasks |

### Output Validation
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-output-refinement-constraints](skills/dspy-output-refinement-constraints/SKILL.md) | Refine outputs with constraints | Format/content validation |

### Advanced Patterns
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-advanced-module-composition](skills/dspy-advanced-module-composition/SKILL.md) | Ensemble and multi-chain patterns | Complex multi-module programs |
| [dspy-custom-module-design](skills/dspy-custom-module-design/SKILL.md) | Build custom DSPy modules | Reusable production components |

### Debugging & Monitoring
| Skill | Description | Best For |
|-------|-------------|----------|
| [dspy-debugging-observability](skills/dspy-debugging-observability/SKILL.md) | MLflow tracing and monitoring | Production debugging, cost tracking |

## 🎯 Choosing the Right Optimizer

| Your Situation | Recommended Skill | Why |
|----------------|-------------------|-----|
| 10-50 labeled examples | [dspy-bootstrap-fewshot](skills/dspy-bootstrap-fewshot/SKILL.md) | Fast, cost-effective |
| 200+ examples, need best performance | [dspy-miprov2-optimizer](skills/dspy-miprov2-optimizer/SKILL.md) | State-of-the-art results |
| Building agents with tools | [dspy-gepa-reflective](skills/dspy-gepa-reflective/SKILL.md) | Optimizes execution traces |
| Production deployment, cost reduction | [dspy-finetune-bootstrap](skills/dspy-finetune-bootstrap/SKILL.md) | Creates efficient fine-tuned models |

## 🔄 Typical Workflow

1. **Design** → [Signature Designer](skills/dspy-signature-designer/SKILL.md) - Define inputs/outputs
2. **Build** → [RAG Pipeline](skills/dspy-rag-pipeline/SKILL.md) or [Agent Builder](skills/dspy-react-agent-builder/SKILL.md) - Create your DSPy program
3. **Validate** → [Output Refinement](skills/dspy-output-refinement-constraints/SKILL.md) - Add constraints
4. **Optimize** → Choose optimizer based on your data
5. **Evaluate** → [Evaluation Suite](skills/dspy-evaluation-suite/SKILL.md) - Measure improvements
6. **Debug** → [Debugging & Observability](skills/dspy-debugging-observability/SKILL.md) - Monitor performance
7. **Deploy** → [Fine-tune Bootstrap](skills/dspy-finetune-bootstrap/SKILL.md) - Optional production optimization

## 📖 Documentation

- [DSPy Framework Guide](docs/dspy-framework.md) - Complete framework reference

## 🚀 Installation

```bash
# Install DSPy 3.1.2 or later
pip install dspy-ai>=3.1.2
```

### Optional Dependencies

```bash
# For ColBERTv2 retrieval
pip install colbert-ai

# For Haystack integration
pip install haystack-ai

# For fine-tuning
pip install transformers datasets
```

## 💡 Quick Start

```python
import dspy

# Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Create a simple classifier
classify = dspy.Predict("text -> sentiment: bool")
result = classify(text="I love this product!")
print(result.sentiment)  # True
```

## 📁 Examples

See [examples/code-snippets.py](examples/code-snippets.py) for production-ready code.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built for the [SkillsMP marketplace](https://github.com/skills-mp). Uses the [DSPy framework](https://dspy.ai/) by Stanford NLP.
