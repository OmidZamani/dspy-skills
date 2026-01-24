# DSPy Skills

A marketplace containing the **dspy-skills** plugin - a comprehensive collection of 14 skills for programming and optimizing LLM applications using the DSPy framework.

## Installation

### Option 1: Install from GitHub

```bash
# In Claude Code, run:
/plugin add OmidZamani/dspy-skills
```

### Option 2: Install Locally

```bash
# Clone the repository
git clone https://github.com/OmidZamani/dspy-skills.git
cd dspy-skills

# In Claude Code, run:
/plugin add /path/to/dspy-skills
```

### Option 3: Development Mode

For development or testing:

```bash
# Launch Claude Code with the plugin directory
cc --plugin-dir /path/to/dspy-skills
```

### Verify Installation

After installation, verify the plugin is loaded:

```bash
# List installed plugins
/plugin list

# You should see "dspy-skills" in the list
```

Skills will automatically activate when you ask questions related to DSPy optimization, RAG pipelines, agents, or evaluation.

## What's Inside

The **dspy-skills** plugin includes 14 specialized skills for DSPy 3.1.2:

### Optimizers
- **dspy-bootstrap-fewshot** - Auto-generate few-shot examples
- **dspy-miprov2-optimizer** - Bayesian optimization for 200+ examples
- **dspy-gepa-reflective** - LLM reflection on execution traces
- **dspy-simba-optimizer** - Mini-batch Bayesian optimization
- **dspy-finetune-bootstrap** - Fine-tune model weights

### Pipelines & Components
- **dspy-rag-pipeline** - RAG with ColBERTv2 retrieval
- **dspy-signature-designer** - Type-safe I/O specifications
- **dspy-evaluation-suite** - Metrics and evaluation
- **dspy-haystack-integration** - DSPy + Haystack integration

### Agents
- **dspy-react-agent-builder** - ReAct agents with tools

### Output Validation
- **dspy-output-refinement-constraints** - Output refinement

### Advanced Patterns
- **dspy-advanced-module-composition** - Ensemble patterns
- **dspy-custom-module-design** - Build custom modules

### Debugging
- **dspy-debugging-observability** - MLflow tracing and monitoring

## Documentation

See [plugins/dspy-skills/README.md](plugins/dspy-skills/README.md) for complete documentation, usage examples, and workflow guides.

## Version Compatibility

All skills target **DSPy 3.1.2** (released January 19, 2026). All code examples and APIs have been verified against this version.

## Repository Structure

```
dspy-skills/
├── .claude-plugin/
│   └── marketplace.json        # Marketplace definition
└── plugins/
    └── dspy-skills/             # The actual plugin
        ├── .claude-plugin/
        │   └── plugin.json     # Plugin manifest
        ├── skills/              # 14 DSPy skills
        ├── docs/                # Documentation
        ├── examples/            # Code examples
        ├── README.md            # Plugin documentation
        └── CONTRIBUTING.md      # Contribution guide
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [plugins/dspy-skills/CONTRIBUTING.md](plugins/dspy-skills/CONTRIBUTING.md) for contribution guidelines.

## Links

- **Repository**: https://github.com/OmidZamani/dspy-skills
- **DSPy Framework**: https://dspy.ai/
- **DSPy GitHub**: https://github.com/stanfordnlp/dspy
