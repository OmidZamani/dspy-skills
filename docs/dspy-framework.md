# DSPy Framework Documentation

Comprehensive reference for the DSPy framework - a declarative system for programming and optimizing LLM applications.

**Stable baseline:** DSPy `3.2.1`. See [DSPy 3.3 prerelease notes](dspy-prerelease-3.3.md) before opting into beta releases.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Signatures](#signatures)
- [Modules](#modules)
- [Adapters](#adapters)
- [Optimizers](#optimizers)
- [Retrieval & RAG](#retrieval--rag)
- [Evaluation](#evaluation)
- [Production Runtime](#production-runtime)

---

## Core Concepts

DSPy abstracts LLM interaction into three layers:

1. **Signatures** - Declarative I/O specifications
2. **Modules** - Composable program building blocks
3. **Adapters** - Bridge between DSPy and LLM APIs

```python
import dspy

# Configure the LM globally
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

---

## Signatures

A **Signature** defines *what* a module does, not *how*. DSPy compiles it into an effective prompt.

### Inline Signatures

```python
# Simple
"question -> answer"

# With types
"sentence -> sentiment: bool"

# Multiple fields
"context: list[str], question: str -> answer: str"
```

### Class-based Signatures

```python
from typing import Literal
import dspy

class Emotion(dspy.Signature):
    """Classify the emotion of the sentence."""
    sentence: str = dspy.InputField(desc="The text to classify.")
    sentiment: Literal['sadness','joy','love','anger','fear','surprise'] = dspy.OutputField()
```

### Type Hints

Supported types:
- Basic: `str`, `int`, `float`, `bool`
- Collections: `list[str]`, `dict[str, int]`
- Optional: `Optional[str]`
- Literals: `Literal['a', 'b', 'c']`
- Pydantic models for complex structures

---

## Modules

Modules are composable building blocks with learnable parameters.

### dspy.Predict

Basic prediction without reasoning:

```python
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence="It's a charming journey.")
print(response.sentiment)  # True
```

### dspy.ChainOfThought

Adds step-by-step reasoning:

```python
qa = dspy.ChainOfThought('question -> answer')
response = qa(question="What's special about ColBERT?")
print(response.reasoning)  # Step-by-step explanation
print(response.answer)
```

### dspy.ReAct

Agent with tool use:

```python
def search_wiki(query: str) -> str:
    """Search Wikipedia for information."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return results[0]['text']

def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    with dspy.PythonInterpreter() as interpreter:
        return interpreter.execute(expression)

agent = dspy.ReAct("question -> answer: float", tools=[search_wiki, calculate])
result = agent(question="What is 100 divided by the year Columbus discovered America?")
```

### dspy.RLM

Explore very large contexts through a sandboxed Python REPL and recursive sub-LM calls:

```python
rlm = dspy.RLM(
    "document, question -> answer",
    max_iterations=12,
    max_llm_calls=30,
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)
result = rlm(document=very_long_document, question="What are the key findings?")
```

`RLM`, `ProgramOfThought`, and `CodeAct` use `dspy.PythonInterpreter` by default and require Deno. Treat `RLM` as experimental and keep sandbox permissions narrow.

---

## Adapters

Adapters translate between DSPy structures and LLM APIs.

### dspy.ChatAdapter (Default)

- Human-readable marker format: `[[ ## field_name ## ]]`
- Auto-fallback to JSONAdapter on failure
- Good for debugging

### dspy.JSONAdapter

- Native JSON output mode
- Lower latency, more reliable parsing
- Best for structured data

```python
# Explicitly set adapter
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter()
)
```

### dspy.XMLAdapter

Use XML tags when they are easier for the selected LM to follow:

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.XMLAdapter(),
)
```

### Native Function Calling

`JSONAdapter` enables native function calling by default. `ChatAdapter` keeps text parsing by default. DSPy falls back to manual parsing when the LM does not support native tool calling.

```python
dspy.ChatAdapter(use_native_function_calling=True)
dspy.JSONAdapter(use_native_function_calling=False)
```

### Multimodal Inputs

Use typed signature inputs such as `dspy.Image`, `dspy.Audio`, and `dspy.File`:

```python
class SummarizeFile(dspy.Signature):
    file: dspy.File = dspy.InputField()
    summary: str = dspy.OutputField()

result = dspy.Predict(SummarizeFile)(file=dspy.File.from_path("./report.pdf"))
```

---

## Optimizers

### Few-Shot Learning Optimizers

#### LabeledFewShot

Simplest optimizer - random selection from training set.

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(k=3)  # Use 3 examples
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `k` | Number of examples to include |

#### BootstrapFewShot

Generates demonstrations using a teacher model.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=dspy.evaluate.answer_exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    teacher_settings={'lm': dspy.LM("openai/gpt-4o")}
)
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Evaluation function | Required |
| `max_bootstrapped_demos` | Max generated demos | 4 |
| `max_labeled_demos` | Max labeled demos | 16 |
| `teacher_settings` | Teacher LM config | None |

#### BootstrapFewShotWithRandomSearch

Extends Bootstrap with random search over demo sets.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `num_candidate_programs` | Candidates to evaluate | 16 |
| `num_threads` | Parallel threads | 6 |

#### KNNFewShot

Dynamic example selection via k-NN.

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(k=3, trainset=trainset)
```

### Instruction Optimizers

#### COPRO

Iterative instruction refinement.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=metric,
    breadth=10,  # Initial candidates
    depth=3,     # Refinement iterations
    prompt_model=dspy.LM("openai/gpt-4o")
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `breadth` | Initial candidates | 10 |
| `depth` | Refinement iterations | 3 |
| `prompt_model` | LM for generation | Default LM |

#### MIPROv2

State-of-the-art: Bayesian optimization of instructions + demos.

```python
import dspy

optimizer = dspy.MIPROv2(
    metric=dspy.evaluate.answer_exact_match,
    auto="medium",  # "light", "medium", "heavy"
    num_threads=24
)
compiled = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Evaluation metric | Required |
| `auto` | Preset config | None |
| `num_trials` | Optimization trials | Varies |
| `num_candidates` | Candidates per trial | 10 |
| `max_bootstrapped_demos` | Set 0 for instruction-only | 4 |
| `max_labeled_demos` | Labeled examples | 16 |
| `prompt_model` | LM for proposals | Default |
| `task_model` | LM for execution | Default |

**Auto presets:**
- `"light"`: ~10 trials, quick optimization
- `"medium"`: ~40 trials, balanced
- `"heavy"`: ~100+ trials, thorough

**Best for:** 200+ training examples, comprehensive tuning.

#### SIMBA

Identifies hard examples and generates improvement rules.

```python
from dspy.teleprompt import SIMBA

optimizer = SIMBA(metric=metric)
```

#### GEPA (Genetic-Pareto)

Newest optimizer: LLM reflection on full execution traces.

```python
import dspy

# GEPA requires a five-argument feedback metric
def gepa_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    is_correct = example.answer.lower() in pred.answer.lower()
    feedback = "Correct answer" if is_correct else f"Expected {example.answer}, got {pred.answer}"
    return dspy.Prediction(score=float(is_correct), feedback=feedback)

optimizer = dspy.GEPA(
    metric=gepa_metric,
    reflection_lm=dspy.LM("openai/gpt-4o"),
    auto="medium"
)
```

| Parameter | Description | Default |
|-----------|-------------|--------|
| `metric` | Accept five arguments; return `dspy.Prediction(score=..., feedback=...)` for textual feedback | Required |
| `reflection_lm` | Strong LM for reflection | Default LM |
| `auto` | "light", "medium", "heavy" | None |

**Best for:** Complex agentic systems, when you have rich textual feedback.

### Fine-tuning Optimizer

#### BootstrapFinetune

Distill a DSPy program into fine-tuned weights.

```python
import dspy

optimizer = dspy.BootstrapFinetune(
    metric=lambda gold, pred, trace=None: gold.label == pred.label,
    train_kwargs={'learning_rate': 5e-5, 'num_train_epochs': 3},
)
program.set_lm(dspy.settings.lm)
finetuned = optimizer.compile(program, trainset=trainset)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Validation metric (optional) |
| `train_kwargs` | Training hyperparameters |

### Ensemble Optimizer

Combine multiple programs.

```python
from dspy.teleprompt import Ensemble

ensembled = Ensemble(reduce_fn=dspy.majority).compile([prog1, prog2, prog3])
```

### BetterTogether Meta-Optimizer

Sequence arbitrary optimizers and return the best validated candidate:

```python
student.set_lm(lm)
optimizer = dspy.BetterTogether(
    metric=metric,
    p=dspy.GEPA(metric=gepa_metric, reflection_lm=reflection_lm, auto="light"),
    w=dspy.BootstrapFinetune(metric=metric),
)
compiled = optimizer.compile(student, trainset=trainset, valset=valset, strategy="p -> w -> p")
```

---

## Retrieval & RAG

### ColBERTv2 Setup

```python
import dspy

colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), rm=colbert)
```

### Local Embedding Retrieval

Use `dspy.Embedder` and `dspy.Embeddings` for an application-owned text corpus:

```python
embedder = dspy.Embedder("openai/text-embedding-3-small")
search = dspy.Embeddings(corpus=corpus, embedder=embedder, k=5)
context = search("Which documents discuss deployment?").passages
```

For large corpora, install `faiss-cpu`. Persist expensive indexes with `search.save(path)` and reload them with `dspy.Embeddings.from_saved(path, embedder)`.

### RAG Module Pattern

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

---

## Evaluation

### Evaluate Class

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=8,
    display_progress=True
)
score = evaluator(program)
```

### Built-in Metrics

```python
# Exact match (normalized, case-insensitive)
dspy.evaluate.answer_exact_match

# F1 token overlap
dspy.evaluate.answer_passage_match
```

### SemanticF1

LLM-based semantic evaluation:

```python
from dspy.evaluate import SemanticF1

semantic_metric = SemanticF1()
score = semantic_metric(example, prediction)
```

### Custom Metrics

```python
def my_metric(example, pred, trace=None):
    """Returns bool, int, or float."""
    return example.answer.lower() == pred.answer.lower()
```

---

## Production Runtime

### Cache Hardening

```python
dspy.configure_cache(restrict_pickle=True)
```

Use `safe_types=[...]` only for trusted application types that must be read from the disk cache.

### State-Only JSON Save

```python
compiled.save("./artifacts/program.json", save_program=False)
loaded = MyProgram()
loaded.load("./artifacts/program.json")
```

Whole-program save uses cloudpickle. Load whole programs only from trusted sources.

### Usage Tracking

```python
dspy.configure(lm=lm, track_usage=True)
prediction = program(question="What is DSPy?")
print(prediction.get_lm_usage())
```

### Async and Streaming

Use `await program.acall(...)` for native async execution. Wrap output-token streaming with `dspy.streamify` and `dspy.streaming.StreamListener`.

### Further Reading

- [Production deployment skill](../skills/dspy-production-deployment/SKILL.md)
- [Adapters and multimodal skill](../skills/dspy-adapters-multimodal/SKILL.md)
- [Reasoning modules skill](../skills/dspy-reasoning-modules/SKILL.md)
- [MCP tool integration skill](../skills/dspy-mcp-tool-integration/SKILL.md)

---

## Best Practices

1. **Start simple** - Use `dspy.Predict` before `ChainOfThought`
2. **Measure first** - Establish baseline metrics before optimizing
3. **Small data works** - BootstrapFewShot works with ~10 examples
4. **Match optimizer to data** - MIPROv2 for 200+, Bootstrap for fewer
5. **Use strong teachers** - GPT-4 as teacher, GPT-3.5 as student
6. **Save compiled programs** - `program.save('optimized.json')`
