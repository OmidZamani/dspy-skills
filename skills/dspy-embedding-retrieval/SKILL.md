---
name: dspy-embedding-retrieval
version: "2.0.0"
dspy-compatibility: "3.2.1"
description: This skill should be used when the user asks to "build local DSPy retrieval", "use dspy.Embedder", "use dspy.Embeddings", "save an embeddings index", "add FAISS retrieval", mentions semantic search, hosted embeddings, local embedding models, `EmbeddingsWithScores`, or needs a DSPy retriever over an application-owned text corpus.
allowed-tools:
  - Read
  - Write
  - Glob
  - Grep
---

# DSPy Embedding Retrieval

## Goal

Build semantic retrieval over an application-owned text corpus with `dspy.Embedder` and `dspy.Embeddings`.

## Basic Hosted Embedder

```python
import dspy

corpus = [
    "DSPy programs are composed from modules.",
    "MIPROv2 optimizes instructions and demonstrations.",
    "RLM explores large contexts with a sandboxed REPL.",
]

embedder = dspy.Embedder("openai/text-embedding-3-small")
search = dspy.Embeddings(corpus=corpus, embedder=embedder, k=2)

result = search("Which optimizer tunes prompts?")
print(result.passages)
print(result.indices)
```

## Use in RAG

```python
class LocalRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

    def forward(self, question: str):
        context = self.retriever(question).passages
        return self.answer(context=context, question=question)
```

## Custom Local Embeddings

Wrap any callable that accepts `list[str]` and returns a 2D numeric array:

```python
from sentence_transformers import SentenceTransformer
import dspy

model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
embedder = dspy.Embedder(model.encode)
search = dspy.Embeddings(corpus=corpus, embedder=embedder, k=5)
```

## Scores, FAISS, and Persistence

Use `dspy.EmbeddingsWithScores` when downstream logic needs similarity thresholds or reranking.

For corpora at or above the `brute_force_threshold` default of `20_000`, DSPy builds a FAISS index. Install FAISS first:

```bash
pip install faiss-cpu
```

Persist the index when embedding the corpus is expensive:

```python
search.save("./retrieval-index")
loaded = dspy.Embeddings.from_saved("./retrieval-index", embedder=embedder)
```

## Related Skills

- Build a complete pipeline: [dspy-rag-pipeline](../dspy-rag-pipeline/SKILL.md)
- Design typed context fields: [dspy-signature-designer](../dspy-signature-designer/SKILL.md)
- Harden caches: [dspy-production-deployment](../dspy-production-deployment/SKILL.md)

## Best Practices

1. Evaluate retrieval quality separately from answer quality.
2. Keep corpus chunking deterministic and versioned.
3. Persist expensive indexes.
4. Use `EmbeddingsWithScores` when debugging relevance.
5. Measure memory and latency before enabling FAISS for large corpora.

## Official Documentation

- **Embedder API**: https://dspy.ai/api/models/Embedder/
- **Embeddings API**: https://dspy.ai/api/tools/Embeddings/
