# Extracting and Applying Optimized Prompts

## Phase 6: Build Optimized Haystack Pipeline

After optimizing your DSPy module, you need to extract the optimized prompt and apply it to your Haystack pipeline.

### Extract DSPy Prompt

```python
def extract_dspy_prompt(compiled_module):
    """Extract the optimized prompt from compiled DSPy module."""
    # Get the predictor's demos and instructions
    predictor = compiled_module.generate

    demos = getattr(predictor, 'demos', [])

    # Build prompt with few-shot examples
    prompt_parts = ["Answer questions using the provided context.\n"]

    for demo in demos:
        prompt_parts.append(f"Context: {demo.context}")
        prompt_parts.append(f"Question: {demo.question}")
        prompt_parts.append(f"Answer: {demo.answer}\n")

    prompt_parts.append("Context: {{context}}")
    prompt_parts.append("Question: {{question}}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)

optimized_prompt = extract_dspy_prompt(compiled)
```

### Apply to Haystack Pipeline

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# Create new pipeline with optimized prompt
optimized_pipeline = Pipeline()
optimized_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
optimized_pipeline.add_component("prompt_builder", PromptBuilder(template=optimized_prompt))
optimized_pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

optimized_pipeline.connect("retriever", "prompt_builder.context")
optimized_pipeline.connect("prompt_builder", "generator")
```

## Tips

- Truncate long context examples to stay within token limits
- Limit to 3-5 few-shot examples for best balance
- Test the Haystack pipeline thoroughly after prompt extraction
- Compare performance against baseline pipeline
