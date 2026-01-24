"""
Production-ready Haystack + DSPy optimizer.

This class wraps Haystack retrievers in DSPy modules, optimizes them,
and exports optimized prompts back to Haystack pipelines.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
import logging

logger = logging.getLogger(__name__)


class HaystackDSPyOptimizer:
    """Optimize Haystack pipelines using DSPy."""

    def __init__(self, document_store, lm_model="openai/gpt-4o-mini"):
        self.doc_store = document_store
        self.retriever = InMemoryBM25Retriever(document_store=document_store)
        lm = dspy.LM(lm_model)
        dspy.configure(lm=lm)

    def create_dspy_module(self, k=3):
        """Create DSPy module wrapping Haystack retriever."""

        class RAGModule(dspy.Module):
            def __init__(inner_self):
                super().__init__()
                inner_self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(inner_self, question):
                results = self.retriever.run(query=question, top_k=k)
                context = [doc.content for doc in results.get('documents', [])]

                if not context:
                    return dspy.Prediction(context=[], answer="No relevant information found.")

                pred = inner_self.generate(context=context, question=question)
                return dspy.Prediction(context=context, answer=pred.answer)

        return RAGModule()

    def optimize(self, trainset, metric=None):
        """Run DSPy optimization."""

        metric = metric or (lambda ex, pred, trace=None:
                           ex.answer.lower() in pred.answer.lower())

        module = self.create_dspy_module()

        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )

        compiled = optimizer.compile(module, trainset=trainset)
        logger.info("DSPy optimization complete")

        return compiled

    def build_optimized_pipeline(self, compiled_module):
        """Create Haystack pipeline with DSPy-optimized prompt."""

        # Extract demos from compiled module
        demos = getattr(compiled_module.generate, 'demos', [])

        # Build optimized prompt template
        prompt_lines = ["Answer the question using the context provided.\n"]

        for i, demo in enumerate(demos[:4]):  # Limit to 4 examples
            prompt_lines.append(f"Example {i+1}:")
            prompt_lines.append(f"Context: {demo.context[:500]}...")  # Truncate
            prompt_lines.append(f"Question: {demo.question}")
            prompt_lines.append(f"Answer: {demo.answer}\n")

        prompt_lines.extend([
            "Now answer this question:",
            "Context: {{context}}",
            "Question: {{question}}",
            "Answer:"
        ])

        optimized_prompt = "\n".join(prompt_lines)

        # Build pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=self.doc_store))
        pipeline.add_component("prompt_builder", PromptBuilder(template=optimized_prompt))
        pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

        pipeline.connect("retriever", "prompt_builder.context")
        pipeline.connect("prompt_builder", "generator")

        return pipeline


# Usage example
if __name__ == "__main__":
    # Setup
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents([
        Document(content="Photosynthesis is the process plants use to convert light into energy."),
        Document(content="Python is a high-level programming language."),
    ])

    trainset = [
        dspy.Example(question="What is photosynthesis?", answer="Process plants use to convert light to energy").with_inputs("question"),
    ]

    # Optimize
    optimizer = HaystackDSPyOptimizer(doc_store)
    compiled = optimizer.optimize(trainset)
    pipeline = optimizer.build_optimized_pipeline(compiled)

    # Run
    result = pipeline.run({"retriever": {"query": "What is photosynthesis?"}})
    print(result)
