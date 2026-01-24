"""Production-ready DSPy code examples.

This module contains comprehensive examples for all DSPy skills.
"""

import dspy
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

def configure_dspy(model: str = "openai/gpt-4o-mini", retriever_url: Optional[str] = None):
    """Configure DSPy with LM and optional retriever."""
    lm = dspy.LM(model)
    
    if retriever_url:
        rm = dspy.ColBERTv2(url=retriever_url)
        dspy.configure(lm=lm, rm=rm)
    else:
        dspy.configure(lm=lm)
    
    return lm


# =============================================================================
# SIGNATURES
# =============================================================================

class QASignature(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Brief, factual answer")


class ClassificationSignature(dspy.Signature):
    """Classify text sentiment."""
    text: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")


class RAGSignature(dspy.Signature):
    """Answer using retrieved context."""
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer grounded in context")


# =============================================================================
# MODULES
# =============================================================================

class SimpleQA(dspy.Module):
    """Basic question answering."""
    
    def __init__(self):
        self.predict = dspy.Predict(QASignature)
    
    def forward(self, question: str):
        return self.predict(question=question)


class ChainOfThoughtQA(dspy.Module):
    """QA with reasoning."""
    
    def __init__(self):
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str):
        return self.cot(question=question)


class RAGModule(dspy.Module):
    """Retrieval-Augmented Generation."""
    
    def __init__(self, num_passages: int = 3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question: str):
        try:
            context = self.retrieve(question).passages
            pred = self.generate(context=context, question=question)
            return dspy.Prediction(context=context, answer=pred.answer)
        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return dspy.Prediction(context=[], answer="Error processing question")


class ReActAgent(dspy.Module):
    """Agent with tool use."""
    
    def __init__(self, tools: list):
        self.react = dspy.ReAct("question -> answer", tools=tools)
    
    def forward(self, question: str):
        return self.react(question=question)


# =============================================================================
# TOOLS FOR AGENTS
# =============================================================================

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        results = dspy.ColBERTv2(
            url='http://20.102.90.50:2017/wiki17_abstracts'
        )(query, k=1)
        return results[0]['text'] if results else "No results found"
    except Exception as e:
        return f"Search error: {e}"


def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return dspy.PythonInterpreter({}).execute(expression)
    except Exception as e:
        return f"Calculation error: {e}"


# =============================================================================
# METRICS
# =============================================================================

def exact_match_metric(example, pred, trace=None) -> bool:
    """Check for exact answer match."""
    if not hasattr(pred, 'answer') or not pred.answer:
        return False
    return example.answer.lower().strip() == pred.answer.lower().strip()


def contains_answer_metric(example, pred, trace=None) -> float:
    """Check if prediction contains the answer."""
    if not hasattr(pred, 'answer') or not pred.answer:
        return 0.0
    return float(example.answer.lower() in pred.answer.lower())


def gepa_feedback_metric(example, pred, trace=None):
    """Metric with feedback for GEPA optimizer."""
    if not hasattr(pred, 'answer') or not pred.answer:
        return 0.0, "No answer generated"
    
    correct = example.answer.lower() in pred.answer.lower()
    
    if correct:
        return 1.0, "Correct answer provided"
    else:
        return 0.0, f"Expected '{example.answer}', got '{pred.answer}'"


# =============================================================================
# OPTIMIZATION EXAMPLES
# =============================================================================

def optimize_with_bootstrap(program, trainset, metric=None):
    """Optimize using BootstrapFewShot."""
    from dspy.teleprompt import BootstrapFewShot
    
    metric = metric or contains_answer_metric
    
    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4
    )
    
    return optimizer.compile(program, trainset=trainset)


def optimize_with_miprov2(program, trainset, auto="medium"):
    """Optimize using MIPROv2."""
    optimizer = dspy.MIPROv2(
        metric=dspy.evaluate.answer_exact_match,
        auto=auto,
        num_threads=24
    )
    
    return optimizer.compile(program, trainset=trainset)


def optimize_with_gepa(program, trainset):
    """Optimize using GEPA with reflection."""
    optimizer = dspy.GEPA(
        metric=gepa_feedback_metric,
        reflection_lm=dspy.LM("openai/gpt-4o"),
        auto="medium"
    )
    
    return optimizer.compile(program, trainset=trainset)


def finetune_program(program, trainset, output_dir="./finetuned"):
    """Fine-tune model weights."""
    optimizer = dspy.BootstrapFinetune(
        metric=exact_match_metric
    )
    
    return optimizer.compile(
        program,
        trainset=trainset,
        train_kwargs={
            'learning_rate': 5e-5,
            'num_train_epochs': 3,
            'output_dir': output_dir
        }
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_program(program, devset, metric=None, num_threads=8):
    """Evaluate program on development set."""
    from dspy.evaluate import Evaluate
    
    metric = metric or contains_answer_metric
    
    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True
    )
    
    return evaluator(program)


def compare_programs(programs: dict, devset, metric=None):
    """Compare multiple program variants."""
    results = {}
    
    for name, program in programs.items():
        score = evaluate_program(program, devset, metric)
        results[name] = score
        logger.info(f"{name}: {score:.2%}")
    
    best = max(results, key=results.get)
    logger.info(f"Best: {best} ({results[best]:.2%})")
    
    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configure
    configure_dspy()
    
    # Simple QA
    qa = SimpleQA()
    result = qa(question="What is the capital of France?")
    print(f"Answer: {result.answer}")
    
    # Chain of Thought
    cot_qa = ChainOfThoughtQA()
    result = cot_qa(question="What is 15% of 80?")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")
    
    # Agent with tools
    agent = ReActAgent(tools=[search_wikipedia, calculate])
    result = agent(question="What is the population of Paris divided by 1000?")
    print(f"Agent answer: {result.answer}")
