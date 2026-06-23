import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def metric(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


program = dspy.ChainOfThought("question -> answer")
trainset = [dspy.Example(question="What is the mock answer?", answer="mocked").with_inputs("question")]
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",
    num_candidates=2,
    max_bootstrapped_demos=1,
    max_labeled_demos=1,
)

print("OK")
