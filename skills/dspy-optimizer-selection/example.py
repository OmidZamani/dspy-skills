import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def metric(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


training_examples = [dspy.Example(question="Q", answer="mocked").with_inputs("question")]
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1, max_labeled_demos=1)
selection_reason = "BootstrapFewShot is enough for a tiny labeled set."

print("OK")
