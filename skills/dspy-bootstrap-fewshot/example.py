import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def metric(example, prediction, trace=None):
    return example.answer == getattr(prediction, "answer", "")


program = dspy.Predict("question -> answer")
trainset = [dspy.Example(question="What is returned?", answer="mocked").with_inputs("question")]
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1, max_labeled_demos=1)

print("OK")
