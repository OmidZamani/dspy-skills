import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def metric(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


teacher = dspy.ChainOfThought("question -> answer")
trainset = [dspy.Example(question="What is the mock answer?", answer="mocked").with_inputs("question")]
optimizer = dspy.BootstrapFinetune(metric=metric, train_kwargs={"epochs": 1})

print("OK")
