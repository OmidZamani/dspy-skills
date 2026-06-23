import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def metric(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


program = dspy.Predict("question -> answer")
trainset = [dspy.Example(question="What is the mock answer?", answer="mocked").with_inputs("question")]
optimizer = dspy.SIMBA(metric=metric, bsize=1, num_candidates=2, max_steps=1)

print("OK")
