import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def exact_match(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


program = dspy.Predict("question -> answer")
devset = [dspy.Example(question="What should this return?", answer="mocked").with_inputs("question")]
evaluator = dspy.Evaluate(devset=devset, metric=exact_match, display_progress=False, display_table=False)

print("OK")
