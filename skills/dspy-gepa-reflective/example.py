import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def feedback_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    is_correct = example.answer == getattr(prediction, "answer", "")
    feedback = "Correct mock answer." if is_correct else "Expected the mock answer."
    return dspy.Prediction(score=float(is_correct), feedback=feedback)


agent = dspy.ReAct("question -> answer", tools=[], max_iters=1)
gepa_cls = getattr(dspy, "GEPA", None)
optimizer = gepa_cls(metric=feedback_metric, auto="light") if gepa_cls else None

print("OK")
