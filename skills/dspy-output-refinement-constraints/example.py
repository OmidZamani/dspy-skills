import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def reward_fn(args, prediction):
    answer = getattr(prediction, "answer", "")
    return float(bool(answer) and len(answer) <= 20)


base = dspy.Predict("question -> answer")
selector = dspy.BestOfN(module=base, N=2, reward_fn=reward_fn, threshold=0.5)

print("OK")
