import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def lookup(topic: str) -> str:
    return f"{topic}: mocked"


tool = dspy.Tool(lookup, name="lookup", desc="Look up a mocked fact.")
agent = dspy.ReAct("question -> answer", tools=[tool], max_iters=1)

print("OK")
