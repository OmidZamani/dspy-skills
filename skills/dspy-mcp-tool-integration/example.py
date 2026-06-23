import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def local_status(topic: str) -> str:
    return f"{topic}: mocked"


tool = dspy.Tool(local_status, name="local_status", desc="Return a mocked status for a topic.")
agent = dspy.ReAct("question -> answer", tools=[tool], max_iters=1)

print("OK")
