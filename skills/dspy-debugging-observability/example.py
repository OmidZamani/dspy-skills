import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))

program = dspy.Predict("question -> answer")
history_before_run = list(getattr(dspy.settings.lm, "history", []))

print("OK")
