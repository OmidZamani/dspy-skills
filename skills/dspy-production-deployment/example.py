import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)

program = dspy.Predict("question -> answer")
async_program = dspy.asyncify(program)

print("OK")
