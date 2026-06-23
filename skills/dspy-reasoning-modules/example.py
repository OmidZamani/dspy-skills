import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))

chain = dspy.ChainOfThought("question -> answer")
program_of_thought = dspy.ProgramOfThought("question -> answer")
parallel_runner = dspy.Parallel(num_threads=1)

print("OK")
