import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class QuestionAnswer(dspy.Signature):
    question: str = dspy.InputField(desc="Question to answer.")
    answer: str = dspy.OutputField(desc="Short answer.")


program = dspy.Predict(QuestionAnswer)

print("OK")
