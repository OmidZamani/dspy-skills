import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class AnswerWithContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context: str, question: str):
        return self.generate(context=context, question=question)


program = AnswerWithContext()

print("OK")
