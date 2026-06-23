import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class SimpleRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=1)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str):
        passages = self.retrieve(question).passages
        return self.generate(context=passages, question=question)


program = SimpleRAG()

print("OK")
