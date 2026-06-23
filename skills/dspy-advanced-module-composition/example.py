import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class TwoStepQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought("question -> reasoning")
        self.answer = dspy.Predict("question, reasoning -> answer")

    def forward(self, question: str):
        rationale = self.reason(question=question)
        return self.answer(question=question, reasoning=rationale.reasoning)


program = TwoStepQA()
comparison = dspy.MultiChainComparison("question -> answer", M=2)
ensemble = dspy.Ensemble(size=1)

print("OK")
