import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class HaystackPromptBridge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("documents, question -> answer")

    def forward(self, documents: list[str], question: str):
        return self.generate(documents=documents, question=question)


program = HaystackPromptBridge()
optimized_prompt_template = "Use the provided documents to answer the question."

print("OK")
