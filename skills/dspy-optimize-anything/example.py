import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def evaluate_artifact(prompt_text: str):
    score = float("answer" in prompt_text.lower())
    return dspy.Prediction(score=score, feedback="Keep prompts explicit about the answer field.")


artifact = "Answer the question with one concise answer."
optimizer_api = getattr(getattr(dspy, "GEPA", None), "optimize_anything", None)
optimization_ready = callable(optimizer_api)

print("OK")
