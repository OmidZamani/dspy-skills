import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


class DescribeImage(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()


adapter = dspy.JSONAdapter()
program = dspy.Predict(DescribeImage)

print("OK")
