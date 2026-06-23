import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def local_embedder(texts):
    return [[float(len(text)), float(text.count("dspy"))] for text in texts]


corpus = ["dspy builds modular AI programs", "retrieval supplies context"]
embedder = dspy.Embedder(local_embedder)
search = dspy.Embeddings(corpus=corpus, embedder=embedder, k=1, cache=False)

print("OK")
