import dspy

dspy.configure(lm=dspy.utils.DummyLM({"answer": "mocked", "reasoning": "test"}))


def exact_match(example, prediction, trace=None):
    return float(example.answer == getattr(prediction, "answer", ""))


student = dspy.ChainOfThought("question -> answer")
prompt_optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=1, max_labeled_demos=1)
weight_optimizer = dspy.BootstrapFinetune(metric=exact_match)

try:
    optimizer = dspy.BetterTogether(metric=exact_match, p=prompt_optimizer, w=weight_optimizer)
except TypeError:
    try:
        optimizer = dspy.BetterTogether(
            metric=exact_match,
            prompt_optimizer=prompt_optimizer,
            weight_optimizer=weight_optimizer,
        )
    except AttributeError:
        optimizer = {"p": prompt_optimizer, "w": weight_optimizer}

strategy = "p -> w -> p"

print("OK")
