from safety_circuits.refusal import regex_refusal


def test_refuses_canonical_phrases():
    assert regex_refusal("I cannot help with that request.")
    assert regex_refusal("I'm sorry, but I won't.")
    assert regex_refusal("As an AI, I must decline.")
    assert regex_refusal("Unfortunately, this isn't something I can do.")


def test_does_not_refuse_compliance():
    assert not regex_refusal("Sure, here is the recipe for chocolate cake.")
    assert not regex_refusal("The capital of France is Paris.")
