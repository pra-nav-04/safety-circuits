from safety_circuits.data import Prompt, build_matched_pairs


def _p(text: str, cat: str = "harm") -> Prompt:
    return Prompt(text=text, category=cat, source="test", expected_refusal=(cat == "harm"))


def test_pairs_length_matched():
    harm = [_p("a" * n) for n in (10, 50, 200)]
    safe = [_p("b" * n, cat="safe") for n in (9, 48, 195, 400)]
    pairs = build_matched_pairs(harm, safe, n_pairs=3, max_len_chars=500)
    assert len(pairs) == 3
    # each safe should be the closest-length unused safe
    for h, s in pairs:
        assert abs(len(h.text) - len(s.text)) < 20


def test_max_len_filter():
    harm = [_p("a" * 1000)]
    safe = [_p("b" * 50, cat="safe")]
    assert build_matched_pairs(harm, safe, n_pairs=1, max_len_chars=400) == []
