"""Model-free tests for the RTP toxicity probe aggregation logic."""

from safety_circuits.toxicity import score_toxicity, RTPReport


def test_score_toxicity_picks_toxic_label():
    """score_toxicity must extract the 'toxic' label regardless of dict order."""
    def fake_pipe(texts):
        return [[
            {"label": "neutral", "score": 0.1},
            {"label": "toxic", "score": 0.9},
        ] for _ in texts]

    scores = score_toxicity(["hello", "world"], pipe=fake_pipe)
    assert len(scores) == 2
    assert all(abs(s - 0.9) < 1e-6 for s in scores)


def test_score_toxicity_missing_label_returns_zero():
    """If 'toxic' label is absent, score is 0.0 (safe default)."""
    def fake_pipe(texts):
        return [[{"label": "neutral", "score": 1.0}] for _ in texts]

    scores = score_toxicity(["x"], pipe=fake_pipe)
    assert scores == [0.0]


def test_score_toxicity_case_insensitive():
    """Label matching is case-insensitive."""
    def fake_pipe(texts):
        return [[{"label": "TOXIC", "score": 0.75}] for _ in texts]

    scores = score_toxicity(["x"], pipe=fake_pipe)
    assert abs(scores[0] - 0.75) < 1e-6


def test_rtp_report_delta():
    """delta_tox = mean_tox_ablated - mean_tox_clean."""
    rows = [
        {"clean_toxicity": 0.1, "ablated_toxicity": 0.3, "delta_toxicity": 0.2,
         "prompt": "p", "clean_continuation": "c", "ablated_continuation": "a"},
        {"clean_toxicity": 0.2, "ablated_toxicity": 0.4, "delta_toxicity": 0.2,
         "prompt": "p", "clean_continuation": "c", "ablated_continuation": "a"},
    ]
    rep = RTPReport(
        n=2,
        mean_tox_clean=0.15,
        mean_tox_ablated=0.35,
        delta_tox=0.20,
        rows=rows,
    )
    assert abs(rep.delta_tox - (rep.mean_tox_ablated - rep.mean_tox_clean)) < 1e-6
    assert rep.n == 2
