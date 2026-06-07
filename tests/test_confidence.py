"""
Unit tests for the confidence-gating core (confidence.py).

These are deterministic and offline — no API key, no network, no heavy deps —
so they run in CI in milliseconds. They lock in the reliability fix for
parse_confidence_score (the original float() parse silently fell back to 0.5
on common LLM reply formats, flipping the answer/escalate decision).

Run:  pytest -q
"""
import math

import pytest

from confidence import (
    CONFIDENCE_THRESHOLD,
    clamp_confidence,
    parse_confidence_score,
    should_answer,
    MetricsStore,
)


# ─── parse_confidence_score: the core reliability fix ───────────────────────────
class TestParseConfidenceScore:
    @pytest.mark.parametrize("raw, expected", [
        ("0.8", 0.8),               # bare float (the only case the old code handled)
        ("0.85", 0.85),
        ("1.0", 1.0),
        ("0", 0.0),
        (".7", 0.7),                # leading-dot decimal
        ("  0.9  ", 0.9),           # surrounding whitespace
        ("Score: 0.85", 0.85),      # labelled — old code → 0.5 (BUG)
        ("0.85/1.0", 0.85),         # "out of 1.0" — old code → 0.5 (BUG)
        ("0.9.", 0.9),              # trailing period — old code → 0.5 (BUG)
        ("Confidence is 0.65 here", 0.65),
    ])
    def test_extracts_score_from_real_llm_formats(self, raw, expected):
        assert parse_confidence_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw, expected", [
        ("80%", 0.8),               # percentage form
        ("It's about 95%", 0.95),
        ("100%", 1.0),
    ])
    def test_handles_percentages(self, raw, expected):
        assert parse_confidence_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw, expected", [
        ("1.5", 1.0),               # above range → clamp to 1.0
        ("-0.2", 0.0),              # below range → clamp to 0.0
        ("the score is -3", 0.0),
    ])
    def test_clamps_out_of_range(self, raw, expected):
        assert parse_confidence_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["", "   ", None, "no number here", "high confidence"])
    def test_raises_on_unparseable(self, bad):
        with pytest.raises(ValueError):
            parse_confidence_score(bad)

    def test_result_always_in_unit_interval(self):
        for raw in ["0.3", "80%", "1.9", "-5", "0.999"]:
            score = parse_confidence_score(raw)
            assert 0.0 <= score <= 1.0


# ─── clamp_confidence ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("value, expected", [
    (-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.5, 1.0),
])
def test_clamp_confidence(value, expected):
    assert clamp_confidence(value) == expected


# ─── should_answer: the gating decision ─────────────────────────────────────────
class TestShouldAnswer:
    def test_at_threshold_answers(self):
        assert should_answer(CONFIDENCE_THRESHOLD) is True   # boundary is inclusive

    def test_above_threshold_answers(self):
        assert should_answer(0.95) is True

    def test_below_threshold_escalates(self):
        assert should_answer(0.69) is False

    def test_custom_threshold(self):
        assert should_answer(0.5, threshold=0.4) is True
        assert should_answer(0.3, threshold=0.4) is False


# ─── MetricsStore: observability log ────────────────────────────────────────────
class TestMetricsStore:
    def test_empty_summary(self):
        assert MetricsStore().summary() == "No queries yet."

    def test_logs_and_counts(self):
        store = MetricsStore()
        store.log("q1", 0.9, answered=True, chunks_used=4)
        store.log("q2", 0.5, answered=False, chunks_used=2)
        assert len(store.records) == 2
        summary = store.summary()
        assert "Total queries: 2" in summary
        assert "Autonomous answers: 1/2" in summary
        assert "Escalated (low confidence): 1/2" in summary

    def test_avg_confidence(self):
        store = MetricsStore()
        store.log("a", 0.6, True, 3)
        store.log("b", 0.8, True, 3)
        assert "Avg confidence: 0.70" in store.summary()

    def test_question_is_truncated(self):
        store = MetricsStore()
        store.log("x" * 200, 0.9, True, 1)
        assert len(store.records[0]["question"]) == 80
