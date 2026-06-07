"""
confidence.py — Pure, dependency-free core logic for confidence-gated RAG.

Extracted from 5_rag_with_confidence.py so the decision-critical logic (score
parsing, threshold gating, metrics) can be unit-tested WITHOUT importing heavy
runtime dependencies (LangChain, FAISS, OpenAI) or hitting a network/API key.

Nothing here imports a third-party package — only the Python standard library —
so `pytest tests/` runs offline in CI in well under a second.
"""
from __future__ import annotations

import re
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # below this → fallback to "I don't know"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4                     # retrieve top 4 chunks

# First numeric token: optional sign + integer or decimal (e.g. "0.85", "80", ".7", "-0.2").
_NUMERIC = re.compile(r"-?\d*\.\d+|-?\d+")


def clamp_confidence(value: float) -> float:
    """Constrain a score to the valid [0.0, 1.0] confidence range."""
    return min(max(value, 0.0), 1.0)


def parse_confidence_score(raw: str) -> float:
    """Extract a confidence score in [0.0, 1.0] from arbitrary LLM text.

    The model is asked for a bare number, but in practice returns things like
    ``"0.8"``, ``"Score: 0.85"``, ``"0.85/1.0"``, ``"0.9."`` or ``"80%"``. The
    original implementation did ``float(text)`` and silently fell back to 0.5 on
    any of those — a real reliability bug, because a wrong confidence value flips
    the answer/escalate decision.

    Rules:
      * take the first numeric token in the string
      * if the string contains '%', treat the number as a percentage (÷100)
      * clamp the result into [0.0, 1.0]

    Raises:
      ValueError: if the input is empty/None or contains no numeric value, so the
                  caller can decide how to handle a genuinely unparseable reply.
    """
    if raw is None or not str(raw).strip():
        raise ValueError("empty confidence response")

    text = str(raw).strip()
    match = _NUMERIC.search(text)
    if not match:
        raise ValueError(f"no numeric confidence found in: {raw!r}")

    value = float(match.group())
    if "%" in text:
        value /= 100.0
    return clamp_confidence(value)


def should_answer(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Gate decision: answer autonomously only when confidence meets the threshold."""
    return confidence >= threshold


# ─── METRICS STORE ────────────────────────────────────────────────────────────
# In production: write to Azure Log Analytics / CloudWatch / Prometheus.
# Here: in-memory list for demo + observability. Pure stdlib → unit-testable.
class MetricsStore:
    def __init__(self):
        self.records: list[dict] = []

    def log(self, question: str, confidence: float, answered: bool, chunks_used: int):
        self.records.append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:80],
            "confidence": confidence,
            "answered_autonomously": answered,
            "chunks_used": chunks_used,
        })

    def summary(self) -> str:
        if not self.records:
            return "No queries yet."
        confidences = [r["confidence"] for r in self.records]
        answered = sum(1 for r in self.records if r["answered_autonomously"])
        return (
            f"Total queries: {len(self.records)} | "
            f"Avg confidence: {sum(confidences)/len(confidences):.2f} | "
            f"Autonomous answers: {answered}/{len(self.records)} | "
            f"Escalated (low confidence): {len(self.records)-answered}/{len(self.records)}"
        )

    def print_all(self):
        print("\n=== Query Metrics (Observability Log) ===")
        for r in self.records:
            status = "✓ AUTO" if r["answered_autonomously"] else "⚠ ESCALATED"
            print(f"  [{r['timestamp']}] {status} | conf={r['confidence']:.2f} | {r['question']}")
        print(f"\n{self.summary()}\n")
