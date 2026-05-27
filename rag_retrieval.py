"""
Agentic RAG Orchestrator — API Troubleshooting
Retrieval Logic: Ingest → Embed → Search → Rerank → Gate → Decide
"""

import json
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    id:             str
    text:           str
    embedding:      Optional[np.ndarray] = None
    error_category: str = ""
    fix_type:       str = ""
    risk_level:     str = "low"
    success_rate:   float = 1.0
    date:           str = ""

@dataclass
class RetrievalResult:
    chunk:          Chunk
    similarity:     float
    rerank_score:   float = 0.0

@dataclass
class AgentDecision:
    action:         str          # auto_apply | suggest | escalate
    fix:            str
    confidence:     float
    risk_level:     str
    reason:         str
    source_chunk_id: str

# ─────────────────────────────────────────────
# 2. KNOWLEDGE BASE — API ERROR FIX CORPUS
# ─────────────────────────────────────────────

RAW_FIXES = [
    {
        "id": "fix-001",
        "text": """
            Timeout error fix: Increase connection timeout from default 30s to 60s.
            Set retry policy: max 3 retries with exponential backoff (1s, 2s, 4s).
            Check downstream service latency — if P99 > 10s, scale horizontally.
            Config: requests.get(url, timeout=60)
        """,
        "error_category": "timeout",
        "fix_type": "config",
        "risk_level": "low",
        "success_rate": 0.92
    },
    {
        "id": "fix-002",
        "text": """
            Authentication failure fix: Token expiry is the most common cause.
            Refresh OAuth token before each request batch.
            Validate token TTL on startup — fail fast if token < 60s remaining.
            Check API key rotation schedule — keys expire every 90 days.
            Code: if token_expires_in(token) < 60: token = refresh_token()
        """,
        "error_category": "auth_failure",
        "fix_type": "code",
        "risk_level": "low",
        "success_rate": 0.88
    },
    {
        "id": "fix-003",
        "text": """
            Rate limit fix: Implement token bucket rate limiter client-side.
            Respect Retry-After header from 429 responses.
            Distribute requests across time window — avoid burst patterns.
            Add jitter to retry delays to prevent thundering herd.
            Target: stay below 80% of rate limit ceiling.
        """,
        "error_category": "rate_limit",
        "fix_type": "code",
        "risk_level": "low",
        "success_rate": 0.95
    },
    {
        "id": "fix-004",
        "text": """
            Payload validation error fix: Schema mismatch between client and server.
            Validate request payload against OpenAPI schema before sending.
            Common causes: missing required fields, wrong data types, extra fields.
            Enable strict mode in serializer — fail on unknown fields.
            Log exact validation error message — do not swallow the error.
        """,
        "error_category": "payload_error",
        "fix_type": "code",
        "risk_level": "low",
        "success_rate": 0.85
    },
    {
        "id": "fix-005",
        "text": """
            Service down fix: Activate circuit breaker — stop sending requests.
            Route to fallback service or cached response.
            Alert oncall immediately via PagerDuty.
            Check health endpoint: GET /api/healthz
            Do not retry aggressively — adds load to degraded service.
        """,
        "error_category": "service_down",
        "fix_type": "escalate",
        "risk_level": "high",
        "success_rate": 0.78
    },
    {
        "id": "fix-006",
        "text": """
            Connection timeout specific: DNS resolution failure or network partition.
            Verify DNS resolution: nslookup api.service.internal
            Check VPN / network routes if in corporate environment.
            Ping the host to confirm basic connectivity.
            Fallback: use IP directly if DNS is flapping.
        """,
        "error_category": "timeout",
        "fix_type": "config",
        "risk_level": "low",
        "success_rate": 0.80
    },
    {
        "id": "fix-007",
        "text": """
            JWT token signature invalid: Clock skew between client and server.
            Synchronize system clocks via NTP — max allowed skew is 5 minutes.
            Check iat (issued at) and exp (expiry) claims in token payload.
            Ensure same timezone configuration on both sides.
            Add clock_skew_leeway=300 in JWT validation config.
        """,
        "error_category": "auth_failure",
        "fix_type": "config",
        "risk_level": "low",
        "success_rate": 0.91
    },
]

# ─────────────────────────────────────────────
# 3. VECTOR KNOWLEDGE BASE
# ─────────────────────────────────────────────

class VectorKnowledgeBase:

    def __init__(self, embed_model: SentenceTransformer):
        self.model  = embed_model
        self.chunks: list[Chunk] = []
        print("\n[KB] Initializing vector knowledge base...")

    def ingest(self, raw_fixes: list[dict]):
        print(f"[KB] Ingesting {len(raw_fixes)} fix documents...")
        texts = [f["text"].strip() for f in raw_fixes]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        for i, fix in enumerate(raw_fixes):
            chunk = Chunk(
                id             = fix["id"],
                text           = fix["text"].strip(),
                embedding      = embeddings[i],
                error_category = fix["error_category"],
                fix_type       = fix["fix_type"],
                risk_level     = fix["risk_level"],
                success_rate   = fix["success_rate"],
                date           = datetime.utcnow().isoformat()
            )
            self.chunks.append(chunk)
            print(f"  [OK] Ingested: {chunk.id} [{chunk.error_category}]")

        print(f"[KB] {len(self.chunks)} chunks ready.\n")

    def search(self, query: str, top_k: int = 20,
               category_filter: Optional[str] = None,
               min_success_rate: float = 0.0) -> list[RetrievalResult]:

        query_embedding = self.model.encode([query])[0]

        candidates = self.chunks

        # Metadata pre-filter
        if category_filter:
            filtered = [c for c in candidates
                        if c.error_category == category_filter]
            # fall back to all if filter returns nothing
            candidates = filtered if filtered else candidates

        # Success rate filter — deprioritize known-bad fixes
        candidates = [c for c in candidates
                      if c.success_rate >= min_success_rate]

        if not candidates:
            return []

        # Cosine similarity
        results = []
        for chunk in candidates:
            score = self._cosine(query_embedding, chunk.embedding)
            results.append(RetrievalResult(chunk=chunk, similarity=score))

        # Sort by similarity, return top_k
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        )

# ─────────────────────────────────────────────
# 4. RERANKER
# ─────────────────────────────────────────────

class Reranker:

    def __init__(self, model: CrossEncoder):
        self.model = model

    def rerank(self, query: str,
               results: list[RetrievalResult],
               top_n: int = 5) -> list[RetrievalResult]:

        if not results:
            return []

        pairs  = [[query, r.chunk.text] for r in results]
        scores = self.model.predict(pairs)

        for r, score in zip(results, scores):
            r.rerank_score = float(score)

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

# ─────────────────────────────────────────────
# 5. DECISION ENGINE
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.40   # below → no relevant fix in KB
AUTO_APPLY_THRESHOLD = 0.75   # similarity >= this + low risk → auto-fix
SUGGEST_THRESHOLD    = 0.55   # similarity >= this → suggest for approval

def make_decision(top_results: list[RetrievalResult],
                  query: str) -> AgentDecision:
    """
    Decision logic uses cosine similarity (0.0–1.0) as confidence.
    Cross-encoder reranker reorders candidates — best chunk is top_results[0].
    Confidence is then weighted by historical success_rate from KB metadata.

    Thresholds:
      similarity >= 0.75 AND risk=low  → auto_apply
      similarity >= 0.55               → suggest (human approves)
      similarity <  0.55 OR risk=high  → escalate to oncall
    """

    if not top_results:
        return AgentDecision(
            action="escalate", fix="No relevant fix found in KB.",
            confidence=0.0, risk_level="unknown",
            reason="KB returned no results above similarity threshold.",
            source_chunk_id="none"
        )

    best       = top_results[0]
    similarity = best.similarity           # cosine similarity 0.0–1.0
    risk       = best.chunk.risk_level

    # Weight similarity by chunk's historical success rate
    # e.g. similarity=0.80, success_rate=0.92 → confidence=0.736
    confidence = round(similarity * best.chunk.success_rate, 3)

    # ── Decision tree ──────────────────────────────────────────
    # Rule 1: High risk always requires human — regardless of confidence
    if risk == "high":
        action = "escalate"
        reason = (f"High-risk action type '{best.chunk.fix_type}' "
                  f"requires human approval regardless of confidence.")

    # Rule 2: High confidence + low risk → auto-apply
    elif similarity >= AUTO_APPLY_THRESHOLD and risk == "low":
        action = "auto_apply"
        reason = (f"Similarity {similarity:.3f} >= threshold {AUTO_APPLY_THRESHOLD} "
                  f"with low risk. Confidence={confidence}. Auto-applying fix.")

    # Rule 3: Medium confidence → suggest for human approval
    elif similarity >= SUGGEST_THRESHOLD:
        action = "suggest"
        reason = (f"Similarity {similarity:.3f} >= suggest threshold {SUGGEST_THRESHOLD}. "
                  f"Confidence={confidence}. Recommending fix — awaiting approval.")

    # Rule 4: Low similarity → not enough evidence, escalate
    else:
        action = "escalate"
        reason = (f"Similarity {similarity:.3f} below suggest threshold {SUGGEST_THRESHOLD}. "
                  f"Insufficient evidence for automated recommendation. Escalating.")

    return AgentDecision(
        action          = action,
        fix             = best.chunk.text.strip(),
        confidence      = confidence,
        risk_level      = risk,
        reason          = reason,
        source_chunk_id = best.chunk.id
    )

# ─────────────────────────────────────────────
# 6. AGENT ORCHESTRATOR — MAIN LOOP
# ─────────────────────────────────────────────

class APITroubleshootingAgent:

    def __init__(self, kb: VectorKnowledgeBase, reranker: Reranker):
        self.kb       = kb
        self.reranker = reranker

    def run(self, error_event: dict) -> dict:
        print("=" * 60)
        print(f"[AGENT] New error event received")
        print(f"  Category : {error_event.get('category', 'unknown')}")
        print(f"  Message  : {error_event.get('message', '')[:80]}")
        print("=" * 60)

        query    = f"{error_event['category']} {error_event['message']}"
        category = error_event.get("category")

        # Step 1 — Vector search (top-20)
        print("\n[STEP 1] Vector similarity search...")
        candidates = self.kb.search(
            query           = query,
            top_k           = 20,
            category_filter = category,
            min_success_rate = 0.75
        )
        print(f"  Retrieved {len(candidates)} candidates")
        for r in candidates[:3]:
            print(f"    {r.chunk.id} | sim={r.similarity:.3f} "
                  f"| success={r.chunk.success_rate}")

        # Similarity gate
        if not candidates or candidates[0].similarity < SIMILARITY_THRESHOLD:
            print(f"\n  [GATE] Top similarity {candidates[0].similarity:.3f}"
                  f" below threshold {SIMILARITY_THRESHOLD} → escalating")
            return self._build_output(error_event, [], AgentDecision(
                action="escalate", fix="No relevant match in KB.",
                confidence=0.0, risk_level="unknown",
                reason="Similarity below threshold.",
                source_chunk_id="none"
            ))

        # Step 2 — Rerank (top-5)
        print("\n[STEP 2] Reranking top candidates...")
        top_5 = self.reranker.rerank(query, candidates, top_n=5)
        print(f"  Top 5 after reranking:")
        for r in top_5:
            print(f"    {r.chunk.id} | sim={r.similarity:.3f} "
                  f"| rerank={r.rerank_score:.3f} "
                  f"| risk={r.chunk.risk_level}")

        # Step 3 — Decision
        print("\n[STEP 3] Decision logic...")
        decision = make_decision(top_5, query)
        print(f"  Action     : {decision.action.upper()}")
        print(f"  Confidence : {decision.confidence}")
        print(f"  Risk level : {decision.risk_level}")
        print(f"  Reason     : {decision.reason}")
        print(f"  Source     : {decision.source_chunk_id}")

        return self._build_output(error_event, top_5, decision)

    def _build_output(self, error_event, results, decision):
        return {
            "ts":           datetime.utcnow().isoformat(),
            "error_event":  error_event,
            "action":       decision.action,
            "confidence":   decision.confidence,
            "risk_level":   decision.risk_level,
            "reason":       decision.reason,
            "fix":          decision.fix[:200] + "..." if len(decision.fix) > 200 else decision.fix,
            "source":       decision.source_chunk_id,
            "top_results":  [
                {
                    "id":           r.chunk.id,
                    "similarity":   round(r.similarity, 3),
                    "rerank_score": round(r.rerank_score, 3),
                    "success_rate": r.chunk.success_rate,
                    "risk":         r.chunk.risk_level
                } for r in results
            ]
        }

# ─────────────────────────────────────────────
# 7. RUN — TEST CASES
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  AGENTIC RAG ORCHESTRATOR - API TROUBLESHOOTING")
    print("=" * 60)

    # Load models
    print("\n[INIT] Loading embedding model...")
    embed_model  = SentenceTransformer("all-MiniLM-L6-v2")

    print("[INIT] Loading reranker model...")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Build KB
    kb       = VectorKnowledgeBase(embed_model)
    kb.ingest(RAW_FIXES)

    reranker = Reranker(rerank_model)
    agent    = APITroubleshootingAgent(kb, reranker)

    # Test cases
    TEST_ERRORS = [
        {
            "category": "timeout",
            "message":  "ReadTimeoutError: HTTPSConnectionPool host=api.payments.internal "
                        "port=443 Read timed out after 30 seconds",
            "service":  "payments-service"
        },
        {
            "category": "auth_failure",
            "message":  "401 Unauthorized: JWT signature verification failed — "
                        "token issued in future (clock skew detected)",
            "service":  "auth-service"
        },
        {
            "category": "rate_limit",
            "message":  "429 Too Many Requests: Rate limit exceeded. "
                        "Retry-After: 60 seconds",
            "service":  "notification-service"
        },
        {
            "category": "service_down",
            "message":  "503 Service Unavailable: upstream connect error "
                        "or disconnect/reset before headers",
            "service":  "inventory-service"
        }
    ]

    results = []
    for error in TEST_ERRORS:
        result = agent.run(error)
        results.append(result)

        print(f"\n{'-'*60}")
        print(f"DECISION: {result['action'].upper()} | "
              f"Confidence: {result['confidence']} | "
              f"Risk: {result['risk_level']}")
        print(f"Reason  : {result['reason']}")
        print(f"Fix     : {result['fix'][:100]}...")
        print(f"Source  : {result['source']}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Error':<20} {'Action':<15} {'Confidence':<12} {'Risk'}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['error_event']['category']:<20} "
              f"{r['action']:<15} "
              f"{str(r['confidence']):<12} "
              f"{r['risk_level']}")

    auto   = sum(1 for r in results if r["action"] == "auto_apply")
    suggest = sum(1 for r in results if r["action"] == "suggest")
    escalate = sum(1 for r in results if r["action"] == "escalate")

    print(f"\n  Auto-apply : {auto}")
    print(f"  Suggest    : {suggest}")
    print(f"  Escalate   : {escalate}")
    print(f"  Auto-resolve rate: {((auto+suggest)/len(results)*100):.0f}%")
    print("\n[DONE]")

if __name__ == "__main__":
    main()
