# AI RAG Portfolio — Production GenAI Architecture

**Mathanraja Ramasamy** | AI Platform Architect | [LinkedIn](https://linkedin.com/in/mathanrajaaiautomation)

Production-pattern GenAI engineering portfolio demonstrating end-to-end AI system design — RAG pipelines, multi-agent orchestration, confidence scoring, observability, and FastAPI deployment.

Built with: **LangChain · LangGraph · FastAPI · FAISS · HuggingFace · OpenAI**

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           RAG Pipeline (Files 1, 5)          │
│  PDF → Chunk → Embed → FAISS → Retrieve     │
│  + Confidence Scoring + HITL Fallback        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│     Multi-Agent Orchestration (File 4)       │
│                                              │
│  Coordinator Agent                           │
│       ├── Research Specialist Agent          │
│       └── Calculator Specialist Agent        │
│                                              │
│  Shared State (TypedDict) across all agents  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│    FastAPI + Observability (Files 3, 6)      │
│  /upload /ask /metrics /history /dashboard  │
│  Confidence tracking · Drift detection       │
└─────────────────────────────────────────────┘
```

---

## Files

| File | Pattern | Key Concepts |
|------|---------|-------------|
| `1_rag_chatbot.py` | Basic RAG | PDF ingestion, chunking, FAISS vector store, LLM Q&A |
| `2_agent_memory.py` | LangChain Agent | Tools, conversation memory, agent reasoning loop |
| `3_fastapi_app.py` | RAG as API | FastAPI endpoints, file upload, async, Pydantic |
| `4_langgraph_multiagent.py` | **LangGraph Multi-Agent** | StateGraph, coordinator + specialist routing, shared TypedDict state, conditional edges |
| `5_rag_with_confidence.py` | **Production RAG** | Confidence scoring, HITL fallback, source attribution, metrics tracking |
| `6_observability_dashboard.py` | **AI Observability** | Drift detection, audit trail, live HTML dashboard, rolling metrics |

---

## Key Architecture Decisions

### Agent State Sharing (LangGraph)
```python
class AgentState(TypedDict):
    question: str        # shared input
    route: str           # coordinator writes routing decision
    research_answer: str # research agent writes here
    calc_answer: str     # calculator agent writes here
    final_answer: str    # whichever specialist ran last
    steps: list[str]     # full audit trail
```
Every agent reads from and writes to the same `AgentState`. The coordinator's routing decision persists in state so conditional edges can read it.

### Confidence Scoring + HITL Fallback
```python
CONFIDENCE_THRESHOLD = 0.70

if confidence < CONFIDENCE_THRESHOLD:
    # Escalate to human — never hallucinate
    return "I don't have reliable information to answer this."
else:
    # Answer autonomously with source attribution
    return generate_answer(context, question)
```

### Drift Detection
```python
recent_avg = avg(confidence_scores[-5:])
if recent_avg < DRIFT_ALERT_THRESHOLD:
    trigger_alert("Confidence drift detected — knowledge base may be stale")
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=your-key-here
```

Run any file:
```bash
python 4_langgraph_multiagent.py   # multi-agent demo
python 6_observability_dashboard.py # open http://localhost:8000/dashboard
```

---

## Production Context

These patterns reflect production AI platform architecture built and delivered at enterprise scale:
- 1M+ users in production
- Zero post-launch incidents
- 23 production AI tools across 12 engineering teams
- 60% operational failure reduction
- RBAC/ABAC governance, full audit trail, HITL escalation

---

*Built during active AI/ML upskilling — May 2026*
