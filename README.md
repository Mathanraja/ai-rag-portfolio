# AI RAG Portfolio — Production GenAI & Agentic Testing Architecture

**Mathanraja Ramasamy** | AI Platform Architect | [LinkedIn](https://linkedin.com/in/mathanrajaaiautomation)

Production-pattern GenAI engineering portfolio covering RAG pipelines, multi-agent orchestration,
LLM quality testing, MLOps, and autonomous agentic testing workflows — aligned with the
[Testleaf AI Agentic Workflows curriculum](https://testleaf.com).

Built with: **LangChain · LangGraph · FastAPI · FAISS · DeepEval · MLflow · XGBoost · OpenAI**

---

## Curriculum Map (Testleaf AI Agentic Workflows — 12 Weeks)

| Week | Topic | Portfolio File |
|------|-------|---------------|
| 1–3  | RAG & Vector Databases | `1_rag_chatbot.py`, `5_rag_with_confidence.py` |
| 4–5  | LangChain & LangGraph | `2_agent_memory.py`, `4_langgraph_multiagent.py`, `8_testcase_generator_agent.py` |
| 6    | LLM Testing with DeepEval | `9_deepeval_llm_testing.py` |
| 7–12 | Agentic Workflows & Deployment | `3_fastapi_app.py`, `6_observability_dashboard.py`, `7_mlflow_experiments.py`, `10_swagger_api_agent.py` |

---

## Architecture Overview

```
User Query / Requirement / Swagger Spec
        │
        ▼
┌────────────────────────────────────────────────┐
│         RAG Pipeline (Files 1, 5)               │
│  PDF → Chunk → Embed → FAISS → Retrieve        │
│  + Confidence Scoring (0.70) + HITL Fallback   │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│   Agentic Workflows — LangGraph (Files 4, 8, 10)│
│                                                 │
│  Coordinator ──► Research Specialist            │
│             └──► Calculator Specialist          │
│                                                 │
│  Testcase Generator Agent                       │
│    analyse → generate → gherkin → playwright   │
│                                                 │
│  Swagger API Agent                              │
│    parse → mock → testgen → pytest → coverage  │
│                                                 │
│  Shared TypedDict State across all nodes        │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  LLM Quality Gates — DeepEval (File 9)          │
│  Relevancy · Faithfulness · Hallucination · Bias│
│  CI/CD gate: fail build if metrics drop         │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  FastAPI + Observability + MLOps (Files 3, 6, 7)│
│  /upload /ask /metrics /dashboard              │
│  Drift detection · MLflow Model Registry       │
└────────────────────────────────────────────────┘
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
| `7_mlflow_experiments.py` | **MLOps / Experiment Tracking** | MLflow runs, hyperparameter sweep, feature importance, Model Registry, Production promotion |
| `8_testcase_generator_agent.py` | **Testcase Generator Agent** | LangGraph workflow, Gherkin BDD output, Playwright stubs, memory-enabled dedup |
| `9_deepeval_llm_testing.py` | **LLM Quality Testing** | DeepEval metrics, hallucination detection, bias checks, CI/CD quality gate |
| `10_swagger_api_agent.py` | **Swagger / OpenAPI Agent** | OpenAPI spec parsing, mock data generation, pytest file generation, coverage matrix |

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
Every agent reads from and writes to the same `AgentState`. The coordinator's routing
decision persists in state so conditional edges can read it.

### Testcase Generator — LangGraph Workflow
```python
# 6-node pipeline: requirement → structured test cases + BDD + Playwright
graph: analyse → generate → gherkin → playwright → validate → finalise

class TestGenState(TypedDict):
    requirement:      str       # raw user story / feature text
    parsed_features:  list[str] # extracted testable features
    test_scenarios:   list[dict]# positive / negative / edge cases
    gherkin_output:   str       # .feature file (BDD)
    playwright_stubs: str       # TypeScript Playwright test stubs
    memory:           list[str] # prior requirements (dedup across runs)
```

### DeepEval Quality Gate (LLM Testing)
```python
# Metrics evaluated on every RAG / agent response
metrics = [
    AnswerRelevancyMetric(threshold=0.75),  # response addresses the question
    FaithfulnessMetric(threshold=0.80),     # no claims beyond retrieved context
    HallucinationMetric(threshold=0.20),    # lower = fewer invented facts
    BiasMetric(threshold=0.10),             # demographic fairness check
]

# CI/CD gate — blocks deployment if any metric fails
if not quality_gate(report):
    sys.exit(1)   # fail the build
```

### Swagger API Agent — Autonomous Test Generation
```python
# LangGraph: parse spec → generate mocks → create test cases → write pytest
graph: parse → mock → testgen → pytest → coverage → finalise

# Input:  any OpenAPI 3.0 JSON/YAML spec
# Output: coverage matrix + executable pytest file with assertions
```

### Confidence Scoring + HITL Fallback
```python
CONFIDENCE_THRESHOLD = 0.70

if confidence < CONFIDENCE_THRESHOLD:
    return "I don't have reliable information to answer this."  # never hallucinate
else:
    return generate_answer(context, question)  # answer with source attribution
```

### MLflow Model Registry + Governed Promotion
```python
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)        # acc, auc, f1, precision, recall
    mlflow.xgboost.log_model(model, registered_model_name="xgboost-churn-predictor")

client.transition_model_version_stage(
    name="xgboost-churn-predictor", version=best_version, stage="Production",
    archive_existing_versions=True
)
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

```bash
# Core agentic patterns
python 4_langgraph_multiagent.py          # multi-agent state sharing demo

# Testleaf curriculum projects
python 8_testcase_generator_agent.py      # requirement → Gherkin + Playwright stubs
python 9_deepeval_llm_testing.py          # LLM quality gates (runs without API key)
python 10_swagger_api_agent.py            # Swagger spec → full pytest test suite

# Observability & MLOps
python 6_observability_dashboard.py       # open http://localhost:8000/dashboard
python 7_mlflow_experiments.py            # XGBoost hyperparameter sweep
mlflow ui --backend-store-uri sqlite:///mlflow_portfolio.db   # → http://localhost:5000
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
