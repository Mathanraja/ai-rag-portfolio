# AI RAG Project — Production Portfolio

## What this project demonstrates (for interviews)

| File | Concept | Interview question it answers |
|------|---------|-------------------------------|
| `1_rag_chatbot.py` | Basic RAG pipeline | "Explain RAG and how you've implemented it" |
| `2_agent_memory.py` | LangChain agent + tools + memory | "What is a LangChain agent?" |
| `3_fastapi_app.py` | RAG as REST API | "How do you expose an AI model as a service?" |
| `4_langgraph_multiagent.py` | LangGraph coordinator + specialists | "How do you build multi-agent systems?" "How do agents share state?" |
| `5_rag_with_confidence.py` | Confidence scoring + fallback | "How do you prevent hallucinations in production?" |
| `6_observability_dashboard.py` | Metrics + drift detection + dashboard | "How do you monitor AI in production?" |

---

## One-time setup

```bash
pip install -r requirements.txt
```

---

## .env file

Create a `.env` file in this folder:
```
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here   # only needed for file 1
```

---

## Run each file

### File 1 — Basic RAG Chatbot
```bash
python 1_rag_chatbot.py
```
Enter any PDF path → ask questions.

### File 2 — LangChain Agent with Memory
```bash
python 2_agent_memory.py
```
Try: "What is RAG?", "What is 25 * 48?", "What time is it?", "What did I just ask?"

### File 3 — RAG as FastAPI
```bash
python 3_fastapi_app.py
```
Open http://localhost:8000/docs → upload PDF → ask questions via Swagger UI.

### File 4 — LangGraph Multi-Agent (KEY FILE for interviews)
```bash
pip install langgraph
python 4_langgraph_multiagent.py
```
Watch the coordinator route to Research vs Calculator agents automatically.
This file answers: "How do you share state across agents?" and "How do you build multi-agent systems?"

### File 5 — RAG with Confidence Scoring
```bash
python 5_rag_with_confidence.py
```
Ask questions about a PDF. Watch confidence scores and fallback behaviour.
Type `metrics` to see the observability summary.

### File 6 — Full Observability Dashboard
```bash
python 6_observability_dashboard.py
```
Open http://localhost:8000/dashboard — live metrics, drift detection, audit trail.
Upload PDF at /docs first, then ask questions, watch the dashboard update.

---

## Interview talking points

### On LangGraph state sharing (the killer question)
"I use LangGraph's StateGraph with a TypedDict shared state object.
Every agent reads from and writes back to the same state.
The coordinator writes its routing decision to `state['route']`.
Specialist agents write their answers to `state['research_answer']` or `state['calc_answer']`.
Conditional edges read from the state to decide which node runs next.
In production with distributed agents, I'd use Redis or a shared database as the state backend."

### On confidence scoring and hallucination prevention
"Every RAG response gets a confidence score before I answer.
Below 0.70 threshold → the system says 'I don't have reliable information' instead of guessing.
This is the HITL escalation pattern — low confidence routes to human review.
I log every score to detect drift: if the rolling average drops, I trigger a re-indexing pipeline."

### On production monitoring
"I track: total queries, average confidence, escalation rate, and recent vs historical confidence.
A drop in recent average confidence is the signal for knowledge base drift.
In production I wire this to Azure Log Analytics with alerts on threshold breaches."
