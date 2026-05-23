"""
STEP 6 — AI Observability Dashboard (FastAPI)
==============================================
Production AI systems need monitoring. This file adds:

  1. METRICS ENDPOINT    — confidence scores, query counts, escalation rate
  2. DRIFT DETECTION     — flags if average confidence drops significantly
  3. QUERY HISTORY       — full audit trail of every question + answer
  4. HEALTH CHECK        — system status at a glance
  5. HTML DASHBOARD      — visual overview at /dashboard

This is what you'd wire to Azure Log Analytics, CloudWatch, or Grafana in production.

Endpoints:
  GET  /health           → system status
  POST /upload           → load PDF
  POST /ask              → ask question (with confidence scoring)
  GET  /metrics          → aggregated metrics + drift alert
  GET  /history          → full query audit trail
  GET  /dashboard        → HTML visual overview

Run:
    python 6_observability_dashboard.py

Open: http://localhost:8000/dashboard
"""

import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="AI RAG Observability API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
DRIFT_ALERT_THRESHOLD = 0.55   # if rolling avg drops below this → drift alert

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
vectorstore     = None
loaded_filename = None
query_history   = []           # full audit trail
llm             = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings      = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─── MODELS ───────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class QueryRecord(BaseModel):
    id: int
    timestamp: str
    question: str
    answer: str
    confidence: float
    answered_autonomously: bool
    chunks_used: int


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def score_confidence(question: str, chunks: list) -> float:
    if not chunks:
        return 0.0
    context = "\n".join([c.page_content[:200] for c in chunks[:3]])
    prompt  = f"""Score how well this context answers the question.
Respond with ONLY a decimal between 0.0 and 1.0.
Question: {question}
Context: {context}"""
    try:
        resp = llm.invoke(prompt)
        return min(max(float(resp.content.strip()), 0.0), 1.0)
    except Exception:
        return 0.5


def get_metrics():
    total     = len(query_history)
    if total == 0:
        return {"total": 0, "avg_confidence": 0, "auto_rate": 0, "escalation_rate": 0, "drift_alert": False}
    confidences  = [r["confidence"] for r in query_history]
    auto_count   = sum(1 for r in query_history if r["answered_autonomously"])
    avg_conf     = sum(confidences) / total
    # Drift: compare last 5 queries avg vs overall avg
    recent_avg   = sum(confidences[-5:]) / min(5, total)
    drift_alert  = recent_avg < DRIFT_ALERT_THRESHOLD
    return {
        "total_queries":       total,
        "avg_confidence":      round(avg_conf, 3),
        "recent_avg_confidence": round(recent_avg, 3),
        "auto_answered":       auto_count,
        "escalated":           total - auto_count,
        "auto_rate_pct":       round(auto_count / total * 100, 1),
        "escalation_rate_pct": round((total - auto_count) / total * 100, 1),
        "drift_alert":         drift_alert,
        "drift_message":       "⚠️ CONFIDENCE DRIFT DETECTED — recent queries scoring low" if drift_alert else "✓ Confidence stable",
    }


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    m = get_metrics()
    return {
        "status":           "ok",
        "pdf_loaded":       vectorstore is not None,
        "loaded_file":      loaded_filename,
        "total_queries":    m.get("total_queries", 0),
        "drift_alert":      m.get("drift_alert", False),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore, loaded_filename
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        loader  = PyPDFLoader(tmp_path)
        pages   = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks  = splitter.split_documents(pages)
        vectorstore     = FAISS.from_documents(chunks, embeddings)
        loaded_filename = file.filename
        return {"message": f"Loaded '{file.filename}'", "pages": len(pages), "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask")
async def ask(req: QuestionRequest):
    if vectorstore is None:
        raise HTTPException(400, "No PDF loaded. POST /upload first.")
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    retriever  = vectorstore.as_retriever(search_kwargs={"k": 4})
    chunks     = retriever.invoke(req.question)
    confidence = score_confidence(req.question, chunks)
    answered   = confidence >= CONFIDENCE_THRESHOLD

    if not answered:
        answer = (f"Insufficient information to answer confidently. "
                  f"Confidence: {confidence:.0%}. Please rephrase or consult the source document.")
    else:
        context = "\n\n".join(doc.page_content for doc in chunks)
        prompt  = ChatPromptTemplate.from_messages([
            ("system", "Answer based only on the context. Be concise and accurate."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": req.question})

    record = {
        "id":                   len(query_history) + 1,
        "timestamp":            datetime.now().isoformat(),
        "question":             req.question,
        "answer":               answer,
        "confidence":           confidence,
        "answered_autonomously": answered,
        "chunks_used":          len(chunks),
        "sources": [f"Page {d.metadata.get('page','?')}: {d.page_content[:100]}..." for d in chunks]
    }
    query_history.append(record)

    return {
        "question":   req.question,
        "answer":     answer,
        "confidence": round(confidence, 3),
        "answered":   answered,
        "sources":    record["sources"],
    }


@app.get("/metrics")
def metrics_endpoint():
    return get_metrics()


@app.get("/history", response_model=list)
def history():
    return list(reversed(query_history))  # most recent first


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    m = get_metrics()
    drift_color  = "#ff4444" if m.get("drift_alert") else "#00cc66"
    drift_msg    = m.get("drift_message", "")
    rows = ""
    for r in reversed(query_history):
        conf_pct = f"{r['confidence']*100:.0f}%"
        status   = "✓ Auto" if r["answered_autonomously"] else "⚠ Escalated"
        color    = "#004400" if r["answered_autonomously"] else "#880000"
        rows += f"""<tr>
          <td>{r['id']}</td>
          <td>{r['timestamp'][11:19]}</td>
          <td>{r['question'][:60]}</td>
          <td>{conf_pct}</td>
          <td style="color:{color};font-weight:600">{status}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>AI RAG Observability Dashboard</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background:#0f0f1a; color:#e0e0e0; margin:0; padding:20px; }}
    h1 {{ color:#4fc3f7; font-size:22px; margin-bottom:4px; }}
    .sub {{ color:#888; font-size:12px; margin-bottom:20px; }}
    .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .card {{ background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:16px 20px; min-width:160px; }}
    .card .val {{ font-size:28px; font-weight:700; color:#4fc3f7; }}
    .card .lbl {{ font-size:11px; color:#888; margin-top:4px; }}
    .drift {{ background:#1a1a2e; border:2px solid {drift_color}; border-radius:8px;
              padding:12px 20px; margin-bottom:24px; color:{drift_color}; font-weight:600; }}
    table {{ width:100%; border-collapse:collapse; background:#1a1a2e; border-radius:8px; overflow:hidden; }}
    th {{ background:#16213e; color:#4fc3f7; padding:10px 12px; text-align:left; font-size:11px; text-transform:uppercase; }}
    td {{ padding:9px 12px; font-size:12px; border-bottom:1px solid #222; }}
    tr:hover {{ background:#16213e; }}
  </style>
</head>
<body>
  <h1>AI RAG Observability Dashboard</h1>
  <div class="sub">Auto-refreshes every 10s | File: {loaded_filename or 'No PDF loaded'}</div>

  <div class="drift">{drift_msg}</div>

  <div class="cards">
    <div class="card"><div class="val">{m.get('total_queries',0)}</div><div class="lbl">Total Queries</div></div>
    <div class="card"><div class="val">{m.get('avg_confidence',0)*100:.0f}%</div><div class="lbl">Avg Confidence</div></div>
    <div class="card"><div class="val">{m.get('recent_avg_confidence',0)*100:.0f}%</div><div class="lbl">Recent Avg (last 5)</div></div>
    <div class="card"><div class="val" style="color:#00cc66">{m.get('auto_rate_pct',0)}%</div><div class="lbl">Auto-Answer Rate</div></div>
    <div class="card"><div class="val" style="color:#ff8800">{m.get('escalation_rate_pct',0)}%</div><div class="lbl">Escalation Rate</div></div>
    <div class="card"><div class="val">{m.get('auto_answered',0)}</div><div class="lbl">Auto Answered</div></div>
    <div class="card"><div class="val">{m.get('escalated',0)}</div><div class="lbl">Escalated (HITL)</div></div>
  </div>

  <table>
    <tr><th>#</th><th>Time</th><th>Question</th><th>Confidence</th><th>Status</th></tr>
    {rows if rows else '<tr><td colspan="5" style="text-align:center;color:#555;padding:20px">No queries yet. POST to /ask to begin.</td></tr>'}
  </table>

  <p style="color:#444;font-size:11px;margin-top:16px">
    API Docs: <a href="/docs" style="color:#4fc3f7">/docs</a> |
    Raw Metrics: <a href="/metrics" style="color:#4fc3f7">/metrics</a> |
    Query History: <a href="/history" style="color:#4fc3f7">/history</a>
  </p>
</body>
</html>"""
    return html


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  AI Observability Dashboard starting...")
    print("  Dashboard: http://localhost:8000/dashboard")
    print("  API Docs:  http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
