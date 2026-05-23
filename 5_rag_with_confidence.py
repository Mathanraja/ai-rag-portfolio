"""
STEP 5 — RAG with Confidence Scoring + Fallback
=================================================
This demonstrates production-grade RAG thinking that senior interviewers test for.

Key additions over basic RAG (file 1):
  1. CONFIDENCE SCORING — every answer gets a score 0.0–1.0
  2. FALLBACK behaviour — below threshold → honest "I don't know"
  3. SOURCE ATTRIBUTION — every answer cites which chunks it used
  4. METRICS TRACKING — tracks all scores (for drift detection demo)
  5. RETRIEVAL QUALITY — checks if retrieved chunks are actually relevant

This is the pattern used in enterprise production AI platforms.

Run:
    python 5_rag_with_confidence.py

Then point it at any PDF and ask questions.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # below this → fallback to "I don't know"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4                     # retrieve top 4 chunks


# ─── METRICS STORE ────────────────────────────────────────────────────────────
# In production: write to Azure Log Analytics / CloudWatch / Prometheus
# Here: in-memory list for demo

class MetricsStore:
    def __init__(self):
        self.records = []

    def log(self, question: str, confidence: float, answered: bool, chunks_used: int):
        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question[:80],
            "confidence": confidence,
            "answered_autonomously": answered,
            "chunks_used": chunks_used,
        }
        self.records.append(record)

    def summary(self):
        if not self.records:
            return "No queries yet."
        confidences = [r["confidence"] for r in self.records]
        answered    = sum(1 for r in self.records if r["answered_autonomously"])
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


metrics = MetricsStore()


# ─── PIPELINE SETUP ───────────────────────────────────────────────────────────
def load_and_index_pdf(pdf_path: str):
    """Load PDF → chunk → embed → FAISS vector store."""
    print(f"\nLoading: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"  Pages: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"  Chunks: {len(chunks)}")

    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  Vector store ready.\n")

    return vectorstore


def score_confidence(question: str, context_chunks: list, llm: ChatOpenAI) -> float:
    """
    Ask the LLM to score how well the retrieved context answers the question.
    Returns a float 0.0–1.0.

    In production: you could also use cosine similarity scores from the vector store,
    or a separate cross-encoder model for more accurate relevance scoring.
    """
    if not context_chunks:
        return 0.0

    context_preview = "\n".join([c.page_content[:200] for c in context_chunks[:3]])

    prompt = f"""Given the following context and question, score how well the context
answers the question on a scale of 0.0 to 1.0.

0.0 = context is completely irrelevant
0.5 = context is partially relevant
1.0 = context directly and completely answers the question

Question: {question}

Context:
{context_preview}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)  # clamp to [0.0, 1.0]
    except (ValueError, AttributeError):
        return 0.5  # default if scoring fails


def answer_with_confidence(
    question: str,
    vectorstore: FAISS,
    llm: ChatOpenAI
) -> dict:
    """
    Full RAG pipeline with confidence scoring.
    Returns: answer, confidence, sources, whether it was answered autonomously.
    """

    # 1. Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    chunks    = retriever.invoke(question)

    # 2. Score confidence
    confidence = score_confidence(question, chunks, llm)

    # 3. Decision: answer autonomously or escalate?
    if confidence < CONFIDENCE_THRESHOLD:
        metrics.log(question, confidence, answered=False, chunks_used=len(chunks))
        return {
            "answer": (
                f"I don't have enough reliable information in this document to answer "
                f"'{question}' confidently. (Confidence: {confidence:.0%})\n"
                f"Please check the source document directly or rephrase your question."
            ),
            "confidence": confidence,
            "answered": False,
            "sources": [],
        }

    # 4. Generate answer using retrieved context
    context = "\n\n".join(doc.page_content for doc in chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Answer the question based ONLY on the context provided. "
            "If the context doesn't contain the answer, say so. "
            "Be concise and accurate."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # 5. Build source citations
    sources = []
    for doc in chunks:
        page    = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        sources.append(f"Page {page}: {preview}...")

    metrics.log(question, confidence, answered=True, chunks_used=len(chunks))

    return {
        "answer":     answer,
        "confidence": confidence,
        "answered":   True,
        "sources":    sources,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n=== RAG with Confidence Scoring ===")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD} ({CONFIDENCE_THRESHOLD*100:.0f}%)")
    print("Below threshold → honest fallback instead of hallucination\n")

    pdf_path = input("Enter path to PDF: ").strip().strip('"')
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    vectorstore = load_and_index_pdf(pdf_path)
    llm         = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    print("Ready! Type questions. Commands: 'metrics' = show stats, 'quit' = exit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "metrics":
            metrics.print_all()
            continue

        result = answer_with_confidence(question, vectorstore, llm)

        # Display result
        conf_pct = result["confidence"] * 100
        status   = "✓ AUTO-ANSWERED" if result["answered"] else "⚠ ESCALATED (low confidence)"

        print(f"\n[{status} | Confidence: {conf_pct:.0f}%]")
        print(f"\n{result['answer']}")

        if result["sources"]:
            print("\n[Sources:]")
            for i, src in enumerate(result["sources"], 1):
                print(f"  {i}. {src}")
        print()


if __name__ == "__main__":
    main()
