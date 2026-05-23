"""
STEP 1 — Simple RAG Chatbot
Flow: PDF → Load → Chunk → Embed → Store in FAISS → Retrieve → Answer

Run:
    python 1_rag_chatbot.py

Then type any question about your PDF.
"""

import os
from dotenv import load_dotenv                                     # reads .env file

# LangChain imports — each does one specific job
from langchain_community.document_loaders import PyPDFLoader              # reads PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter       # breaks text into chunks
from langchain_community.vectorstores import FAISS                        # stores embeddings in memory
from langchain_community.embeddings import HuggingFaceEmbeddings          # free local embeddings
from langchain_anthropic import ChatAnthropic                             # Claude LLM
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load your ANTHROPIC_API_KEY from .env file
load_dotenv()


# ─── STEP 1: Load PDF ────────────────────────────────────────────────────────
def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)       # point it at your PDF file
    pages = loader.load()                # returns a list of Document objects (one per page)
    print(f"  Loaded {len(pages)} pages from PDF")
    return pages


# ─── STEP 2: Chunk ───────────────────────────────────────────────────────────
def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = ~500 characters
        chunk_overlap=50,    # chunks overlap by 50 chars so context isn't lost at edges
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")
    return chunks


# ─── STEP 3: Embed + Store ───────────────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # free local model, runs on CPU
    vectorstore = FAISS.from_documents(chunks, embeddings)              # stores all vectors in memory
    print(f"  Vector store built with {len(chunks)} vectors")
    return vectorstore


# ─── STEP 4: Build RAG Chain ─────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}   # fetch top 3 most relevant chunks for each question
    )
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # fast and cheap Claude model
        temperature=0,                       # 0 = deterministic, no creativity (good for facts)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the following context:\n\n{context}"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve docs in parallel, then generate answer
    chain = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    ).assign(
        answer=(
            (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
            | prompt
            | llm
            | StrOutputParser()
        )
    )
    return chain


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n=== RAG Chatbot ===\n")

    # Get PDF path from user
    pdf_path = input("Enter path to your PDF file: ").strip().strip('"')
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("\nBuilding RAG pipeline...")
    pages      = load_pdf(pdf_path)
    chunks     = chunk_documents(pages)
    vectorstore = build_vectorstore(chunks)
    chain      = build_rag_chain(vectorstore)

    print("\nReady! Ask questions about your PDF. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        # Run the chain
        result = chain.invoke(question)

        print(f"\nBot: {result['answer']}")

        # Show which chunks were used (helpful for learning)
        print("\n[Sources used:]")
        for i, doc in enumerate(result["context"], 1):
            page_num = doc.metadata.get("page", "?")
            preview  = doc.page_content[:100].replace("\n", " ")
            print(f"  {i}. Page {page_num}: {preview}...")

        print()


if __name__ == "__main__":
    main()
