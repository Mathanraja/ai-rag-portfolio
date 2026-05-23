"""
STEP 2 — LangChain Agent with Tools and Memory

An "agent" is an LLM that can:
  1. Decide WHICH tool to use
  2. Use the tool to get information
  3. Remember previous messages (memory)

Tools in this example:
  - get_current_time  → returns today's date/time
  - calculate         → evaluates math expressions
  - search_topic      → returns info about AI topics

Run:
    python 2_agent_memory.py

Try asking: "What time is it?", "What is 25 * 48?", "Explain RAG", "What did I just ask you?"
"""

import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool                                         # decorator to make a function into a tool
from langchain.memory import ConversationBufferMemory                    # remembers chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# ─── TOOLS ───────────────────────────────────────────────────────────────────
# @tool decorator tells LangChain: "this function is a tool the agent can call"
# The docstring is what the LLM reads to decide WHEN to use this tool

@tool
def get_current_time() -> str:
    """Use this to get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time: %I:%M %p")


@tool
def calculate(expression: str) -> str:
    """
    Use this to evaluate a math expression.
    Examples: '2 + 2', '100 / 4', '15 * 33', '2 ** 10'
    """
    try:
        # eval() runs the math — we restrict it to math operations only (safe)
        allowed = {k: v for k, v in __builtins__.items()
                   if k in ("abs", "round", "min", "max", "sum", "pow")} \
                   if isinstance(__builtins__, dict) else {}
        result = eval(expression, {"__builtins__": allowed}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {e}"


@tool
def search_topic(topic: str) -> str:
    """
    Use this to look up information about AI and tech topics.
    Good for: RAG, LangChain, LLM, embeddings, FAISS, FastAPI, Python, agents
    """
    knowledge_base = {
        "rag": (
            "RAG (Retrieval-Augmented Generation) is an AI pattern that combines "
            "a vector search engine with an LLM. Steps: load documents → chunk → "
            "embed → store in vector DB → retrieve top-k chunks → pass to LLM for answer. "
            "Used to make LLMs answer from YOUR documents, not just training data."
        ),
        "langchain": (
            "LangChain is a Python framework for building LLM applications. "
            "Key components: Chains (sequence of steps), Agents (LLM decides what to do), "
            "Memory (remembers conversation), Tools (functions the agent can call), "
            "Vectorstores (store and search embeddings)."
        ),
        "embedding": (
            "Embeddings convert text into vectors (lists of numbers). "
            "Similar meaning → similar vectors (close in vector space). "
            "Used in RAG to find chunks that are semantically similar to a question. "
            "OpenAI text-embedding-ada-002 returns 1536-dimensional vectors."
        ),
        "faiss": (
            "FAISS (Facebook AI Similarity Search) is a library for storing and "
            "searching vectors efficiently. In RAG, we store all chunk embeddings in FAISS. "
            "When user asks a question, we embed the question and find nearest vectors."
        ),
        "agent": (
            "A LangChain agent is an LLM that can use tools. "
            "The LLM decides: 1) Do I need a tool? 2) Which tool? 3) What input? "
            "Then it calls the tool, reads the output, and decides what to do next. "
            "This loop continues until the agent has enough info to answer."
        ),
        "fastapi": (
            "FastAPI is a Python web framework for building APIs. "
            "Key features: automatic docs at /docs, async support, Pydantic validation. "
            "Used to expose your RAG chatbot as a REST API that any frontend can call."
        ),
        "python": (
            "Python is a high-level language popular for AI/ML. "
            "Key AI libraries: LangChain, OpenAI, FastAPI, pandas, numpy, scikit-learn. "
            "Start with: variables, functions, lists/dicts, classes, then pip packages."
        ),
    }

    topic_lower = topic.lower()
    for key, info in knowledge_base.items():
        if key in topic_lower:
            return info

    return (
        f"No detailed info on '{topic}' in the knowledge base. "
        f"Available topics: {', '.join(knowledge_base.keys())}"
    )


# ─── BUILD AGENT ─────────────────────────────────────────────────────────────
def build_agent():
    tools = [get_current_time, calculate, search_topic]

    # Memory — stores the full conversation so the agent can refer back to it
    memory = ConversationBufferMemory(
        memory_key="chat_history",   # must match the variable name in the prompt
        return_messages=True,        # return as message objects (not just text)
    )

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt template — defines how the conversation is structured
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI assistant specialising in AI/ML topics. "
            "Use available tools when appropriate. Be concise and clear."
        )),
        MessagesPlaceholder(variable_name="chat_history"),   # memory goes here
        ("human", "{input}"),                                # user message
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool call results
    ])

    # create_openai_tools_agent = agent that uses OpenAI's function-calling feature
    agent = create_openai_tools_agent(llm, tools, prompt)

    # AgentExecutor runs the agent loop: think → tool → think → answer
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,    # set to False to hide the agent's thinking steps
        max_iterations=5,
    )
    return agent_executor


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n=== LangChain Agent with Memory ===")
    print("Tools available: get_current_time, calculate, search_topic")
    print("Type 'quit' to exit.\n")

    agent = build_agent()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        print()  # blank line before verbose output
        response = agent.invoke({"input": user_input})
        print(f"\nAgent: {response['output']}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
