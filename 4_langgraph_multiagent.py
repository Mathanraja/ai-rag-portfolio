"""
STEP 4 — LangGraph Multi-Agent System
======================================
This is the file that answers the interview question:
"How do you build a multi-agent system with coordinator + specialist agents?"

Architecture:
  User Question
       ↓
  Coordinator Agent   ← decides which specialist to call
       ↓
  ┌────┴────────────┐
  ▼                 ▼
Research Agent   Calculator Agent
(answers facts)  (does math)
       ↓                 ↓
  Final Answer ←─────────┘

Key concepts demonstrated:
  - StateGraph: the graph that controls agent flow
  - State: shared memory between all agents (this answers "agent state sharing")
  - Nodes: each agent is a node in the graph
  - Edges: conditional routing between agents
  - END: terminal node when task is complete

Run:
    pip install langgraph
    python 4_langgraph_multiagent.py
"""

import os
import re
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()


# ─── SHARED STATE ─────────────────────────────────────────────────────────────
# This TypedDict is the "shared memory" between all agents.
# Every agent reads from it and writes back to it.
# This is what interviewers mean by "state management across agents."

class AgentState(TypedDict):
    question: str           # original user question
    route: str              # coordinator's decision: "research" | "calculate" | "done"
    research_answer: str    # filled by research agent
    calc_answer: str        # filled by calculator agent
    final_answer: str       # filled by whichever specialist ran last
    steps: list[str]        # audit trail — every action logged here


# ─── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ─── NODE 1: COORDINATOR ──────────────────────────────────────────────────────
# The coordinator reads the question and decides which specialist to call.
# It doesn't answer the question itself — it routes.

def coordinator_node(state: AgentState) -> AgentState:
    """Coordinator: reads question, decides which specialist agent to call."""
    print("\n[Coordinator] Analyzing question...")

    response = llm.invoke([
        SystemMessage(content="""You are a routing coordinator.
Analyze the user's question and respond with ONLY one word:
- 'calculate' if the question involves math, numbers, or calculations
- 'research' if the question asks for information, facts, or explanations
Do not explain. Just respond with one word."""),
        HumanMessage(content=state["question"])
    ])

    route = response.content.strip().lower()
    if route not in ("calculate", "research"):
        route = "research"  # default fallback

    print(f"[Coordinator] Routing to: {route}")

    return {
        **state,
        "route": route,
        "steps": state["steps"] + [f"Coordinator → routed to '{route}'"]
    }


# ─── NODE 2: RESEARCH AGENT ───────────────────────────────────────────────────
# Handles factual / informational questions

def research_node(state: AgentState) -> AgentState:
    """Research specialist: answers factual questions."""
    print("[Research Agent] Generating answer...")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research specialist. Answer the question clearly and concisely. "
            "Focus on facts and accuracy."
        )),
        HumanMessage(content=state["question"])
    ])

    answer = response.content.strip()
    print(f"[Research Agent] Done.")

    return {
        **state,
        "research_answer": answer,
        "final_answer": answer,
        "steps": state["steps"] + ["Research Agent → answered question"]
    }


# ─── NODE 3: CALCULATOR AGENT ─────────────────────────────────────────────────
# Handles math / calculation questions

def calculator_node(state: AgentState) -> AgentState:
    """Calculator specialist: handles math and numerical questions."""
    print("[Calculator Agent] Computing answer...")

    # First try Python eval for pure math expressions
    question = state["question"]
    # Extract any math expression from the question
    math_match = re.search(r"[\d\s\+\-\*\/\(\)\.\^]+", question)

    if math_match:
        expression = math_match.group().strip()
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            answer = f"The answer is {result}. (Computed: {expression} = {result})"
            print(f"[Calculator Agent] Computed: {expression} = {result}")
            return {
                **state,
                "calc_answer": answer,
                "final_answer": answer,
                "steps": state["steps"] + [f"Calculator Agent → computed {expression} = {result}"]
            }
        except Exception:
            pass  # fall through to LLM

    # Fallback: ask LLM to solve it
    response = llm.invoke([
        SystemMessage(content=(
            "You are a math specialist. Solve the given mathematical question step by step. "
            "Show your working clearly."
        )),
        HumanMessage(content=question)
    ])

    answer = response.content.strip()
    return {
        **state,
        "calc_answer": answer,
        "final_answer": answer,
        "steps": state["steps"] + ["Calculator Agent → solved with LLM"]
    }


# ─── ROUTING FUNCTION ─────────────────────────────────────────────────────────
# This function reads state["route"] and tells LangGraph which node to visit next.
# This is the "conditional edge" — the brain of the routing logic.

def route_decision(state: AgentState) -> Literal["research", "calculate"]:
    """Read the coordinator's decision from state and return the next node name."""
    return state["route"]


# ─── BUILD THE GRAPH ──────────────────────────────────────────────────────────
def build_graph():
    # Create a StateGraph — the container for all nodes and edges
    graph = StateGraph(AgentState)

    # Add nodes (each node = one agent function)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("research",    research_node)
    graph.add_node("calculator",  calculator_node)

    # Entry point — always start at coordinator
    graph.set_entry_point("coordinator")

    # Conditional edge: after coordinator runs, call route_decision()
    # to decide which node comes next
    graph.add_conditional_edges(
        "coordinator",       # from this node
        route_decision,      # call this function to decide
        {
            "research":   "research",    # if returns "research" → go to research node
            "calculate":  "calculator",  # if returns "calculate" → go to calculator node
        }
    )

    # After either specialist runs → END (task complete)
    graph.add_edge("research",   END)
    graph.add_edge("calculator", END)

    # Compile the graph into a runnable
    return graph.compile()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n=== LangGraph Multi-Agent System ===")
    print("Architecture: Coordinator → Research Agent | Calculator Agent")
    print("Type 'quit' to exit.\n")

    app = build_graph()

    test_questions = [
        "What is RAG in AI systems?",
        "What is 1234 * 5678?",
        "Explain how transformer models work",
        "If I have 350 items at $12.50 each, what is the total cost?",
    ]

    print("Running with test questions first...\n")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQuestion: {q}")

        # Initial state — shared memory starts here
        initial_state: AgentState = {
            "question": q,
            "route": "",
            "research_answer": "",
            "calc_answer": "",
            "final_answer": "",
            "steps": []
        }

        # Run the graph
        result = app.invoke(initial_state)

        print(f"\nAnswer: {result['final_answer']}")
        print(f"Steps taken: {' → '.join(result['steps'])}")
        print("-" * 60)

    # Interactive mode
    print("\n\nNow try your own questions:")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        initial_state: AgentState = {
            "question": question,
            "route": "",
            "research_answer": "",
            "calc_answer": "",
            "final_answer": "",
            "steps": []
        }

        result = app.invoke(initial_state)
        print(f"\nAnswer: {result['final_answer']}")
        print(f"Route taken: {' → '.join(result['steps'])}")


if __name__ == "__main__":
    main()
