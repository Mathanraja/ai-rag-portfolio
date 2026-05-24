"""
8_testcase_generator_agent.py — Testcase Generator Agent (LangChain + LangGraph)

Pattern: Agentic Test Generation
Testleaf Curriculum: Week 4–5 — LangChain & LangGraph (Hands-On Project)

Key Concepts:
  - LangGraph workflow: requirements → analyse → generate → validate → output
  - LangChain prompt engineering for structured test case DSL
  - Memory-enabled agent: stores prior generations to avoid duplication
  - Output: structured Gherkin BDD + Playwright script per test case
  - Supports user stories, acceptance criteria, and free-text requirements

Use Case:
  Ingest a user story or feature requirement and autonomously produce
  a full suite of test cases (positive, negative, edge) with BDD scenarios
  and executable Playwright test stubs — ready for a CI/CD pipeline.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.2):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
    )


# ── Agent State (shared across all LangGraph nodes) ───────────────────────────

class TestGenState(TypedDict):
    requirement:      str            # raw input from user
    parsed_features:  list[str]      # extracted functional features
    test_scenarios:   list[dict]     # positive / negative / edge cases
    gherkin_output:   str            # BDD Gherkin feature file text
    playwright_stubs: str            # Playwright test stubs (TypeScript)
    memory:           list[str]      # prior requirement summaries (dedup)
    validation_notes: str            # QA review of generated cases
    final_output:     dict           # structured final deliverable


# ── Node 1: Requirement Analyser ──────────────────────────────────────────────

def analyse_requirements(state: TestGenState) -> TestGenState:
    """
    Parse the raw requirement into discrete functional features.
    This primes the generator with a clear scope, reducing hallucination.
    """
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior QA analyst. Extract discrete, testable functional "
            "features from the requirement. Return a JSON array of strings only — "
            "no markdown, no explanation."
        )),
        HumanMessage(content=f"Requirement:\n{state['requirement']}")
    ])

    response = llm.invoke(prompt.format_messages())
    raw = response.content.strip()

    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        features = json.loads(raw.strip())
    except json.JSONDecodeError:
        features = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    print(f"\n  [Analyser] Extracted {len(features)} feature(s):")
    for f in features:
        print(f"    • {f}")

    return {**state, "parsed_features": features}


# ── Node 2: Test Case Generator ───────────────────────────────────────────────

def generate_test_cases(state: TestGenState) -> TestGenState:
    """
    For each extracted feature, generate positive, negative, and edge test cases
    as a structured JSON array with id, type, title, steps, expected_result.
    """
    llm = get_llm(temperature=0.3)

    features_block = "\n".join(f"- {f}" for f in state["parsed_features"])
    memory_block   = (
        "Previously generated requirements (avoid duplicates):\n" +
        "\n".join(f"- {m}" for m in state["memory"])
        if state["memory"] else "No prior history."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior QA engineer. Generate comprehensive test cases "
            "covering positive, negative, and edge scenarios for the given features.\n\n"
            "Return a JSON array where each item has:\n"
            "  id, type (positive|negative|edge), title, preconditions (list), "
            "  steps (list), expected_result\n\n"
            "Return JSON only — no markdown, no explanation."
        )),
        HumanMessage(content=(
            f"Features to test:\n{features_block}\n\n"
            f"Context:\n{memory_block}"
        ))
    ])

    response = llm.invoke(prompt.format_messages())
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        scenarios = json.loads(raw.strip())
    except json.JSONDecodeError:
        scenarios = [{"id": "TC001", "type": "positive",
                      "title": "Manual review required", "steps": [], "expected_result": raw}]

    print(f"\n  [Generator] Created {len(scenarios)} test case(s):")
    for tc in scenarios:
        print(f"    [{tc.get('type','?'):8s}] {tc.get('id','?')} — {tc.get('title','?')}")

    return {**state, "test_scenarios": scenarios}


# ── Node 3: Gherkin BDD Writer ────────────────────────────────────────────────

def write_gherkin(state: TestGenState) -> TestGenState:
    """
    Convert structured test cases into a Gherkin feature file.
    """
    llm = get_llm(temperature=0.1)

    cases_json = json.dumps(state["test_scenarios"], indent=2)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a BDD expert. Convert the JSON test cases into a valid Gherkin "
            "feature file (.feature). Use Feature, Background (if applicable), "
            "Scenario, Given/When/Then/And keywords. Output plain text only."
        )),
        HumanMessage(content=(
            f"Feature: {state['requirement'][:80]}\n\n"
            f"Test cases:\n{cases_json}"
        ))
    ])

    response = llm.invoke(prompt.format_messages())
    print(f"\n  [Gherkin] BDD feature file written ({len(response.content.splitlines())} lines)")
    return {**state, "gherkin_output": response.content}


# ── Node 4: Playwright Stub Generator ────────────────────────────────────────

def generate_playwright_stubs(state: TestGenState) -> TestGenState:
    """
    Generate executable Playwright TypeScript test stubs from the test cases.
    Each stub includes page object navigation, locator placeholders, and assertions.
    """
    llm = get_llm(temperature=0.1)

    cases_json = json.dumps(state["test_scenarios"][:6], indent=2)  # cap at 6 for LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a Playwright automation expert. Generate TypeScript Playwright "
            "test stubs for the given test cases. Use:\n"
            "  - import {{ test, expect }} from '@playwright/test'\n"
            "  - test.describe() groups\n"
            "  - async ({ page }) => {} pattern\n"
            "  - TODO comments for locators that need real selectors\n"
            "Output TypeScript code only."
        )),
        HumanMessage(content=f"Test cases:\n{cases_json}")
    ])

    response = llm.invoke(prompt.format_messages())
    print(f"\n  [Playwright] Test stubs generated ({len(response.content.splitlines())} lines)")
    return {**state, "playwright_stubs": response.content}


# ── Node 5: Validator ─────────────────────────────────────────────────────────

def validate_output(state: TestGenState) -> TestGenState:
    """
    QA self-review: check for coverage gaps, missing negative/edge cases,
    and flag any vague or untestable scenarios.
    """
    llm = get_llm(temperature=0.0)

    summary = {
        "total_cases":     len(state["test_scenarios"]),
        "positive_cases":  sum(1 for t in state["test_scenarios"] if t.get("type") == "positive"),
        "negative_cases":  sum(1 for t in state["test_scenarios"] if t.get("type") == "negative"),
        "edge_cases":      sum(1 for t in state["test_scenarios"] if t.get("type") == "edge"),
        "features_covered": len(state["parsed_features"]),
    }

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a QA lead reviewing test coverage. Given the summary and test cases, "
            "identify: (1) coverage gaps, (2) missing scenarios, (3) quality issues. "
            "Be concise — 3-5 bullet points max."
        )),
        HumanMessage(content=(
            f"Summary: {json.dumps(summary)}\n\n"
            f"Requirements: {state['requirement']}\n\n"
            f"Features: {state['parsed_features']}"
        ))
    ])

    response = llm.invoke(prompt.format_messages())
    print(f"\n  [Validator] Review complete")

    # Update memory with this requirement
    memory_entry = state["requirement"][:100]
    updated_memory = (state["memory"] or []) + [memory_entry]

    return {**state, "validation_notes": response.content, "memory": updated_memory}


# ── Node 6: Finalise Output ───────────────────────────────────────────────────

def finalise_output(state: TestGenState) -> TestGenState:
    final = {
        "requirement":      state["requirement"],
        "features_extracted": state["parsed_features"],
        "test_cases":       state["test_scenarios"],
        "gherkin":          state["gherkin_output"],
        "playwright_stubs": state["playwright_stubs"],
        "validation_notes": state["validation_notes"],
        "summary": {
            "total":    len(state["test_scenarios"]),
            "positive": sum(1 for t in state["test_scenarios"] if t.get("type") == "positive"),
            "negative": sum(1 for t in state["test_scenarios"] if t.get("type") == "negative"),
            "edge":     sum(1 for t in state["test_scenarios"] if t.get("type") == "edge"),
        },
    }
    return {**state, "final_output": final}


# ── Build LangGraph Workflow ───────────────────────────────────────────────────

def build_testgen_graph() -> StateGraph:
    graph = StateGraph(TestGenState)

    graph.add_node("analyse",    analyse_requirements)
    graph.add_node("generate",   generate_test_cases)
    graph.add_node("gherkin",    write_gherkin)
    graph.add_node("playwright", generate_playwright_stubs)
    graph.add_node("validate",   validate_output)
    graph.add_node("finalise",   finalise_output)

    graph.set_entry_point("analyse")
    graph.add_edge("analyse",    "generate")
    graph.add_edge("generate",   "gherkin")
    graph.add_edge("gherkin",    "playwright")
    graph.add_edge("playwright", "validate")
    graph.add_edge("validate",   "finalise")
    graph.add_edge("finalise",   END)

    return graph.compile()


# ── Pretty Print Results ──────────────────────────────────────────────────────

def print_results(output: dict):
    final = output.get("final_output", {})
    summary = final.get("summary", {})

    print("\n" + "=" * 70)
    print("  TESTCASE GENERATOR — OUTPUT SUMMARY")
    print("=" * 70)
    print(f"  Total test cases : {summary.get('total', 0)}")
    print(f"  Positive         : {summary.get('positive', 0)}")
    print(f"  Negative         : {summary.get('negative', 0)}")
    print(f"  Edge cases       : {summary.get('edge', 0)}")

    print("\n  ── Gherkin BDD Feature File ─────────────────────────────────")
    print(final.get("gherkin", ""))

    print("\n  ── Playwright TypeScript Stubs ──────────────────────────────")
    print(final.get("playwright_stubs", ""))

    print("\n  ── Validation / Coverage Notes ──────────────────────────────")
    print(final.get("validation_notes", ""))
    print("=" * 70)


# ── Demo Requirements ─────────────────────────────────────────────────────────

SAMPLE_REQUIREMENTS = [
    """
    User Story: As a registered user, I want to log in to the application
    using my email and password so that I can access my personalised dashboard.

    Acceptance Criteria:
    - Valid email + correct password → redirect to dashboard
    - Invalid credentials → show error message, do not redirect
    - Account locked after 5 failed attempts
    - 'Forgot password' link visible on login page
    - Session persists for 30 minutes of inactivity
    """,

    """
    Feature: Product Search
    Users must be able to search for products by keyword, filter by category
    and price range, and sort results by relevance or price.
    Results page must load in under 2 seconds for up to 1000 results.
    """
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Testcase Generator Agent")
    print("  Pattern : LangGraph workflow — requirements → test cases")
    print("  Output  : Structured JSON + Gherkin BDD + Playwright stubs")
    print("=" * 70)

    agent = build_testgen_graph()
    memory: list[str] = []   # persists across runs in the same session

    for i, requirement in enumerate(SAMPLE_REQUIREMENTS, 1):
        print(f"\n{'─' * 70}")
        print(f"  Processing Requirement {i}/{len(SAMPLE_REQUIREMENTS)}")
        print(f"{'─' * 70}")

        initial_state: TestGenState = {
            "requirement":      requirement.strip(),
            "parsed_features":  [],
            "test_scenarios":   [],
            "gherkin_output":   "",
            "playwright_stubs": "",
            "memory":           memory,
            "validation_notes": "",
            "final_output":     {},
        }

        result = agent.invoke(initial_state)
        print_results(result)

        # carry memory forward to next requirement (dedup)
        memory = result.get("memory", [])

    print("\n  LangGraph nodes executed: analyse → generate → gherkin → playwright → validate → finalise")
    print("  Shared TestGenState flowed through all nodes — same pattern as production agentic pipelines.\n")


if __name__ == "__main__":
    main()
