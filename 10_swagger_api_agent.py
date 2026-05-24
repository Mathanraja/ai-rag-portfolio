"""
10_swagger_api_agent.py — Swagger / OpenAPI Test Agent

Pattern: Agentic API Test Generation
Testleaf Curriculum: Week 7–12 — Building & Deploying Agentic Workflows

Key Concepts:
  - Agent reads a Swagger/OpenAPI JSON/YAML spec autonomously
  - Generates test cases for every endpoint (happy path, error, edge)
  - Produces assertions: status codes, schema validation, response time
  - Generates mock data using Faker-style patterns
  - Outputs pytest + requests test file ready for CI/CD

Use Case:
  Given any REST API with a Swagger spec, this agent automatically generates
  a full API test suite — saving hours of manual test design and eliminating
  gaps in coverage for new endpoints.
"""

from __future__ import annotations

import json
import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
    )


# ── Sample OpenAPI Spec (inline for demo) ─────────────────────────────────────
# In production: load from file, URL, or Swagger registry

SAMPLE_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title":   "User Management API",
        "version": "1.0.0",
    },
    "servers": [{"url": "https://api.example.com/v1"}],
    "paths": {
        "/users": {
            "get": {
                "summary": "List all users",
                "parameters": [
                    {"name": "page",  "in": "query", "schema": {"type": "integer"}, "required": False},
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}, "required": False},
                    {"name": "role",  "in": "query", "schema": {"type": "string", "enum": ["admin", "user", "viewer"]}, "required": False},
                ],
                "responses": {
                    "200": {"description": "List of users"},
                    "401": {"description": "Unauthorized"},
                    "403": {"description": "Forbidden"},
                }
            },
            "post": {
                "summary": "Create a new user",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["email", "name", "role"],
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                    "name":  {"type": "string", "minLength": 2, "maxLength": 100},
                                    "role":  {"type": "string", "enum": ["admin", "user", "viewer"]},
                                    "phone": {"type": "string"},
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {"description": "User created"},
                    "400": {"description": "Validation error"},
                    "409": {"description": "Email already exists"},
                }
            }
        },
        "/users/{userId}": {
            "get": {
                "summary": "Get user by ID",
                "parameters": [
                    {"name": "userId", "in": "path", "schema": {"type": "string"}, "required": True},
                ],
                "responses": {
                    "200": {"description": "User found"},
                    "404": {"description": "User not found"},
                    "401": {"description": "Unauthorized"},
                }
            },
            "put": {
                "summary": "Update user",
                "parameters": [
                    {"name": "userId", "in": "path", "schema": {"type": "string"}, "required": True},
                ],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name":  {"type": "string"},
                                    "phone": {"type": "string"},
                                    "role":  {"type": "string", "enum": ["admin", "user", "viewer"]},
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "User updated"},
                    "404": {"description": "User not found"},
                    "400": {"description": "Validation error"},
                }
            },
            "delete": {
                "summary": "Delete user",
                "parameters": [
                    {"name": "userId", "in": "path", "schema": {"type": "string"}, "required": True},
                ],
                "responses": {
                    "204": {"description": "User deleted"},
                    "404": {"description": "User not found"},
                    "403": {"description": "Cannot delete admin user"},
                }
            }
        },
        "/auth/token": {
            "post": {
                "summary": "Generate auth token",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["email", "password"],
                                "properties": {
                                    "email":    {"type": "string"},
                                    "password": {"type": "string"},
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Token returned"},
                    "401": {"description": "Invalid credentials"},
                    "429": {"description": "Rate limit exceeded"},
                }
            }
        }
    }
}


# ── Agent State ───────────────────────────────────────────────────────────────

class SwaggerAgentState(TypedDict):
    spec:           dict             # raw OpenAPI spec
    endpoints:      list[dict]       # parsed endpoints
    test_cases:     list[dict]       # generated test cases
    mock_data:      dict             # generated mock payloads per endpoint
    pytest_code:    str              # generated pytest file
    coverage_report: dict            # endpoint × scenario coverage matrix
    final_output:   dict


# ── Node 1: Spec Parser ───────────────────────────────────────────────────────

def parse_spec(state: SwaggerAgentState) -> SwaggerAgentState:
    """Extract all endpoints with methods, params, request bodies, and response codes."""
    spec     = state["spec"]
    base_url = spec.get("servers", [{}])[0].get("url", "https://api.example.com")
    paths    = spec.get("paths", {})

    endpoints = []
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            endpoints.append({
                "path":          path,
                "method":        method.upper(),
                "full_url":      base_url + path,
                "summary":       details.get("summary", ""),
                "parameters":    details.get("parameters", []),
                "request_body":  details.get("requestBody", {}),
                "responses":     list(details.get("responses", {}).keys()),
            })

    print(f"\n  [Parser] Found {len(endpoints)} endpoints:")
    for ep in endpoints:
        print(f"    {ep['method']:<7} {ep['path']}")

    return {**state, "endpoints": endpoints}


# ── Node 2: Mock Data Generator ───────────────────────────────────────────────

def generate_mock_data(state: SwaggerAgentState) -> SwaggerAgentState:
    """
    For each endpoint with a request body, generate:
      - valid_payload  (positive test)
      - invalid_payload (negative test — missing required field)
      - edge_payload    (edge case — boundary values, special chars)
    """
    llm = get_llm(temperature=0.3)
    mock_data = {}

    endpoints_with_body = [ep for ep in state["endpoints"] if ep["request_body"]]
    spec_summary = json.dumps(
        {ep["path"] + " " + ep["method"]: ep["request_body"]
         for ep in endpoints_with_body},
        indent=2
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an API test data engineer. For each endpoint's request body schema, "
            "generate three payloads: valid, invalid (missing a required field), "
            "and edge (boundary values, special characters, max-length strings).\n\n"
            "Return JSON: {\"<path> <METHOD>\": {\"valid\": {...}, \"invalid\": {...}, \"edge\": {...}}}\n"
            "JSON only — no markdown."
        )),
        HumanMessage(content=f"Endpoint schemas:\n{spec_summary}")
    ])

    response = llm.invoke(prompt.format_messages())
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        mock_data = json.loads(raw.strip())
    except json.JSONDecodeError:
        mock_data = {"parse_error": raw[:200]}

    print(f"\n  [MockGen] Generated payloads for {len(endpoints_with_body)} endpoint(s)")
    return {**state, "mock_data": mock_data}


# ── Node 3: Test Case Generator ───────────────────────────────────────────────

def generate_test_cases(state: SwaggerAgentState) -> SwaggerAgentState:
    """
    Generate structured test cases for every endpoint covering:
    - Happy path (200/201/204)
    - Auth failure (401/403)
    - Not found (404)
    - Validation error (400/422)
    - Edge cases (empty params, SQL injection in path, rate limits)
    """
    llm = get_llm(temperature=0.2)

    endpoints_summary = json.dumps(state["endpoints"], indent=2)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior API test engineer. For each endpoint, generate test cases "
            "covering all documented response codes plus common edge cases.\n\n"
            "Return a JSON array where each item has:\n"
            "  endpoint_path, method, test_id, scenario (happy|auth|validation|notfound|edge),\n"
            "  description, setup (dict), expected_status (int), assertions (list of strings)\n\n"
            "JSON array only — no markdown."
        )),
        HumanMessage(content=f"Endpoints:\n{endpoints_summary}")
    ])

    response = llm.invoke(prompt.format_messages())
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        test_cases = json.loads(raw.strip())
    except json.JSONDecodeError:
        test_cases = [{"test_id": "TC_PARSE_ERROR", "description": raw[:200]}]

    print(f"\n  [TestGen] Generated {len(test_cases)} test cases")
    scenario_counts = {}
    for tc in test_cases:
        s = tc.get("scenario", "unknown")
        scenario_counts[s] = scenario_counts.get(s, 0) + 1
    for scenario, count in scenario_counts.items():
        print(f"    • {scenario:<12} {count}")

    return {**state, "test_cases": test_cases}


# ── Node 4: Pytest Code Generator ────────────────────────────────────────────

def generate_pytest_code(state: SwaggerAgentState) -> SwaggerAgentState:
    """Generate a ready-to-run pytest file from the test cases."""
    llm = get_llm(temperature=0.1)

    # Pass top 8 test cases to stay within token limits for demo
    cases_subset = json.dumps(state["test_cases"][:8], indent=2)
    mock_data_str = json.dumps(state["mock_data"], indent=2)
    base_url = state["spec"].get("servers", [{}])[0].get("url", "https://api.example.com")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a Python test engineer. Generate a complete pytest test file using "
            "the `requests` library for the given API test cases.\n\n"
            "Requirements:\n"
            "  - Import: requests, pytest, os\n"
            "  - BASE_URL from environment variable API_BASE_URL\n"
            "  - AUTH_TOKEN from environment variable API_TOKEN\n"
            "  - session fixture with auth header\n"
            "  - Each test function: test_<test_id>_<scenario>()\n"
            "  - Assertions: status code, response time < 2s, JSON schema spot-checks\n"
            "  - @pytest.mark.parametrize for data-driven cases where applicable\n"
            "  - TODO comments for values that need real test data\n\n"
            "Output Python code only."
        )),
        HumanMessage(content=(
            f"BASE_URL: {base_url}\n\n"
            f"Test cases:\n{cases_subset}\n\n"
            f"Mock payloads:\n{mock_data_str}"
        ))
    ])

    response = llm.invoke(prompt.format_messages())
    print(f"\n  [PytestGen] Test file generated ({len(response.content.splitlines())} lines)")
    return {**state, "pytest_code": response.content}


# ── Node 5: Coverage Report ───────────────────────────────────────────────────

def build_coverage_report(state: SwaggerAgentState) -> SwaggerAgentState:
    """Build a coverage matrix: which endpoints × which scenarios are covered."""
    matrix = {}
    for ep in state["endpoints"]:
        key = f"{ep['method']} {ep['path']}"
        covered = [
            tc.get("scenario", "unknown")
            for tc in state["test_cases"]
            if tc.get("endpoint_path") == ep["path"] and tc.get("method") == ep["method"]
        ]
        matrix[key] = {
            "expected_codes": ep["responses"],
            "scenarios_covered": list(set(covered)),
            "test_count": len(covered),
        }

    total_endpoints = len(state["endpoints"])
    covered_endpoints = sum(1 for v in matrix.values() if v["test_count"] > 0)

    report = {
        "matrix": matrix,
        "total_endpoints": total_endpoints,
        "covered_endpoints": covered_endpoints,
        "total_test_cases": len(state["test_cases"]),
        "coverage_pct": round(covered_endpoints / total_endpoints * 100, 1) if total_endpoints else 0,
    }

    print(f"\n  [Coverage] {covered_endpoints}/{total_endpoints} endpoints covered "
          f"({report['coverage_pct']}%)")
    return {**state, "coverage_report": report}


# ── Node 6: Finalise ──────────────────────────────────────────────────────────

def finalise(state: SwaggerAgentState) -> SwaggerAgentState:
    final = {
        "api_title":      state["spec"].get("info", {}).get("title", "Unknown"),
        "endpoints":      state["endpoints"],
        "test_cases":     state["test_cases"],
        "pytest_code":    state["pytest_code"],
        "coverage_report": state["coverage_report"],
        "mock_data":      state["mock_data"],
    }
    return {**state, "final_output": final}


# ── Build LangGraph Workflow ───────────────────────────────────────────────────

def build_swagger_agent() -> StateGraph:
    graph = StateGraph(SwaggerAgentState)

    graph.add_node("parse",    parse_spec)
    graph.add_node("mock",     generate_mock_data)
    graph.add_node("testgen",  generate_test_cases)
    graph.add_node("pytest",   generate_pytest_code)
    graph.add_node("coverage", build_coverage_report)
    graph.add_node("finalise", finalise)

    graph.set_entry_point("parse")
    graph.add_edge("parse",    "mock")
    graph.add_edge("mock",     "testgen")
    graph.add_edge("testgen",  "pytest")
    graph.add_edge("pytest",   "coverage")
    graph.add_edge("coverage", "finalise")
    graph.add_edge("finalise", END)

    return graph.compile()


# ── Print Results ─────────────────────────────────────────────────────────────

def print_results(output: dict):
    final = output.get("final_output", {})
    cov   = final.get("coverage_report", {})

    print("\n" + "=" * 70)
    print(f"  SWAGGER API AGENT — {final.get('api_title', 'API')}")
    print("=" * 70)
    print(f"  Endpoints    : {cov.get('total_endpoints', 0)}")
    print(f"  Test Cases   : {cov.get('total_test_cases', 0)}")
    print(f"  Coverage     : {cov.get('coverage_pct', 0)}%")

    print("\n  ── Coverage Matrix ──────────────────────────────────────────")
    for endpoint, data in cov.get("matrix", {}).items():
        scenarios = ", ".join(data["scenarios_covered"]) or "none"
        print(f"    {endpoint:<35} {data['test_count']:>2} tests  [{scenarios}]")

    print("\n  ── Generated pytest file (first 40 lines) ──────────────────")
    pytest_lines = final.get("pytest_code", "").splitlines()
    for line in pytest_lines[:40]:
        print(f"    {line}")
    if len(pytest_lines) > 40:
        print(f"    ... ({len(pytest_lines) - 40} more lines)")

    print("\n" + "─" * 70)
    print("  To save the pytest file:")
    print("    with open('test_api_generated.py', 'w') as f:")
    print("        f.write(result['final_output']['pytest_code'])")
    print("  Then: pytest test_api_generated.py -v")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Swagger / OpenAPI Test Agent")
    print("  Pattern : LangGraph workflow — spec → mock data → tests → pytest")
    print("  Output  : Coverage matrix + executable pytest file")
    print("=" * 70)

    agent = build_swagger_agent()

    initial_state: SwaggerAgentState = {
        "spec":            SAMPLE_OPENAPI_SPEC,
        "endpoints":       [],
        "test_cases":      [],
        "mock_data":       {},
        "pytest_code":     "",
        "coverage_report": {},
        "final_output":    {},
    }

    result = agent.invoke(initial_state)
    print_results(result)

    print("\n  LangGraph nodes: parse → mock → testgen → pytest → coverage → finalise")
    print("  Same agentic pattern as production API automation pipelines.\n")


if __name__ == "__main__":
    main()
