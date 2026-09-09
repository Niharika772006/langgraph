import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
#
# NOTE: Render runs this as a long-lived web service (uvicorn app:app),
# not an interactive terminal, so there is no input() prompt available
# at startup. The GEMINI_API_KEY must be set as an environment variable
# in the Render dashboard (Environment tab) before the service starts.

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Add it under your Render service's "
        "Environment tab, then redeploy."
    )

print("API Key configured successfully.")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)


# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    test_code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        # Same namespace for globals and locals
        scope = {}

        exec(clean_code, scope, scope)

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


# ==========================================
# 4. GRAPH NODES
# ==========================================
#
# The original script had interactive nodes (task_input_node,
# manager_decision_node) that call input(). Those only make sense in a
# terminal session, not inside an HTTP request handled by a web service,
# so this version exposes the developer -> tester part of the pipeline
# through a FastAPI endpoint instead. The task now comes from the
# request body rather than input().


# ==========================================
# DEVELOPER
# ==========================================

def real_time_developer(state: CrewState):

    print("\n[Developer] Writing code using LLM...")

    task = state["messages"][-1].content

    dev_prompt = f"""
You are an expert Python developer.

Write a complete and clean Python solution for this task:

{task}

Requirements:
1. Return ONLY executable Python code.
2. Do not use markdown.
3. Put the main solution inside a clearly named function.
4. The function should return results instead of requiring input().
5. Do not add explanations.
"""

    response = llm.invoke(dev_prompt)

    code_str = str(response.content)

    # Remove markdown if model still returns it
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("\nGENERATED CODE:\n")
    print(code_str)

    return {
        "code": code_str
    }


# ==========================================
# TESTER
# ==========================================

def real_time_tester(state: CrewState):

    print("\n[Tester] Generating executable tests...")

    task = state["messages"][-1].content
    generated_code = state["code"]

    tester_prompt = f"""
You are a Senior Python QA Engineer.

TASK:
{task}

DEVELOPER CODE:
{generated_code}

Create 3 to 5 executable Python test cases for the developer code.

Requirements:
1. Return ONLY Python code.
2. Do not use markdown.
3. Use assert statements.
4. Include normal cases and edge cases.
5. Call the function that exists in the developer code.
6. Each successful test should print a message.
7. If a test fails, Python should raise an AssertionError.

Example:

assert add(2, 3) == 5
print("Test 1 passed")
"""

    response = llm.invoke(tester_prompt)

    test_code = str(response.content)

    test_code = (
        test_code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("\nGENERATED TEST CODE:\n")
    print(test_code)

    # Combine developer code and tests
    full_code = f"""
# ===== DEVELOPER CODE =====

{generated_code}


# ===== TEST CASES =====

{test_code}
"""

    # ACTUALLY RUN THE TESTS
    execution_result = run_python_code.invoke({
        "code": full_code
    })

    report = f"""
==============================
       TEST EXECUTION REPORT
==============================

TASK:
{task}

------------------------------
GENERATED CODE
------------------------------

{generated_code}

------------------------------
GENERATED TESTS
------------------------------

{test_code}

------------------------------
TEST RESULT
------------------------------

{execution_result}
"""

    return {
        "test_code": test_code,
        "report": report
    }


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
#
# Simplified graph: developer -> tester -> END.
# (The "store / try another" loop from the original CLI version doesn't
# translate to a single request/response HTTP call, so it's dropped
# here. If you want that behavior back, add a separate endpoint that
# re-invokes the graph with a new task.)

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)

rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

rt_app = rt_workflow.compile()

print("Developer-Tester pipeline compiled successfully!")


# ==========================================
# 6. FASTAPI APP
# ==========================================
#
# Render's start command is `uvicorn app:app --host 0.0.0.0 --port $PORT`,
# so this module must expose an ASGI object literally named `app`.

app = FastAPI(title="Developer-Tester Pipeline")


class TaskRequest(BaseModel):
    task: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/run")
def run_task(request: TaskRequest):
    try:
        result = rt_app.invoke(
            {
                "messages": [HumanMessage(content=request.task)],
                "next_step": None,
                "code": None,
                "test_code": None,
                "report": None
            },
            config={"recursion_limit": 50}
        )

        return {
            "code": result.get("code"),
            "test_code": result.get("test_code"),
            "report": result.get("report")
        }

    except Exception:
        return {
            "error": traceback.format_exc()
        }
