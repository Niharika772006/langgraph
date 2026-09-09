import os
import sys
import io
import getpass
import traceback
from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

try:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        # Fall back to an interactive, hidden prompt if the env var
        # isn't set (useful when running the script locally/CI).
        api_key = getpass.getpass("Enter your GEMINI_API_KEY: ").strip()

    if not api_key:
        raise ValueError("GEMINI_API_KEY is empty or unavailable.")

    os.environ["GEMINI_API_KEY"] = api_key

    print("API Key configured successfully.")

except Exception as e:
    raise ValueError(
        f"Unable to load GEMINI_API_KEY: {e}"
    )


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

def task_input_node(state: CrewState):

    print("\n" + "=" * 50)
    print("--- NEW TASK INITIALIZATION ---")

    user_task = input(
        "Enter the coding task (or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":
        return {
            "next_step": "exit"
        }

    return {
        "messages": [
            HumanMessage(content=user_task)
        ],
        "next_step": "developer"
    }


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
# MANAGER
# ==========================================

def manager_decision_node(state: CrewState):

    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : TEST REPORT ---")
    print(state.get("report", "No report available."))
    print("=" * 50)

    user_input = input(
        "\nCommand (store / another): "
    ).lower().strip()

    if user_input == "store":
        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "task_input"
    }


# ==========================================
# ARCHIVER
# ==========================================

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] Task stored successfully. "
        "Closing workflow."
    )

    return {
        "next_step": "exit"
    }


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================

rt_workflow = StateGraph(CrewState)


rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# Start workflow

rt_workflow.add_edge(
    START,
    "task_input"
)


# Route after input

def route_from_input(state):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# Main workflow

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# Route after manager decision

def route_from_decision(state):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


rt_workflow.add_edge(
    "archiver",
    END
)


# Compile

rt_app = rt_workflow.compile()

print(
    "Interactive Developer-Tester pipeline "
    "compiled successfully!"
)


# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":

    try:

        rt_app.invoke(
            {
                "messages": [],
                "next_step": None,
                "code": None,
                "test_code": None,
                "report": None
            },
            config={
                "recursion_limit": 50
            }
        )

    except KeyboardInterrupt:

        print("\nStopped by user.")

    except Exception as e:

        print(
            f"\nAn error occurred:\n"
            f"{traceback.format_exc()}"
        )
