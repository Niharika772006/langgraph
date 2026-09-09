import os
import sys
import io
import traceback
from typing import TypedDict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it in your Render service's Environment settings."
    )

# Override with GEMINI_MODEL env var if you need a different model name.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    task: str
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _extract_text(content) -> str:
    """Safely parse Gemini's content format (plain string or list-of-dict)."""
    if isinstance(content, list):
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else str(first)
    return str(content)


# ==========================================
# 4. GRAPH NODES
# ==========================================
def developer_node(state: CrewState):
    task = state["task"]
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    response = llm.invoke(dev_prompt)
    code_str = _extract_text(response.content)
    return {"code": code_str}


def tester_node(state: CrewState):
    task = state["task"]

    raw_test_cases = generate_test_cases.invoke(task)
    cases_str = (
        _extract_text(raw_test_cases)
        if isinstance(raw_test_cases, list)
        else str(raw_test_cases)
    )

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### GENERATED CODE\n```python\n{state['code']}\n```\n\n"
        f"### EXECUTION OUTPUT\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED\n{cases_str}"
    )
    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

graph_app = workflow.compile()


# ==========================================
# 6. INPUT/OUTPUT SCHEMAS FOR THE PLAYGROUND
# ==========================================
class GraphInput(BaseModel):
    task: str


class GraphOutput(BaseModel):
    task: str
    code: Optional[str] = None
    report: Optional[str] = None


runnable = graph_app.with_types(input_type=GraphInput, output_type=GraphOutput)


# ==========================================
# 7. FASTAPI + LANGSERVE APP
# ==========================================
app = FastAPI(
    title="Real-Time Dev/Tester Crew",
    version="1.0",
    description="A LangGraph agent that writes Python code for a task and tests it.",
)


@app.get("/")
async def root():
    return {"message": "Visit /agent/playground/ to use the app."}


add_routes(
    app,
    runnable,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
