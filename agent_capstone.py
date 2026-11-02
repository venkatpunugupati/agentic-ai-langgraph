# agent_capstone.py — Week 5 Day 5 capstone: the Week 2 acceptance task,
# ported to LangGraph. Same three tools, same agent node shape as Day 1,
# plus a SQLite checkpointer (Day 2 pattern) so the run persists per thread.

import ast
import operator as op
import sqlite3
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ---- Tools (same three as Day 1) ----

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '240*0.15' or '(9*3)+2'.
    Supports +, -, *, /, ** and parentheses. No percentages, no words."""
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_count(text: str) -> int:
    """Count the words in the given text.
    Example: word_count('the quick brown fox') returns 4."""
    return len(text.split())


@tool
def get_current_time() -> str:
    """Get the current time. Takes no arguments."""
    return datetime.now().strftime("%I:%M %p on %B %d, %Y")


tools = [calculator, word_count, get_current_time]


# ---- Graph (Day 1 shape) ----

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# Checkpointer (Day 2 working form: direct sqlite3 connection,
# NOT SqliteSaver.from_conn_string, which returns a context manager here).
conn = sqlite3.connect("checkpoints_capstone.sqlite", check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(conn))


# ---- Driver ----

# The Week 2 capstone acceptance task: paste the EXACT question from
# agent.py's capstone run here (the one that returned 98).
TASK = "Count the words in 'the quick brown fox jumps over the lazy dog', multiply by the current hour, then subtract 10" 

config = {"configurable": {"thread_id": "capstone-2"}}

print("=== Acceptance task ===")
for update in app.stream(
    {"messages": [HumanMessage(content=TASK)]},
    config=config,
    stream_mode="updates",
):
    print(update)

print("=== Same-thread recall (checkpoint proof) ===")
for update in app.stream(
    {"messages": [HumanMessage(content="What answer did you just compute?")]},
    config=config,
    stream_mode="updates",
):
    print(update)
