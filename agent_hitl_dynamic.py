from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import re
from langchain_core.tools import tool
from datetime import datetime
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.types import interrupt, Command

# Defining State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Defining Tools     

@tool
def word_count(text: str) -> str:
    """ Count the words from the text"""
    return str(len(text.split()))
@tool
def get_current_time() -> str:
   """Returns the current local date and time. Call this for any question about the current time or date. Takes no arguments, pass {}."""
   return  datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '9*3' or '240*0.15'... """
    if not re.fullmatch(r"[\d\s+\-*/%().]+", expression):
        return "Error: invalid characters in expression"
    # Dynamic gate: pause only when a number exceeds 100
    big = [int(n) for n in re.findall(r"\d+", expression) if int(n) > 100]
    if big:
        decision = interrupt({
            "tool": "calculator",
            "expression": expression,
            "reason": f"numbers over 100: {big}",
        })
        if decision != "approve":
            return "Declined by human."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

# Tools Node
tools = [calculator, word_count, get_current_time]
tools_node = ToolNode(tools)
# Augment the LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

#Agent Node
def agent_node(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

#Conditional Edge 

def should_continue(state):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END
#Graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
#print(app.get_graph().draw_mermaid())
# 1. Initialize the checkpointer
#checkpointer = InMemorySaver()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
sqlite_checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=sqlite_checkpointer)

def ask(thread_id, question):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"=== {thread_id}: {question} ===")
    for step in app.stream({"messages": [HumanMessage(content=question)]}, config=config, stream_mode="updates"):
        print(step)
    return config

# Control: no pause expected, straight to 27
ask("venkat-dyn-1", "What is 9 * 3?")

# Gated approve: expect __interrupt__ with the payload, then 1500
cfg = ask("venkat-dyn-2", "What is 500 * 3?")
for step in app.stream(Command(resume="approve"), config=cfg, stream_mode="updates"):
    print(step)

# Gated decline: expect __interrupt__, then a declined tool result, no 1500
cfg = ask("venkat-dyn-3", "What is 500 * 3?")
for step in app.stream(Command(resume="decline"), config=cfg, stream_mode="updates"):
    print(step)

history = list(app.get_state_history(cfg))
print("checkpoints:", len(history))
for s in history:
    print(s.metadata.get("step"), s.metadata.get("source"))

