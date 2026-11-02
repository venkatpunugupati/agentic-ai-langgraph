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

# Defining State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Defining Tools     
@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '9*3' or '240*0.15'... """

    if not re.fullmatch(r"[\d\s+\-*/%().]+", expression):
     return "Error: invalid characters in expression"
    try:
     return str(eval(expression ,{"__builtins__":{}},{}))
    except Exception as e:
     return f"Error: {e}"
@tool
def word_count(text: str) -> str:
    """ Count the words from the text"""
    return str(len(text.split()))
@tool
def get_current_time() -> str:
   """Returns the current local date and time. Call this for any question about the current time or date. Takes no arguments, pass {}."""
   return  datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

#Graph 

def should_continue(state):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

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
app = graph.compile(checkpointer=sqlite_checkpointer, interrupt_before=["tools"])

question = "count the words in 'the quick brown fox jumps over the lazy dog' and multiply by 3"
config_decline = {"configurable": {"thread_id": "venkat-hitl-5"}}
for step in app.stream({"messages": [HumanMessage(content=question)]}, config=config_decline, stream_mode="updates"):
    print(step)

state = app.get_state(config_decline)
last = state.values["messages"][-1]
declined = AIMessage(content="The human declined this tool call.", tool_calls=[], id=last.id)
app.update_state(config_decline, {"messages": [declined]})

while state.next:
    app.invoke(None, config=config_decline)
    state = app.get_state(config_decline)
print("final:", state.values["messages"][-1].content)
print("tool messages:", [m.name for m in state.values["messages"] if type(m).__name__ == "ToolMessage"])


history = list(app.get_state_history(config_decline))
print("checkpoints:", len(history))
for s in history:
    print(s.metadata.get("step"), s.metadata.get("source"))

