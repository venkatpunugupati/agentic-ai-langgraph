from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import re
from langchain_core.tools import tool
from datetime import datetime
from langchain_core.messages import HumanMessage

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
app = graph.compile()
#print(app.get_graph().draw_mermaid())
questions = [
    "count the words in 'the quick brown fox jumps over the lazy dog' and multiply by 3",
    "what time is it?",
    "what is 15% of 240?",
    "What is the Capital of USA",
]
for question in questions:  
    print("Q:--",question)
    for step in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode="updates"):
        print("@@@@@@@@@@@",question)
        print(step)
