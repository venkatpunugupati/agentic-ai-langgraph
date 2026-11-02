from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import re

# Shared state: messages + supervisor routing decision
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '9*3' or '240*0.15'... """
    if not re.fullmatch(r"[\d\s+\-*/%().]+", expression):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool
def word_count(text: str) -> str:
    """Count the words from the text"""
    return str(len(text.split()))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SUPERVISOR_PROMPT = """You are a supervisor managing two workers:
- counter: counts words in a text (word_count tool)
- calculator: evaluates arithmetic (calculator tool)
Look at the conversation. If the next step needs a worker, reply with exactly one word: counter or calculator.
If the task is already complete, reply with FINISH: followed by the final answer.
If no worker can do the task, reply with FINISH: followed by an honest statement that no worker can do it."""

def supervisor_node(state):
    response = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    text = response.content.strip()
    if text.upper().startswith("FINISH"):
        nxt = "FINISH"
    else:
        words = text.split()
        word = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
        nxt = word if word in ("counter", "calculator") else "FINISH"
    return {"messages": [response], "next": nxt}

COUNTER_PROMPT = "You count words. Always call word_count with the text to count, nothing else."
CALCULATOR_PROMPT = "You evaluate arithmetic. Always call calculator with the expression, nothing else."

def counter_node(state):
    worker_llm = llm.bind_tools([word_count])
    response = worker_llm.invoke([SystemMessage(content=COUNTER_PROMPT)] + state["messages"])
    if not response.tool_calls:
        return {"messages": [response]}
    call = response.tool_calls[0]
    result = word_count.invoke(call["args"])
    return {"messages": [response, ToolMessage(content=result, name="word_count", tool_call_id=call["id"])]}

def calculator_node(state):
    worker_llm = llm.bind_tools([calculator])
    response = worker_llm.invoke([SystemMessage(content=CALCULATOR_PROMPT)] + state["messages"])
    if not response.tool_calls:
        return {"messages": [response]}
    call = response.tool_calls[0]
    result = calculator.invoke(call["args"])
    return {"messages": [response, ToolMessage(content=result, name="calculator", tool_call_id=call["id"])]}

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("counter", counter_node)
graph.add_node("calculator", calculator_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"],
    {"counter": "counter", "calculator": "calculator", "FINISH": END})
graph.add_edge("counter", "supervisor")
graph.add_edge("calculator", "supervisor")
app = graph.compile()

def run(question):
    print(f"=== {question} ===")
    for step in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode="updates"):
        print(step)
    print()

run("count the words in 'the quick brown fox jumps over the lazy dog' and multiply by 3")
run("What is 9 * 3?")
run("What time is it?")
