# LangGraph + Multi-Agent 

## What's inside

| File | What it builds |
| --- | --- | --- |
| `agent_graph.py` | The graph model: StateGraph, agent + ToolNode, hand-written `should_continue` edge |
| `agent_checkpoint.py` |Checkpointing: InMemorySaver and SQLite persistence, keyed by thread |
| `agent_hitl.py` | Static human-in-the-loop: `interrupt_before=["tools"]`, approve / edit / decline |
| `agent_hitl_decline.py` | The decline path, isolated: no tool execution on decline |
| `agent_hitl_dynamic.py` | Dynamic gate: `interrupt()` inside a tool when a value crosses a threshold |
| `agent_multi.py` | Supervisor / worker multi-agent: centralized LLM routing, narrow workers |
| `agent_capstone.py` | Capstone |

Tools throughout: `word_count`, `calculator`, `get_current_time`
(model: `gpt-4o-mini`, temperature 0).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install langgraph langchain-openai
export OPENAI_API_KEY="your-key-here"


If the SQLite checkpointer import fails, also install
langgraph-checkpoint-sqlite. The key is never stored in the repo — it comes
from the local shell export only.


Run it
bash
python agent_graph.py
python agent_checkpoint.py
python agent_hitl.py
python agent_multi.py
python agent_capstone.py

Note: agent_capstone.py runs the TASK string hardcoded in the file —
command-line arguments are ignored.
Verified results
Graph model: word-count ×3 traced agent → tools → agent → tools →
agent → END = 27; 15% of 240 = 36.0, including a live error-observation
self-correction (calculator('15% of 240') rejected, then 0.15 * 240).
Checkpointing: 3 checkpoints per simple turn (1 input + 2 loop), step
numbers continuous across turns, history append-only. A fresh process
recalled state from checkpoints.sqlite — persistence across processes,
not just turns.
Human-in-the-loop: approve → 27, edit → 12 (4 words ×3), decline → no
tool execution. Dynamic gate: 9×3 ran with no interrupt, 500×3 interrupted
and approved to 1500. On a decline, the verdict rests on the ToolMessage
("Declined by human"), not the final text — the model may still answer
from its own arithmetic.
Supervisor: word-count ×3 routed supervisor → counter → supervisor →
calculator → supervisor → FINISH = 27; 9×3 went straight to the calculator;
a time question ended FINISH with "No worker can do it" and no worker node
running. The supervisor decides WHO, the worker decides HOW.
Capstone: the Week 2 acceptance question — count the words, multiply by
the current hour, subtract 10 — returned (9 × 12) − 10 = 98 on Week 2's raw
agent and (9 × 14) − 10 = 116 on LangGraph at hour 14. Same formula,
different hour: a match, not a mismatch.
