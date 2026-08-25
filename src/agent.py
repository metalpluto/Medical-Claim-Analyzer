"""
agent.py

LangGraph agent that takes the structured output of extractor.py +
classifier.py and decides what should happen next: either draft an
appeal letter (for denials that are usually appealable) or recommend
an internal fix (for denials that need correction before resubmission,
like coding errors).

Graph shape:

    [analyze] --> [decide_action] --+--> [draft_appeal] --> END
                                      +-> [recommend_fix]  --> END

Requires GOOGLE_API_KEY set in the environment (or a .env file).
Run `pip install -r requirements.txt` first.
"""

import os
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


class ClaimState(TypedDict):
    text: str
    extracted: dict
    category: str
    confidence: float
    action: Literal["appeal", "fix"] | None
    output: str


APPEALABLE = {"medical_necessity", "missing_authorization", "eligibility_issue"}
NEEDS_FIX = {"coding_error", "timely_filing", "duplicate_claim"}

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_output_tokens=500)


def _extract_text(content) -> str:
    """Normalize an LLM response's .content into a plain string.

    Newer langchain-google-genai versions can return .content as a
    list of content blocks (e.g. [{"type": "text", "text": "...", ...}])
    instead of a plain string. Pull just the text parts out.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def analyze_node(state: ClaimState) -> ClaimState:
    # extracted + category/confidence are already populated by the
    # pipeline before the graph runs; this node is a hook for any
    # additional LLM-based analysis (e.g. summarizing the letter).
    return state


def decide_action_node(state: ClaimState) -> ClaimState:
    if state["category"] in APPEALABLE:
        state["action"] = "appeal"
    else:
        state["action"] = "fix"
    return state


def route_action(state: ClaimState) -> str:
    return "draft_appeal" if state["action"] == "appeal" else "recommend_fix"


def draft_appeal_node(state: ClaimState) -> ClaimState:
    prompt = f"""You are a healthcare revenue cycle assistant. Draft a concise,
professional appeal letter for this denied medical claim.

Denial category: {state['category']}
Extracted claim details: {state['extracted']}
Original denial text: {state['text']}

Keep the appeal under 150 words, reference the claim number and codes,
and clearly state why the denial should be reversed."""
    response = llm.invoke(prompt)
    state["output"] = _extract_text(response.content)
    return state


def recommend_fix_node(state: ClaimState) -> ClaimState:
    prompt = f"""You are a healthcare revenue cycle assistant. This claim was
denied for a reason that needs internal correction before resubmission,
not an appeal.

Denial category: {state['category']}
Extracted claim details: {state['extracted']}
Original denial text: {state['text']}

In under 100 words, explain exactly what needs to be corrected and
the recommended next step for the billing team."""
    response = llm.invoke(prompt)
    state["output"] = _extract_text(response.content)
    return state


def build_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("draft_appeal", draft_appeal_node)
    graph.add_node("recommend_fix", recommend_fix_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "decide_action")
    graph.add_conditional_edges("decide_action", route_action, {
        "draft_appeal": "draft_appeal",
        "recommend_fix": "recommend_fix",
    })
    graph.add_edge("draft_appeal", END)
    graph.add_edge("recommend_fix", END)

    return graph.compile()


def run_agent(text: str, extracted: dict, category: str, confidence: float) -> str:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Add it to a .env file or your "
            "environment before running the agent step."
        )
    app = build_graph()
    result = app.invoke({
        "text": text,
        "extracted": extracted,
        "category": category,
        "confidence": confidence,
        "action": None,
        "output": "",
    })
    return result["output"]
