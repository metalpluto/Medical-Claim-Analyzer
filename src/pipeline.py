"""
pipeline.py

Ties together the three stages:
    1. extractor.py  -> structured fields (NLP: information extraction)
    2. classifier.py -> denial category    (NLP: text classification)
    3. agent.py       -> next-action / appeal draft (LangGraph agent)

Stage 3 requires a GOOGLE_API_KEY    Stages 1-2 run fully offline
"""

from . import extractor, classifier


def analyze_letter(text: str, use_agent: bool = True) -> dict:
    extracted = extractor.extract(text).to_dict()
    prediction = classifier.predict(text)

    result = {
        "extracted": extracted,
        "category": prediction["category"],
        "confidence": prediction["confidence"],
        "agent_output": None,
    }

    if use_agent:
        from . import agent  # deferred import: only needed if agent runs
        try:
            result["agent_output"] = agent.run_agent(
                text=text,
                extracted=extracted,
                category=prediction["category"],
                confidence=prediction["confidence"],
            )
        except RuntimeError as e:
            result["agent_output"] = f"[skipped: {e}]"

    return result
