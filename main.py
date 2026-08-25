"""
main.py

CLI demo for the Medical Claim Denial Analyzer.

Usage:
    python main.py                     # run on a built-in sample letter
    python main.py --no-agent          # skip the LangGraph agent step
    python main.py --file letter.txt   # analyze your own letter
"""

import argparse
import json
import os
from dotenv import load_dotenv  
from src.pipeline import analyze_letter

load_dotenv()
print("KEY LOADED:", os.environ.get("GOOGLE_API_KEY"))

SAMPLE_LETTER = (
    "Claim number CLM-84421 for patient services rendered on 03/12/2026 "
    "has been denied. Reason: The requested procedure (CPT 29881) was "
    "performed without prior authorization on file. Please submit a "
    "retroactive authorization request or appeal with supporting "
    "documentation from the treating physician."
)


def main():
    parser = argparse.ArgumentParser(description="Analyze a medical claim denial letter.")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing the denial letter.")
    parser.add_argument("--no-agent", action="store_true", help="Skip the LangGraph agent step.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = SAMPLE_LETTER
        print("(No --file given, using a built-in sample letter.)\n")

    result = analyze_letter(text, use_agent=not args.no_agent)

  
    print("EXTRACTED FIELDS")
    print(json.dumps(result["extracted"], indent=2))

    print("PREDICTED DENIAL CATEGORY")
    print(f"{result['category']}  (confidence: {result['confidence']})")

    if result["agent_output"]:
        print("AGENT OUTPUT")
        print(result["agent_output"])


if __name__ == "__main__":
    main()
