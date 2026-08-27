"""
batch.py

Batch mode: process every denial letter (.txt file) in a folder at
once, instead of one file at a time, and write a single CSV summary
report covering all of them.
"""

import csv
from pathlib import Path

from .pipeline import analyze_letter


def run_batch(folder_path: str, output_path: str = "batch_results.csv", use_agent: bool = False) -> list[dict]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a folder")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")

    results = []
    for i, file in enumerate(txt_files, start=1):
        text = file.read_text(encoding="utf-8")
        print(f"[{i}/{len(txt_files)}] Analyzing {file.name}...")
        result = analyze_letter(text, use_agent=use_agent)
        results.append({
            "file": file.name,
            "claim_number": result["extracted"].get("claim_number"),
            "category": result["category"],
            "confidence": result["confidence"],
            "agent_output": result["agent_output"] or "",
        })

    _write_csv(results, output_path)
    print(f"\nBatch complete. {len(results)} letter(s) processed.")
    print(f"Results saved to {output_path}")
    return results


def _write_csv(results: list[dict], output_path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
