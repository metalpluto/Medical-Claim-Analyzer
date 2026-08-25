"""
extractor.py

Rule-based NLP entity extraction for medical claim denial letters.

This is classic NLP: turning unstructured free text into structured
fields a downstream system (classifier, agent, database) can use.
Real-world claims text is messy — inconsistent formatting, abbreviations,
and no fixed schema — so extraction has to be resilient to that.

Fields extracted:
    claim_number   e.g. "CLM-84421"
    cpt_codes      5-digit procedure codes, e.g. "99214"
    icd10_codes    diagnosis codes, e.g. "M54.5"
    dollar_amounts monetary values mentioned, e.g. "$185.00"
    dates          dates in common formats, e.g. "03/12/2026"
"""

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedClaim:
    claim_number: str | None = None
    cpt_codes: list[str] = field(default_factory=list)
    icd10_codes: list[str] = field(default_factory=list)
    dollar_amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_number": self.claim_number,
            "cpt_codes": self.cpt_codes,
            "icd10_codes": self.icd10_codes,
            "dollar_amounts": self.dollar_amounts,
            "dates": self.dates,
        }


# Patterns 

CLAIM_NUMBER_RE = re.compile(r"\bCLM-\d{4,6}\b")
CPT_CODE_RE = re.compile(r"\bCPT\s*(?:code\s*)?(\d{5})\b", re.IGNORECASE)
ICD10_CODE_RE = re.compile(r"\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b")
DOLLAR_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def extract(text: str) -> ExtractedClaim:
    """Run all extraction patterns against a denial letter's text."""
    claim_match = CLAIM_NUMBER_RE.search(text)

    return ExtractedClaim(
        claim_number=claim_match.group(0) if claim_match else None,
        cpt_codes=sorted(set(CPT_CODE_RE.findall(text))),
        icd10_codes=sorted(set(ICD10_CODE_RE.findall(text))),
        dollar_amounts=sorted(set(DOLLAR_RE.findall(text))),
        dates=sorted(set(DATE_RE.findall(text))),
    )


if __name__ == "__main__":
    sample = (
        "Claim number CLM-84421 for patient services rendered on 03/12/2026 "
        "has been denied. Reason: The requested procedure (CPT 29881) was "
        "performed without prior authorization on file, billed at $412.00 "
        "under diagnosis code M25.561."
    )
    result = extract(sample)
    print(result.to_dict())
