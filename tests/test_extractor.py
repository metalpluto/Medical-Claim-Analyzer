"""
tests/test_extractor.py

Unit tests for the rule-based NLP extractor. extract() is a pure
function with no randomness and no external dependencies, so every
test here is fully deterministic and requires no API key or network
access.
"""

from src.extractor import extract


def test_extracts_claim_number():
    text = "Claim number CLM-84421 for patient services has been denied."
    result = extract(text)
    assert result.claim_number == "CLM-84421"


def test_missing_claim_number_returns_none():
    text = "This letter mentions no claim identifier at all."
    result = extract(text)
    assert result.claim_number is None


def test_extracts_single_cpt_code():
    text = "The requested procedure (CPT 29881) was performed without authorization."
    result = extract(text)
    assert result.cpt_codes == ["29881"]


def test_extracts_multiple_unique_cpt_codes_sorted():
    text = "Procedures CPT 99214 and CPT code 72148 were both billed, then CPT 99214 again."
    result = extract(text)
    # duplicates collapsed, result sorted
    assert result.cpt_codes == ["72148", "99214"]


def test_extracts_icd10_code():
    text = "Denied under diagnosis code M54.5 for the visit."
    result = extract(text)
    assert "M54.5" in result.icd10_codes


def test_extracts_dollar_amount():
    text = "The claim was billed at $412.00 and denied in full."
    result = extract(text)
    assert "$412.00" in result.dollar_amounts


def test_extracts_dollar_amount_with_thousands_separator():
    text = "Total billed charges were $1,250.75 for the procedure."
    result = extract(text)
    assert "$1,250.75" in result.dollar_amounts


def test_extracts_date():
    text = "Services were rendered on 03/12/2026 and denied shortly after."
    result = extract(text)
    assert "03/12/2026" in result.dates


def test_extracts_all_fields_from_realistic_letter():
    text = (
        "Claim number CLM-84421 for patient services rendered on 03/12/2026 "
        "has been denied. Reason: The requested procedure (CPT 29881) was "
        "performed without prior authorization on file, billed at $412.00 "
        "under diagnosis code M25.561."
    )
    result = extract(text)
    assert result.claim_number == "CLM-84421"
    assert result.cpt_codes == ["29881"]
    assert "M25.561" in result.icd10_codes
    assert result.dollar_amounts == ["$412.00"]
    assert result.dates == ["03/12/2026"]


def test_extract_on_empty_string_returns_empty_result():
    result = extract("")
    assert result.claim_number is None
    assert result.cpt_codes == []
    assert result.icd10_codes == []
    assert result.dollar_amounts == []
    assert result.dates == []


def test_to_dict_matches_fields():
    text = "Claim number CLM-10001 billed at $50.00 on 01/01/2026."
    result = extract(text)
    d = result.to_dict()
    assert d["claim_number"] == "CLM-10001"
    assert d["dollar_amounts"] == ["$50.00"]
    assert d["dates"] == ["01/01/2026"]
    assert set(d.keys()) == {
        "claim_number", "cpt_codes", "icd10_codes", "dollar_amounts", "dates"
    }
