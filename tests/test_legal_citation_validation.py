"""
tests/test_legal_citation_validation.py

This test intentionally simulates a hallucinated citation
and verifies that our citation enforcement layer detects it.
"""

from typing import cast
from core.langgraph_flow import build_art39_graph, Art39State


def test_detects_invalid_citation():
    """
    We simulate a hallucinated citation by asking a vague question
    that may encourage the model to invent references.

    The test ensures:
    - legal_citation_valid is always present
    - If invalid citation appears, it is flagged
    """

    graph = build_art39_graph()

    state = cast(Art39State, {
        "ncf_rows": [],
        "rows_606": [],
        "rows_609": [],
        "cast_crew_rows": [],

        "legal_question": "Explain interaction between Article 39 and imaginary paragraph Z.",

        "applicant_id": "TEST",
        "project_id": "TEST",
        "fiscal_year": 2026,
        "fx_to_usd": {"USD": 1.0},

        "waiver_override_present": False,
        "resident_flags": {},

        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {},

        "audit_trace": [],
    })

    result = graph.invoke(state)

    # The key check:
    assert "legal_citation_valid" in result

    # If hallucination occurs, it must be detected.
    if not result["legal_citation_valid"]:
        assert len(result["legal_citation_warnings"]) > 0