"""
Golden Tests for Art. 39 LangGraph Compliance Pipeline.

These tests validate full end-to-end behavior.
If any of these fail after refactoring,
the regulatory logic has changed.
"""

from typing import cast
from core.langgraph_flow import build_art39_graph, Art39State


def _base_state():
    """
    Returns a reusable base state with compliant participation
    and FX configuration.
    """
    return {
        "cast_crew_rows": [
            {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
            {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
            {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
        ],
        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,
        "fx_to_usd": {"DOP": 1 / 58.5, "USD": 1.0},
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},
        "audit_trace": [],
    }


# ============================================================
# GOLDEN TEST 1 — Fully Compliant Case
# ============================================================

def test_full_compliance_case():
    state = _base_state()

    state.update({
        "ncf_rows": [{
            "Numero NCF": "B0100000001",
            "RNC": "101234567",
            "Total": "40000000",  # 40M DOP
            "Tipo de Documento": "B01",
        }],
        "rows_606": [{
            "RNC o Cédula": "101234567",
            "NCF": "B0100000001",
            "Total Monto Facturado": "40000000",
        }],
        "rows_609": [],
    })

    graph = build_art39_graph()
    result = graph.invoke(cast(Art39State, state))

    assert result["compliance_status"]["eligible_to_request"] is True
    assert round(result["threshold_result"]["total_validated_usd"], 2) == 683760.68
    assert round(result["credit_result"]["gross_credit_usd"], 2) == 170940.17


# ============================================================
# GOLDEN TEST 2 — Threshold Failure
# ============================================================

def test_threshold_failure():
    state = _base_state()

    state.update({
        "ncf_rows": [{
            "Numero NCF": "B0100000002",
            "RNC": "101234567",
            "Total": "10000000",  # 10M DOP
            "Tipo de Documento": "B01",
        }],
        "rows_606": [{
            "RNC o Cédula": "101234567",
            "NCF": "B0100000002",
            "Total Monto Facturado": "10000000",
        }],
        "rows_609": [],
    })

    graph = build_art39_graph()
    result = graph.invoke(cast(Art39State, state))

    assert result["compliance_status"]["eligible_to_request"] is False
    assert result["threshold_result"]["eligible_to_request"] is False


# ============================================================
# GOLDEN TEST 3 — B17 Missing 609 Enforcement
# ============================================================

def test_b17_missing_609_blocks_validation():
    state = _base_state()

    state.update({
        "ncf_rows": [{
            "Numero NCF": "B1700000001",
            "RNC": "101234567",
            "Total": "40000000",
            "Tipo de Documento": "B17",
            "Foreign ID Tributaria": "FOREIGN-ID",
            "Numero de Documento": "B1700000001",
        }],
        "rows_606": [{
            "RNC o Cédula": "101234567",
            "NCF": "B1700000001",
            "Total Monto Facturado": "40000000",
        }],
        "rows_609": [],  # Missing 609 on purpose
    })

    graph = build_art39_graph()
    result = graph.invoke(cast(Art39State, state))

    assert result["threshold_result"]["total_validated_usd"] == 0.0
    assert result["compliance_status"]["eligible_to_request"] is False