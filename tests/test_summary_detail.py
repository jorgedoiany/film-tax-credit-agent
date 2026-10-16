"""
Tests for the compliance summary: requestable credit, blocking reasons,
excluded-expense detail, FX configuration issues and production origin.
"""

import json
from typing import cast

import pytest

from core.langgraph_flow import build_art39_graph, Art39State

# 1 Dominican out of 3 people (33%): meets the 25% minimum.
CREW_OK = [
    {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
    {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
    {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
]

# 1 Dominican out of 5 people (20%): below the 25% minimum.
CREW_LOW = [
    {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
    {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
    {"Position/Job Title": "DP", "Name": "Ann Lee", "Nationality": "British"},
    {"Position/Job Title": "Editor", "Name": "Karl Voss", "Nationality": "German"},
    {"Position/Job Title": "Sound", "Name": "Rui Costa", "Nationality": "Portuguese"},
]


def _state(total="40000000", crew=CREW_OK, fx=None, **extra):
    state = {
        "cast_crew_rows": crew,
        "ncf_rows": [{
            "Numero NCF": "B0100000001",
            "RNC": "101234567",
            "Total": total,
            "Tipo de Documento": "B01",
        }],
        "rows_606": [{
            "RNC o Cédula": "101234567",
            "NCF": "B0100000001",
            "Total Monto Facturado": total,
        }],
        "rows_609": [],
        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,
        "fx_to_usd": fx if fx is not None else {"DOP": 1 / 58.5, "USD": 1.0},
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},
        "audit_trace": [],
    }
    state.update(extra)
    return cast(Art39State, state)


def _run(**kwargs):
    return build_art39_graph().invoke(_state(**kwargs))


def test_eligible_request_exposes_requestable_credit():
    result = _run(production_origin="foreign")
    status = result["compliance_status"]

    assert status["eligible_to_request"] is True
    assert status["gross_credit_usd"] == 170940.17
    assert status["requestable_credit_usd"] == 170940.17
    assert status["blocking_reasons"] == []
    assert status["excluded_expenses"] == []


def test_below_threshold_has_zero_requestable_credit_and_reason():
    result = _run(total="10000000", production_origin="foreign")
    status = result["compliance_status"]

    assert status["eligible_to_request"] is False
    assert status["gross_credit_usd"] > 0          # informational
    assert status["requestable_credit_usd"] == 0.0
    assert len(status["blocking_reasons"]) == 1
    assert "below" in status["blocking_reasons"][0]
    assert result["threshold_result"]["shortfall_usd"] > 0


def test_foreign_production_below_participation_is_blocked():
    result = _run(crew=CREW_LOW, production_origin="foreign")
    status = result["compliance_status"]

    assert status["threshold_met"] is True
    assert status["participation_applicable"] is True
    assert status["participation_met"] is False
    assert status["eligible_to_request"] is False
    assert status["requestable_credit_usd"] == 0.0
    assert any("participation" in r for r in status["blocking_reasons"])


def test_dominican_production_is_not_blocked_by_participation():
    result = _run(crew=CREW_LOW, production_origin="dominican")
    status = result["compliance_status"]

    assert status["participation_applicable"] is False
    assert status["participation_met"] is True
    assert status["eligible_to_request"] is True
    assert status["requestable_credit_usd"] == status["gross_credit_usd"]
    assert status["warnings"] == []


def test_missing_origin_assumes_foreign_and_warns():
    result = _run(crew=CREW_LOW)
    status = result["compliance_status"]

    assert status["participation_applicable"] is True
    assert status["eligible_to_request"] is False
    assert any("production_origin" in w for w in status["warnings"])


def test_invalid_origin_raises():
    with pytest.raises(ValueError):
        _run(production_origin="martian")


def test_missing_fx_rate_is_reported_as_configuration_issue():
    result = _run(fx={"USD": 1.0}, production_origin="foreign")
    status = result["compliance_status"]

    assert status["eligible_to_request"] is False
    assert len(status["fx_excluded_expenses"]) == 1
    assert "missing FX rate" in status["fx_excluded_expenses"][0]["reason"]
    assert any("FX rate" in r for r in status["blocking_reasons"])


def test_excluded_expense_detail_lists_reasons():
    state = _state(production_origin="foreign")
    state["rows_606"] = []  # the invoice is not reported in the 606

    result = build_art39_graph().invoke(state)
    excluded = result["compliance_status"]["excluded_expenses"]

    assert len(excluded) == 1
    assert excluded[0]["expense_id"] == "B0100000001"
    assert "Not found in 606." in excluded[0]["reasons"]


def test_detailed_results_are_json_serializable():
    result = _run(production_origin="foreign")

    for key in ("threshold_result", "participation_result",
                "credit_result", "compliance_status"):
        json.dumps(result[key])

    plan = result["credit_result"]["year_plan"]
    assert isinstance(plan, list) and isinstance(plan[0], dict)
    assert plan[0]["fiscal_year"] == 2026
    assert plan[0]["credit_applied_usd"] == 125000.0
