"""
Tests for the eligibility node wired into the Art. 39 flow.
"""

import json
from typing import cast

import pytest

from core.langgraph_flow import build_art39_graph, Art39State

CREW_OK = [
    {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
    {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
    {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
]

CONTEXT = {
    "approved_budget_total": 100_000_000,
    "budget_currency": "DOP",
    "permit_start": "2026-01-01",
    "permit_end": "2026-12-31",
}


def _row(ncf="B0100000001", cost=34_000_000, itbis=6_120_000,
         concept="other_general", payment="Transfer", doc="B01", **extra):
    row = {
        "Numero NCF": ncf,
        "RNC": "101234567",
        "Total": str(cost + itbis),
        "Tipo de Documento": doc,
        "date": "2026-03-10",
        "account": "100",
        "supplier": "Supplier SRL",
        "cost": cost,
        "itbis": itbis,
        "payment_method": payment,
        "concept_code": concept,
    }
    row.update(extra)
    return row


def _606(row):
    return {
        "RNC o Cédula": row["RNC"],
        "NCF": row["Numero NCF"],
        "Total Monto Facturado": row["Total"],
    }


def _state(rows, rows_606=None, rows_609=None, context=CONTEXT):
    state = {
        "cast_crew_rows": CREW_OK,
        "ncf_rows": rows,
        "rows_606": [_606(r) for r in rows] if rows_606 is None else rows_606,
        "rows_609": rows_609 or [],
        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,
        "fx_to_usd": {"DOP": 1 / 58.5, "USD": 1.0},
        "production_origin": "foreign",
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},
        "audit_trace": [],
    }
    if context is not None:
        state["project_context"] = context
    return cast(Art39State, state)


def _run(*args, **kwargs):
    return build_art39_graph().invoke(_state(*args, **kwargs))


def test_without_project_context_the_guide_rules_are_not_evaluated():
    result = _run([_row()], context=None)
    status = result["compliance_status"]

    assert status["guide_rules_evaluated"] is False
    assert any("project_context" in w for w in status["warnings"])
    assert status["eligible_to_request"] is True  # same behavior as before


def test_valid_expense_is_admitted_in_full():
    result = _run([_row()])
    status = result["compliance_status"]
    row = result["expense_validation_outputs"][0]

    assert status["guide_rules_evaluated"] is True
    assert status["eligible_to_request"] is True
    assert row["eligibility"]["status"] == "admitted"
    assert row["validated_amount"] == pytest.approx(40_120_000, abs=0.05)
    assert status["validated_spend_usd"] == pytest.approx(685_811.97, abs=0.02)
    assert status["warnings"] == []


def test_cash_payment_is_excluded_by_the_guide():
    result = _run([_row(payment="Cash")])
    status = result["compliance_status"]

    assert status["eligible_to_request"] is False
    assert status["validated_spend_usd"] == 0.0
    assert any("Cash payment" in r for r in status["excluded_expenses"][0]["reasons"])


def test_producer_fees_above_6_percent_are_partially_admitted():
    row = _row(cost=10_000_000, itbis=1_800_000, concept="producer_fees")
    result = _run([row])
    status = result["compliance_status"]
    out = result["expense_validation_outputs"][0]

    assert out["eligibility"]["status"] == "partial"
    assert out["validated_amount"] == pytest.approx(6_000_000, abs=0.05)

    partial = status["partially_admitted_expenses"]
    assert len(partial) == 1
    assert partial[0]["excluded_amount"] == pytest.approx(5_800_000, abs=0.05)
    assert status["excluded_expenses"] == []

    summary = result["eligibility_summary"]
    assert summary["executed_total"] == pytest.approx(11_800_000, abs=0.05)
    assert summary["admitted_total"] == pytest.approx(6_000_000, abs=0.05)
    assert summary["rejected_total"] == pytest.approx(5_800_000, abs=0.05)


def test_expense_missing_from_606_stays_rejected_with_its_reason():
    result = _run([_row()], rows_606=[])
    excluded = result["compliance_status"]["excluded_expenses"]

    assert len(excluded) == 1
    assert excluded[0]["row_number"] == 1
    assert "Not found in 606." in excluded[0]["reasons"]


def test_payment_abroad_with_609_is_admitted():
    row = _row(
        ncf="B1700000001", cost=40_000_000, itbis=0, doc="B17",
        **{"Foreign ID Tributaria": "F1", "Numero de Documento": "B1700000001"},
    )
    rows_609 = [{"ID Tributaria": "F1", "Número de Documento": "B1700000001"}]

    result = _run([row], rows_609=rows_609)
    out = result["expense_validation_outputs"][0]

    assert out["validatable"] is True
    assert out["eligibility"]["status"] == "admitted"
    assert round(result["threshold_result"]["total_validated_usd"], 2) == 683760.68


def test_missing_required_field_raises_with_row_and_field():
    row = _row()
    del row["concept_code"]

    with pytest.raises(ValueError, match="concept_code"):
        _run([row])


def test_eligibility_results_are_json_serializable():
    result = _run([_row(), _row(ncf="B0100000002", payment="Cash")])

    json.dumps(result["compliance_status"])
    json.dumps(result["expense_validation_outputs"])
    json.dumps(result["eligibility_summary"])
