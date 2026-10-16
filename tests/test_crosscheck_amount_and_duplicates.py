"""
Tests for 606 amount matching and duplicate invoice detection.

An invoice is identified by (supplier RNC, NCF). Amounts in the ledger must
match the 606 report, and repeated invoices must not be counted twice.
"""

from typing import cast

from core.langgraph_flow import build_art39_graph, Art39State


def _state(ncf_rows, rows_606):
    return cast(Art39State, {
        "cast_crew_rows": [
            {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
            {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
            {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
        ],
        "ncf_rows": ncf_rows,
        "rows_606": rows_606,
        "rows_609": [],
        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,
        "fx_to_usd": {"DOP": 1 / 58.5, "USD": 1.0},
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},
        "audit_trace": [],
    })


def _ledger(ncf, total, rnc="101234567"):
    return {
        "Numero NCF": ncf,
        "RNC": rnc,
        "Total": total,
        "Tipo de Documento": "B01",
    }


def _606(ncf, total, rnc="101234567"):
    return {"RNC o Cédula": rnc, "NCF": ncf, "Total Monto Facturado": total}


def _run(ncf_rows, rows_606):
    return build_art39_graph().invoke(_state(ncf_rows, rows_606))


def test_amount_differs_from_606_is_excluded():
    result = _run(
        [_ledger("B0100000010", "40000000")],
        [_606("B0100000010", "10000000")],
    )

    assert result["threshold_result"]["total_validated_usd"] == 0.0
    assert result["compliance_status"]["eligible_to_request"] is False

    row = result["expense_validation_outputs"][0]
    assert row["validatable"] is False
    assert any("differs" in reason for reason in row["exclusion_reasons"])


def test_amount_within_tolerance_still_counts():
    result = _run(
        [_ledger("B0100000011", "40000000.00")],
        [_606("B0100000011", "40000000.01")],
    )

    assert result["expense_validation_outputs"][0]["validatable"] is True
    assert round(result["threshold_result"]["total_validated_usd"], 2) == 683760.68


def test_duplicate_invoice_is_counted_once():
    result = _run(
        [_ledger("B0100000012", "40000000"), _ledger("B0100000012", "40000000")],
        [_606("B0100000012", "40000000")],
    )

    first, second = result["expense_validation_outputs"]
    assert first["validatable"] is True
    assert second["validatable"] is False
    assert any("Duplicate of row 1" in reason for reason in second["exclusion_reasons"])

    # Counted once (40M DOP), not twice (80M DOP).
    assert round(result["threshold_result"]["total_validated_usd"], 2) == 683760.68


def test_same_ncf_from_different_suppliers_both_count():
    result = _run(
        [
            _ledger("B0100000013", "20000000", rnc="101111111"),
            _ledger("B0100000013", "20000000", rnc="102222222"),
        ],
        [
            _606("B0100000013", "20000000", rnc="101111111"),
            _606("B0100000013", "20000000", rnc="102222222"),
        ],
    )

    assert all(r["validatable"] for r in result["expense_validation_outputs"])
    assert round(result["threshold_result"]["total_validated_usd"], 2) == 683760.68
