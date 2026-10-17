"""
Tests for the adaptations made to the ported eligibility engine:
payments abroad (B17/E47), upstream rejections and skipping NCF rules.
"""

from datetime import date

import pytest

from core.eligibility.engine import evaluate_project
from core.eligibility.models import ExpenseRecord, ProjectContext, Rejection
from core.threshold_validator import FXPolicy

FX = FXPolicy(to_usd_rate={"DOP": 1 / 58.5, "USD": 1.0})

PROJECT = ProjectContext(
    project_id="MOVIE-001",
    applicant_id="101234567",
    fiscal_year=2026,
    approved_budget_total=100_000_000,
    budget_currency="DOP",
    permit_start=date(2026, 1, 1),
    permit_end=date(2026, 12, 31),
)


def _exp(expense_id, concept="other_general", cost=1_000_000, ncf="B0100000001",
         rnc="101000001", payment="Transfer", itbis=None):
    itbis = round(cost * 0.18, 2) if itbis is None else itbis
    return ExpenseRecord(
        expense_id=expense_id,
        project_id="MOVIE-001",
        applicant_id="101234567",
        fiscal_year=2026,
        date=date(2026, 3, 10),
        account="100",
        ncf=ncf,
        supplier="Supplier",
        rnc=rnc,
        details="",
        cost=cost,
        itbis=itbis,
        total=cost + itbis,
        payment_method=payment,
        concept_code=concept,
        currency="DOP",
        supplier_sirecine=False,
    )


def _decisions(expenses, **kwargs):
    report = evaluate_project(expenses, PROJECT, FX, **kwargs)
    return report, {d.expense_id: d for d in report.decisions}


def test_payment_abroad_skips_the_itbis_matrix():
    _, d = _decisions([_exp("F1", ncf="B1700000001", itbis=0.0)])

    assert d["F1"].status == "admitted"
    assert d["F1"].admitted_cost == pytest.approx(1_000_000, abs=0.05)


def test_payment_abroad_is_still_rejected_when_paid_in_cash():
    _, d = _decisions([_exp("F2", ncf="E4700000001", itbis=0.0, payment="Cash")])

    assert d["F2"].status == "rejected"
    assert d["F2"].primary_rule == "PAYMENT_CASH"


def test_unknown_receipt_type_is_still_unsupported():
    _, d = _decisions([_exp("U1", ncf="B1500000001")])

    assert d["U1"].status == "rejected"
    assert d["U1"].primary_rule == "NCF_TYPE_UNSUPPORTED"


def test_prerejected_expense_is_rejected_and_not_counted_for_caps():
    a = _exp("A", concept="producer_fees", cost=4_000_000, ncf="B0100000001", rnc="101000001")
    b = _exp("B", concept="producer_fees", cost=4_000_000, ncf="B0100000002", rnc="101000002")
    upstream = {
        "B": (Rejection("REPORTING_CROSSCHECK", "Not found in 606.",
                        "Guide JRD-G43, 8.8", 4_720_000.0),),
    }

    # Without the upstream rejection both count: 9.44M claimed vs a 6M cap.
    _, plain = _decisions([a, b])
    assert plain["A"].status == "partial"

    # With it, only A counts: 4.72M claimed, under the cap, so A is fully admitted.
    report, d = _decisions([a, b], prerejected=upstream, skip_ncf_rules=True)

    assert d["B"].status == "rejected"
    assert d["B"].admitted_cost == 0
    assert d["B"].rejections[0].rule == "REPORTING_CROSSCHECK"
    assert d["A"].status == "admitted"
    assert report.caps["producer_fees"].claimed == pytest.approx(4_720_000, abs=0.05)
    assert report.caps["producer_fees"].excess == 0


def test_duplicate_ncf_is_checked_by_default_and_skipped_on_request():
    first = _exp("D1", ncf="B0100000050", rnc="101000050")
    second = _exp("D2", ncf="B0100000050", rnc="101000050")

    _, default = _decisions([first, second])
    assert default["D1"].status == "admitted"
    assert default["D2"].primary_rule == "NCF_DUPLICATE"

    _, skipped = _decisions([first, second], skip_ncf_rules=True)
    assert skipped["D1"].status == "admitted"
    assert skipped["D2"].status == "admitted"


def test_missing_ncf_is_checked_by_default_and_skipped_on_request():
    expense = _exp("N1", ncf=None)

    _, default = _decisions([expense])
    assert default["N1"].primary_rule == "NCF_MISSING"

    _, skipped = _decisions([expense], skip_ncf_rules=True)
    assert skipped["N1"].status == "admitted"
