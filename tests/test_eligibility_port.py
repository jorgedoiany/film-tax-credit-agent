"""
Smoke tests for the eligibility engine ported from the hackathon repo.
Confirms it imports and evaluates inside the Tax Agent before any adaptation.
"""

from datetime import date

import pytest

from core.eligibility.engine import evaluate_project
from core.eligibility.models import ExpenseRecord, ProjectContext
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


def _expense(expense_id, concept, cost, payment="Transfer", day=10):
    itbis = round(cost * 0.18, 2)
    return ExpenseRecord(
        expense_id=expense_id,
        project_id="MOVIE-001",
        applicant_id="101234567",
        fiscal_year=2026,
        date=date(2026, 3, day),
        account="100",
        ncf=f"B01000000{day:02d}",
        supplier="Supplier SRL",
        rnc=f"1010000{day:02d}",
        details="",
        cost=cost,
        itbis=itbis,
        total=cost + itbis,
        payment_method=payment,
        concept_code=concept,
        currency="DOP",
        supplier_sirecine=False,
    )


def _decisions():
    expenses = [
        _expense("E1", "other_general", 1_000_000, day=1),
        _expense("E2", "other_general", 1_000_000, payment="Cash", day=2),
        _expense("E3", "producer_fees", 10_000_000, day=3),
    ]
    report = evaluate_project(expenses, PROJECT, FX)
    return report, {d.expense_id: d for d in report.decisions}


def test_valid_expense_is_admitted_in_full():
    _, d = _decisions()
    assert d["E1"].status == "admitted"
    assert d["E1"].admitted_cost == pytest.approx(1_180_000, abs=0.05)


def test_cash_payment_is_rejected():
    _, d = _decisions()
    assert d["E2"].status == "rejected"
    assert d["E2"].admitted_cost == 0
    assert d["E2"].primary_rule == "PAYMENT_CASH"


def test_producer_fees_above_6_percent_are_partially_admitted():
    report, d = _decisions()
    # 6% of the 100,000,000 budget = 6,000,000 admitted out of 11,800,000
    assert d["E3"].status == "partial"
    assert d["E3"].admitted_cost == pytest.approx(6_000_000, abs=0.05)
    assert report.caps["producer_fees"].excess > 0
