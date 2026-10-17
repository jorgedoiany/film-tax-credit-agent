from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Protocol, Tuple


class FXConverter(Protocol):
    """Compatible with FXPolicy from core/threshold_validator.py."""

    def convert_to_usd(self, amount: float, currency: str) -> Optional[float]: ...


@dataclass(frozen=True)
class ProjectContext:
    project_id: str
    applicant_id: str
    fiscal_year: int
    approved_budget_total: float   # base for the 6% and 3% caps (documented parameter)
    budget_currency: str
    permit_start: date             # Single Filming Permit validity (Art. 188 b)
    permit_end: date


@dataclass(frozen=True)
class ExpenseRecord:
    """An expense as delivered by the budgeting app. Data only, no verdict."""
    expense_id: str
    project_id: str
    applicant_id: str
    fiscal_year: int
    date: date
    account: str
    ncf: Optional[str]
    supplier: str
    rnc: str
    details: str                   # free text ("Observaciones"); the engine does NOT interpret it
    cost: float                    # cost, excluding ITBIS
    itbis: float
    total: float
    payment_method: Optional[str]  # "Transfer" | "Check" | "Cash" | None
    concept_code: str              # picked from a closed catalog (dropdown)
    currency: str
    supplier_sirecine: bool        # Art. 188 (a), simulated


@dataclass(frozen=True)
class Rejection:
    rule: str
    message: str
    legal_ref: str
    amount: float                  # amount (cost plus ITBIS) rejected by this rule (expense currency)


@dataclass(frozen=True)
class ExpenseDecision:
    expense_id: str
    status: str                    # "admitted" | "partial" | "rejected"
    cost: float                    # amount of the expense: cost plus ITBIS  
    admitted_cost: float           # admitted part of that amount (cost plus ITBIS)
    rejections: Tuple[Rejection, ...]

    @property
    def primary_rule(self) -> Optional[str]:
        return self.rejections[0].rule if self.rejections else None


@dataclass(frozen=True)
class AccountSummary:
    account: str
    executed: float                # executed budget (budget currency)
    admitted: float                # admissible execution for the incentive


@dataclass(frozen=True)
class CapInfo:
    cap: float
    claimed: float
    excess: float


@dataclass(frozen=True)
class EligibilityReport:
    project_id: str
    currency: str
    decisions: Tuple[ExpenseDecision, ...]
    by_account: Tuple[AccountSummary, ...]
    executed_total: float
    admitted_total: float
    rejected_total: float
    rejected_by_primary_rule: Dict[str, float]
    caps: Dict[str, CapInfo]
    unconverted_expense_ids: Tuple[str, ...]