"""
core/threshold_validator.py

Threshold Engine for Art. 39 (Transferable Tax Credit - DR Film Incentives).

Art. 39 requires (as a general rule) that the spend executed at the time of the
request is >= USD 500,000 to request the transferable tax credit.

Operational nuance reflected in the law:
- A production can request credit for partial executions once the threshold is met.
- Multiple projects may be accumulated to reach the minimum threshold, as long as:
  - They belong to the same applicant entity (person or company)
  - They are executed within the same fiscal period (year), per the law's wording

This module is deterministic:
- It does not interpret legal text via LLM.
- It only aggregates numbers and applies explicit rules.
- It returns an auditable rationale (trace) to support compliance reporting.

IMPORTANT DESIGN DECISION:
- The threshold must be computed on VALIDATED spend, not raw ledger totals.
  Validated spend is produced by the Expense Validation Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class ValidatedExpenseRow:
    """
    A small, stable structure representing the minimum data needed for threshold checks.

    This is intentionally separate from your full Expense model to avoid tight coupling
    and keep threshold logic simple and testable.

    Fields:
    - expense_id: traceability
    - applicant_id: who claims the incentive (RNC or internal ID)
    - project_id: project identifier (one film/series)
    - fiscal_year: year used for the "same fiscal period" constraint
    - validated_amount_local: validated amount in local currency (DOP) OR original currency
    - currency: currency code of the validated amount (e.g., "DOP", "USD")
    - validated: must be True to count
    """
    expense_id: str
    applicant_id: str
    project_id: str
    fiscal_year: int
    validated_amount: float
    currency: str
    validated: bool


@dataclass(frozen=True)
class FXPolicy:
    """
    FX conversion policy.

    Why needed:
    - Art. 39 threshold is expressed in USD.
    - Production ledgers may be in DOP, USD, or mixed.
    - We must be deterministic: conversion rates must be explicit inputs.

    Strategy:
    - Provide a mapping of currency -> USD conversion rate.
      Example: {"DOP": 1/58.5, "USD": 1.0}
    - If a currency is missing, we do NOT guess; we mark threshold as not computable.
    """
    to_usd_rate: Dict[str, float]

    def convert_to_usd(self, amount: float, currency: str) -> Optional[float]:
        code = str(currency).strip().upper()
        if code not in self.to_usd_rate:
            return None
        return amount * self.to_usd_rate[code]


@dataclass(frozen=True)
class ThresholdPolicy:
    """
    Policy configuration for Art. 39 threshold.

    Default threshold: USD 500,000.

    We keep it configurable in case legislation changes or the product is reused.
    """
    minimum_usd: float = 500_000.00


@dataclass(frozen=True)
class ThresholdResult:
    """
    Output of the threshold evaluation.

    - eligible_to_request: whether the applicant meets the minimum executed spend
    - total_validated_usd: total validated spend in USD counted by the engine
    - counted_expenses: expense IDs included in the calculation
    - excluded_expenses: expense IDs excluded (and why)
    - rationale: audit-friendly explanation
    """
    eligible_to_request: bool
    total_validated_usd: float
    counted_expenses: List[str]
    excluded_expenses: List[Dict[str, Any]]
    rationale: List[str]


# ============================================================
# Engine
# ============================================================

class ThresholdEngine:
    """
    ThresholdEngine computes whether an applicant meets the USD minimum executed spend.

    The engine supports two evaluation modes:

    1) Single Project Mode:
       - Compute threshold for one project_id.

    2) Accumulation Mode (same fiscal year):
       - Compute threshold across multiple projects for the same applicant_id
         within the same fiscal_year.

    IMPORTANT:
    - This engine does not decide *whether accumulation is legally acceptable*
      beyond the explicit constraints we model:
        - same applicant_id
        - same fiscal_year
    - If you want additional constraints (e.g., "same legal entity classification"),
      add them upstream when building ValidatedExpenseRow.
    """

    def __init__(
        self,
        *,
        fx_policy: FXPolicy,
        threshold_policy: Optional[ThresholdPolicy] = None,
    ) -> None:
        self._fx = fx_policy
        self._policy = threshold_policy or ThresholdPolicy()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def evaluate_single_project(
        self,
        rows: List[ValidatedExpenseRow],
        *,
        applicant_id: str,
        project_id: str,
        fiscal_year: int,
    ) -> ThresholdResult:
        """
        Evaluate threshold for a specific applicant_id + project_id + fiscal_year.
        """
        scoped = [
            r for r in rows
            if r.applicant_id == applicant_id
            and r.project_id == project_id
            and r.fiscal_year == fiscal_year
        ]
        return self._evaluate(scoped, mode="single_project", applicant_id=applicant_id, fiscal_year=fiscal_year)

    def evaluate_accumulated(
        self,
        rows: List[ValidatedExpenseRow],
        *,
        applicant_id: str,
        fiscal_year: int,
        project_ids: Optional[List[str]] = None,
    ) -> ThresholdResult:
        """
        Evaluate threshold by accumulating spend across multiple projects.

        If project_ids is provided, accumulation is limited to those projects.
        If not provided, we include all projects for that applicant in the fiscal year.
        """
        scoped = [
            r for r in rows
            if r.applicant_id == applicant_id
            and r.fiscal_year == fiscal_year
            and (project_ids is None or r.project_id in set(project_ids))
        ]
        return self._evaluate(scoped, mode="accumulated", applicant_id=applicant_id, fiscal_year=fiscal_year)

    # --------------------------------------------------------
    # Internal logic
    # --------------------------------------------------------

    def _evaluate(
        self,
        scoped_rows: List[ValidatedExpenseRow],
        *,
        mode: str,
        applicant_id: str,
        fiscal_year: int,
    ) -> ThresholdResult:
        """
        Core deterministic aggregation routine.

        Rules:
        - Only rows with validated=True count toward threshold.
        - Only amounts convertible to USD count.
        - If any validated row cannot be converted (missing FX rate), we exclude it and
          record an exclusion. This is intentional: we do not guess FX rates.
        """
        rationale: List[str] = []
        counted_expenses: List[str] = []
        excluded_expenses: List[Dict[str, Any]] = []

        total_usd = 0.0

        if mode == "single_project":
            rationale.append(f"Threshold mode: single_project (applicant={applicant_id}, fiscal_year={fiscal_year}).")
        else:
            rationale.append(f"Threshold mode: accumulated (applicant={applicant_id}, fiscal_year={fiscal_year}).")

        if not scoped_rows:
            rationale.append("No expense rows found for the requested scope.")
            return ThresholdResult(
                eligible_to_request=False,
                total_validated_usd=0.0,
                counted_expenses=[],
                excluded_expenses=[],
                rationale=rationale,
            )

        for row in scoped_rows:
            # Skip anything not validated; threshold must be computed on validated spend.
            if not row.validated:
                excluded_expenses.append({
                    "expense_id": row.expense_id,
                    "reason": "Excluded: not validated by Expense Validation Engine.",
                    "project_id": row.project_id,
                })
                continue

            usd = self._fx.convert_to_usd(row.validated_amount, row.currency)
            if usd is None:
                excluded_expenses.append({
                    "expense_id": row.expense_id,
                    "reason": f"Excluded: missing FX rate for currency={row.currency}.",
                    "project_id": row.project_id,
                })
                continue

            total_usd += usd
            counted_expenses.append(row.expense_id)

        rationale.append(f"Counted validated expenses: {len(counted_expenses)}.")
        rationale.append(f"Excluded expenses: {len(excluded_expenses)}.")
        rationale.append(f"Total validated spend (USD): {total_usd:,.2f}.")
        rationale.append(f"Minimum threshold (USD): {self._policy.minimum_usd:,.2f}.")

        eligible = total_usd >= self._policy.minimum_usd
        if eligible:
            rationale.append("Result: threshold met. Applicant is eligible to request under Art. 39 (threshold condition).")
        else:
            remaining = self._policy.minimum_usd - total_usd
            rationale.append(f"Result: threshold NOT met. Missing USD {remaining:,.2f} to reach minimum.")

        return ThresholdResult(
            eligible_to_request=eligible,
            total_validated_usd=round(total_usd, 2),
            counted_expenses=counted_expenses,
            excluded_expenses=excluded_expenses,
            rationale=rationale,
        )


# ============================================================
# Convenience builder (optional but practical)
# ============================================================

def build_validated_rows_from_validator_output(
    validator_outputs: List[Dict[str, Any]],
    *,
    applicant_id_field: str = "applicant_id",
    project_id_field: str = "project_id",
    fiscal_year_field: str = "fiscal_year",
    expense_id_field: str = "expense_id",
    validated_field: str = "validatable",
    validated_amount_field: str = "validated_amount",
    currency_field: str = "currency",
) -> List[ValidatedExpenseRow]:
    """
    Convert a list of dict outputs into ValidatedExpenseRow objects.

    Why this exists:
    - In a Streamlit MVP, you may store intermediate results as dicts in session state.
    - This helper provides a deterministic mapping layer.

    Example expected dict keys:
      {
        "expense_id": "EXP-001",
        "applicant_id": "101234567",
        "project_id": "MOVIE-2026-01",
        "fiscal_year": 2026,
        "validatable": True,
        "validated_amount": 12000.50,
        "currency": "DOP"
      }

    If a required field is missing, we raise a ValueError.
    This is intentional: missing keys indicate upstream pipeline issues.
    """
    rows: List[ValidatedExpenseRow] = []

    required = [
        applicant_id_field,
        project_id_field,
        fiscal_year_field,
        expense_id_field,
        validated_field,
        validated_amount_field,
        currency_field,
    ]

    for item in validator_outputs:
        for key in required:
            if key not in item:
                raise ValueError(f"Missing required field '{key}' in validator output item: {item}")

        rows.append(ValidatedExpenseRow(
            expense_id=str(item[expense_id_field]),
            applicant_id=str(item[applicant_id_field]),
            project_id=str(item[project_id_field]),
            fiscal_year=int(item[fiscal_year_field]),
            validated_amount=float(item[validated_amount_field]),
            currency=str(item[currency_field]).strip().upper(),
            validated=bool(item[validated_field]),
        ))

    return rows
