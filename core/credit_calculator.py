"""
core/credit_calculator.py

Credit Calculator for Art. 39 (Transferable Tax Credit - DR Film Incentives).

What this module computes (MVP scope):
1) Gross transferable tax credit:
      gross_credit_usd = 25% * total_validated_spend_usd

2) Usable credit for a transferee (the buyer/assignee of the credit):
   The law includes a limitation on the amount that can be compensated against
   Income Tax (ISR) in a given fiscal period. We model this as:

      usable_credit_usd = min(gross_credit_usd, isr_cap_usd)

   Where:
      isr_cap_usd = 25% * transferee_isr_tax_due_usd

   IMPORTANT:
   - This does NOT reduce the "credit issued" by DGCINE/DGII.
     It only models how much the transferee can apply in a given period.
   - Carry-forward rules apply for unused portions (up to 3 subsequent periods).

3) Carry-forward planning (simplified):
   The law states the transferee can use the credit in the acquisition fiscal year
   and up to three subsequent fiscal periods.
   This module produces a schedule-like breakdown for planning, assuming:
   - The transferee provides expected ISR due per year (optional).
   - If not provided, we output a warning and only provide gross credit.

MVP limitations (explicit):
- We do not model the national 10% ISR cap across all credits, because that depends
  on aggregate national tax data not available to the production.
- We do not model IT-1 / asset tax offsets in detail; we expose placeholders.

Design principles:
- Deterministic and auditable.
- No guessing.
- Requires explicit inputs (validated spend in USD; transferee ISR due in USD if needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class CreditPolicy:
    """
    Configurable policy for credit computations.

    Default (Art. 39):
    - credit_rate = 25%
    - transferee_isr_utilization_cap_rate = 25% of ISR due per fiscal period
    - carry_forward_years = 3 (usable in acquisition year + 3 subsequent years)
    """
    credit_rate: float = 0.25
    transferee_isr_utilization_cap_rate: float = 0.25
    carry_forward_years: int = 3


@dataclass(frozen=True)
class CreditComputationInputs:
    """
    Inputs required to compute the credit.

    - total_validated_spend_usd: validated spend in USD (post expense validation)
    - acquisition_fiscal_year: year in which transferee acquires the credit (or uses it first)
    - transferee_isr_due_by_year_usd: optional mapping of fiscal_year -> ISR due in USD
      This is needed if we want to compute "usable" credit per year deterministically.
    """
    total_validated_spend_usd: float
    acquisition_fiscal_year: int
    transferee_isr_due_by_year_usd: Optional[Dict[int, float]] = None


@dataclass(frozen=True)
class CreditYearPlan:
    """
    Plan for one fiscal year:
    - fiscal_year
    - isr_due_usd: transferee income tax due in that year (if provided)
    - isr_cap_usd: 25% of ISR due (cap for using this credit in that year)
    - credit_applied_usd: amount of credit planned to apply
    - remaining_credit_usd: remaining after application
    """
    fiscal_year: int
    isr_due_usd: Optional[float]
    isr_cap_usd: Optional[float]
    credit_applied_usd: float
    remaining_credit_usd: float


@dataclass(frozen=True)
class CreditResult:
    """
    Output:
    - gross_credit_usd: 25% of validated spend
    - policy: policy used
    - year_plan: optional schedule if ISR due inputs are provided
    - warnings: non-blocking warnings
    - rationale: audit-friendly rationale
    """
    gross_credit_usd: float
    policy: CreditPolicy
    year_plan: List[CreditYearPlan]
    warnings: List[str]
    rationale: List[str]


# ============================================================
# Engine
# ============================================================

class CreditCalculator:
    """
    CreditCalculator is purely computational.

    It does NOT:
    - validate expenses
    - decide eligibility
    - query DGII/DGCINE
    - infer tax amounts

    It only applies deterministic arithmetic rules once upstream validators
    have provided validated spend and (optionally) transferee ISR due data.
    """

    def __init__(self, policy: Optional[CreditPolicy] = None) -> None:
        self._policy = policy or CreditPolicy()

    def compute(self, inputs: CreditComputationInputs) -> CreditResult:
        warnings: List[str] = []
        rationale: List[str] = []

        # --------------------------------------------------------
        # 0) Basic sanity checks
        # --------------------------------------------------------
        if inputs.total_validated_spend_usd < 0:
            # Negative validated spend indicates an upstream pipeline error.
            raise ValueError("total_validated_spend_usd cannot be negative.")

        # --------------------------------------------------------
        # 1) Compute gross credit (issued credit basis)
        # --------------------------------------------------------
        gross_credit = inputs.total_validated_spend_usd * self._policy.credit_rate

        rationale.append(f"Validated spend (USD): {inputs.total_validated_spend_usd:,.2f}.")
        rationale.append(f"Credit rate (policy): {self._policy.credit_rate:.2%}.")
        rationale.append(f"Gross transferable credit (USD): {gross_credit:,.2f}.")

        # --------------------------------------------------------
        # 2) Build year-by-year utilization plan (optional)
        # --------------------------------------------------------
        year_plan: List[CreditYearPlan] = []

        isr_map = inputs.transferee_isr_due_by_year_usd
        if not isr_map:
            # MVP: if not provided, we cannot compute usable per-year amounts.
            warnings.append(
                "Transferee ISR due amounts were not provided; cannot compute per-year usable credit schedule. "
                "Returning gross credit only."
            )
            rationale.append(
                "No transferee ISR due schedule provided; year-by-year utilization plan not computed."
            )
            return CreditResult(
                gross_credit_usd=round(gross_credit, 2),
                policy=self._policy,
                year_plan=[],
                warnings=warnings,
                rationale=rationale,
            )

        # We compute plan for acquisition year + carry_forward_years subsequent years.
        start_year = inputs.acquisition_fiscal_year
        end_year = start_year + self._policy.carry_forward_years

        remaining = gross_credit
        rationale.append(
            f"Carry-forward window: acquisition year {start_year} through {end_year} "
            f"(inclusive) per policy."
        )
        rationale.append(
            f"Utilization cap per year: {self._policy.transferee_isr_utilization_cap_rate:.2%} "
            "of transferee ISR due."
        )

        for year in range(start_year, end_year + 1):
            isr_due = isr_map.get(year)

            if isr_due is None:
                # If ISR due is missing for a year, we cannot compute cap deterministically.
                warnings.append(
                    f"Missing transferee ISR due for fiscal year {year}; "
                    "cannot compute usable credit for that year."
                )
                year_plan.append(CreditYearPlan(
                    fiscal_year=year,
                    isr_due_usd=None,
                    isr_cap_usd=None,
                    credit_applied_usd=0.0,
                    remaining_credit_usd=round(remaining, 2),
                ))
                continue

            if isr_due < 0:
                raise ValueError(f"ISR due for year {year} cannot be negative.")

            isr_cap = isr_due * self._policy.transferee_isr_utilization_cap_rate

            # Apply as much as allowed by cap and remaining credit
            applied = min(remaining, isr_cap)
            remaining -= applied

            year_plan.append(CreditYearPlan(
                fiscal_year=year,
                isr_due_usd=round(isr_due, 2),
                isr_cap_usd=round(isr_cap, 2),
                credit_applied_usd=round(applied, 2),
                remaining_credit_usd=round(remaining, 2),
            ))

            rationale.append(
                f"Year {year}: ISR due={isr_due:,.2f}, cap={isr_cap:,.2f}, "
                f"applied={applied:,.2f}, remaining={remaining:,.2f}."
            )

            if remaining <= 0:
                rationale.append("All credit allocated within carry-forward window.")
                break

        if remaining > 0:
            # After the final carry-forward year, any remaining credit may expire.
            warnings.append(
                f"Remaining credit after carry-forward window (USD {remaining:,.2f}). "
                "Unused portion may not be usable after the allowed periods."
            )
            rationale.append(
                "Warning: credit remains after the modeled utilization window."
            )

        return CreditResult(
            gross_credit_usd=round(gross_credit, 2),
            policy=self._policy,
            year_plan=year_plan,
            warnings=warnings,
            rationale=rationale,
        )


# ============================================================
# Convenience function for quick calculations (optional)
# ============================================================

def calculate_gross_credit_usd(total_validated_spend_usd: float, *, credit_rate: float = 0.25) -> float:
    """
    Lightweight helper for situations where you only need the gross credit.

    We keep it as a pure function to support usage in Streamlit prototypes.
    """
    if total_validated_spend_usd < 0:
        raise ValueError("total_validated_spend_usd cannot be negative.")
    if credit_rate < 0 or credit_rate > 1:
        raise ValueError("credit_rate must be between 0 and 1.")
    return round(total_validated_spend_usd * credit_rate, 2)
