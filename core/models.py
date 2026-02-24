"""
core/models.py

This module defines small, stable domain models (dataclasses) that are shared
across the project. We keep them simple and explicit so the compliance logic
remains deterministic and easy to test.

Key design choice:
- We represent "evidence" (NCF validity, 606/609 reporting, withholding) as flags
  that come from upstream data pipelines (e.g., CSV uploads, DGII checks, or
  manual user input in Streamlit). The validator does NOT "guess" these facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class ProductionStage(str, Enum):
    DEVELOPMENT = "development"
    PRE_PRODUCTION = "pre_production"
    PRODUCTION = "production"
    POST_PRODUCTION = "post_production"


class ExpenseType(str, Enum):
    GOODS = "goods"
    SERVICES = "services"
    PAYROLL = "payroll"
    RENTAL = "rental"
    TRAVEL = "travel"
    FEES = "fees"
    FINANCING = "financing"
    MARKETING = "marketing"
    DISTRIBUTION = "distribution"
    OTHER = "other"


@dataclass(frozen=True)
class EvidenceFlags:
    """
    EvidenceFlags is a compact representation of whether the expense has the
    *minimum* documentation signals needed for compliance.

    Why flags and not raw PDFs?
    - For an MVP, we want deterministic logic and fast iteration.
    - The UI (Streamlit) can ask for these flags explicitly and attach files,
      but the compliance engine should only rely on structured facts.

    These flags can later be upgraded to be computed by:
    - DGII NCF lookup tool (NCF validity)
    - Parsing 606/609 exports
    - IR-17 / IT-1 reported withholding consistency checks
    """
    has_invoice: bool = False
    ncf_present: bool = False
    ncf_valid_in_dgii: bool = False
    reported_in_606: bool = False
    reported_in_609: bool = False
    payment_proof_present: bool = False

    # Cheque-specific requirements
    cheque_has_authorized_signature: bool = True
    cheque_has_supplier_acknowledgement: bool = True

    # Readability requirement (illegible invoices are effectively non-verifiable)
    invoice_is_legible: bool = True


@dataclass(frozen=True)
class WithholdingInfo:
    """
    WithholdingInfo represents what the expense claims to have withheld.

    Why do we model withholding explicitly?
    - The guide indicates expenses can be excluded if "required withholdings"
      were not applied, depending on service type and contributor type.
    - Determining the "required" withholding rate can be complex; in the MVP
      we compute it via policy rules (configurable), not LLM reasoning.

    Note:
    - These are amounts, not rates, because real documents show amounts.
    - If you only have rates, convert them to amounts upstream.
    """
    isr_withheld_amount: float = 0.0
    itbis_withheld_amount: float = 0.0


@dataclass(frozen=True)
class Expense:
    """
    The canonical expense record used by the compliance engine.

    Minimal fields to support Art. 39 validation:
    - category/type: for exclusion rules (marketing, financing, etc.)
    - stage: required because reports are organized by stages (pre/pro/post)
    - budget_line: helpful for traceability to cost report templates
    - amounts: gross, subtotal, itbis (if available)
    - evidence: documentation flags
    - withholding: amounts withheld (if any)

    We intentionally keep this structure stable. Any additional raw fields from
    a CSV/Excel upload can be kept in `raw` for traceability without affecting
    deterministic validation logic.
    """
    expense_id: str
    description: str
    stage: ProductionStage
    expense_type: ExpenseType

    gross_amount: float

    # If invoices separate base and ITBIS, capture them (optional).
    subtotal_amount: Optional[float] = None
    itbis_amount: Optional[float] = None

    supplier_name: Optional[str] = None
    supplier_rnc: Optional[str] = None
    ncf: Optional[str] = None

    evidence: EvidenceFlags = field(default_factory=EvidenceFlags)
    withholding: WithholdingInfo = field(default_factory=WithholdingInfo)

    # "Directly related to production" is a legal standard; we capture it as an
    # explicit input decision (human or model-assisted classification).
    directly_related: bool = True

    # For caps like "producer fee <= 6% of total budget", we need tags.
    is_producer_fee: bool = False
    is_development_cost: bool = False

    # Keep raw original record for audit trail (never used for logic directly).
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    """
    A single issue found during validation.

    severity:
      - "BLOCKER": expense should be treated as non-validatable for now
      - "WARNING": potentially fixable / requires attention, but may not block
    """
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """
    Result for a single expense.

    validatable:
      - True means the expense is acceptable for inclusion in "validated spend"
        for threshold and credit estimation.
      - False means the expense should be excluded (or marked pending fixes).

    rationale:
      - short, explicit reasons suitable for legal/compliance audit.
    """
    expense_id: str
    validatable: bool
    validated_amount: float
    issues: List[ValidationIssue] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
