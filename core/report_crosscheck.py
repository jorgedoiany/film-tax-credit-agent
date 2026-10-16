"""
core/report_crosscheck.py

Deterministic cross-checking between:

1) The production's "NCF Bible" expense ledger
2) DGII 606 report export (purchases of goods/services)
3) DGII 609 report export (payments for services abroad)

Core Compliance Policy (Confirmed):

- ALL fiscal receipt types used in film production MUST appear in 606.
- ONLY FOREIGN_PAYMENT (B17 / E47) must also appear in 609.

Design principles:
- No assumptions.
- No probabilistic logic.
- Deterministic matching with audit trace.
- Explicit verification state.
- Strict enforcement only AFTER verification attempt.

This module does NOT block expenses.
It only computes reporting facts.
Blocking is handled later in the Expense Validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math

from .document_type_normalizer import (
    normalize_document_type,
    CanonicalDocumentType,
)


# ============================================================
# Reporting Policy (Deterministic and Locked per Requirements)
# ============================================================

@dataclass(frozen=True)
class ReportingPolicy:
    """
    Canonical reporting requirements.

    Confirmed production rule:

    - All canonical types must be reported in 606.
    - Only FOREIGN_PAYMENT must be reported in 609.
    """

    must_be_in_606: frozenset[str]
    must_be_in_609: frozenset[str]

    @staticmethod
    def default() -> "ReportingPolicy":
        return ReportingPolicy(
            must_be_in_606=frozenset({
                CanonicalDocumentType.FISCAL_CREDIT_INVOICE,
                CanonicalDocumentType.PURCHASE_FROM_NON_REGISTERED,
                CanonicalDocumentType.MINOR_EXPENSE,
                CanonicalDocumentType.SPECIAL_REGIME_NO_TAX,
                CanonicalDocumentType.FOREIGN_PAYMENT,
            }),
            must_be_in_609=frozenset({
                CanonicalDocumentType.FOREIGN_PAYMENT,
            }),
        )


# ============================================================
# Result Object
# ============================================================

@dataclass(frozen=True)
class CrossCheckResult:
    reported_in_606: bool
    reported_in_609: bool
    reporting_verified: bool
    rationale: List[str]
    amount_matches_606: Optional[bool] = None


# ============================================================
# Utility Helpers
# ============================================================

def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _amount_close(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, abs_tol=tol)


# ============================================================
# Index Builders
# ============================================================

def build_606_index(rows_606: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    606 primary deterministic key:
        (RNC, NCF)
    """
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for row in rows_606:
        rnc = _norm_str(row.get("RNC o Cédula"))
        ncf = _norm_str(row.get("NCF"))

        if not rnc or not ncf:
            continue

        key = (rnc, ncf)
        index.setdefault(key, []).append(row)

    return index


def build_609_index(rows_609: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    609 deterministic key:
        (ID Tributaria, Número de Documento)
    """
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for row in rows_609:
        foreign_id = _norm_str(row.get("ID Tributaria"))
        doc_number = _norm_str(row.get("Número de Documento"))

        if not foreign_id or not doc_number:
            continue

        key = (foreign_id, doc_number)
        index.setdefault(key, []).append(row)

    return index


# ============================================================
# Core Cross-Check Logic
# ============================================================

def cross_check_one_expense(
    ncf_row: Dict[str, Any],
    *,
    index_606: Dict[Tuple[str, str], List[Dict[str, Any]]],
    index_609: Dict[Tuple[str, str], List[Dict[str, Any]]],
    policy: ReportingPolicy,
    amount_tolerance: float = 0.01,
) -> CrossCheckResult:

    rationale: List[str] = []

    raw_doc_type = ncf_row.get("Tipo de Documento")
    canonical_doc_type = normalize_document_type(raw_doc_type)

    rationale.append(
        f"Canonical document type resolved as {canonical_doc_type} (raw={_norm_str(raw_doc_type)})."
    )

    requires_606 = canonical_doc_type in policy.must_be_in_606
    requires_609 = canonical_doc_type in policy.must_be_in_609

    # ------------------------------------------------------------
    # 606 MATCH
    # ------------------------------------------------------------

    supplier_rnc = _norm_str(ncf_row.get("RNC"))
    ncf = _norm_str(ncf_row.get("Numero NCF"))
    total_amount = _to_float(ncf_row.get("Total"))

    reported_606 = False
    attempted_606 = False
    amount_matches_606: Optional[bool] = None

    if supplier_rnc and ncf:
        attempted_606 = True
        candidates = index_606.get((supplier_rnc, ncf), [])

        if candidates:
            # Attempt amount match first
            matched = None
            for row in candidates:
                amt_606 = _to_float(row.get("Total Monto Facturado"))
                if _amount_close(total_amount, amt_606, tol=amount_tolerance):
                    matched = row
                    break

            reported_606 = True
            amount_matches_606 = matched is not None

            if amount_matches_606:
                rationale.append(
                    f"606 match found by (RNC, NCF)=({supplier_rnc}, {ncf}), "
                    f"candidates={len(candidates)}, amount matches."
                )
            else:
                amounts_606 = ", ".join(
                    str(c.get("Total Monto Facturado")) for c in candidates
                )
                rationale.append(
                    f"606 match found by (RNC, NCF)=({supplier_rnc}, {ncf}) "
                    f"but the amount differs: ledger={ncf_row.get('Total')}, "
                    f"606=[{amounts_606}]."
                )
        else:
            rationale.append(
                f"No 606 match found for (RNC, NCF)=({supplier_rnc}, {ncf})."
            )
    else:
        rationale.append(
            "Insufficient data to attempt 606 match (missing RNC and/or Numero NCF)."
        )

    # ------------------------------------------------------------
    # 609 MATCH (Only for FOREIGN_PAYMENT)
    # ------------------------------------------------------------

    foreign_id = _norm_str(ncf_row.get("Foreign ID Tributaria"))
    doc_number = _norm_str(ncf_row.get("Numero de Documento"))

    reported_609 = False
    attempted_609 = False

    if requires_609:
        if foreign_id and doc_number:
            attempted_609 = True
            candidates = index_609.get((foreign_id, doc_number), [])

            if candidates:
                reported_609 = True
                rationale.append(
                    f"609 match found by (ID Tributaria, Número de Documento)="
                    f"({foreign_id}, {doc_number}), candidates={len(candidates)}."
                )
            else:
                rationale.append(
                    f"No 609 match found for (ID Tributaria, Número de Documento)="
                    f"({foreign_id}, {doc_number})."
                )
        else:
            rationale.append(
                "Insufficient data to attempt required 609 match "
                "(missing Foreign ID Tributaria and/or Numero de Documento)."
            )
    else:
        rationale.append(
            "609 not required for this canonical document type."
        )

    # ------------------------------------------------------------
    # Verification State
    # ------------------------------------------------------------

    reporting_verified = True

    if requires_606 and not attempted_606:
        reporting_verified = False
        rationale.append(
            "Reporting NOT verified: 606 is required but could not be attempted."
        )

    if requires_609 and not attempted_609:
        reporting_verified = False
        rationale.append(
            "Reporting NOT verified: 609 is required but could not be attempted."
        )

    if requires_606:
        rationale.append("Policy: this document type must appear in 606.")

    if requires_609:
        rationale.append("Policy: this document type must appear in 609.")

    return CrossCheckResult(
        reported_in_606=reported_606,
        reported_in_609=reported_609,
        reporting_verified=reporting_verified,
        rationale=rationale,
        amount_matches_606=amount_matches_606,
    )


# ============================================================
# Bulk Cross-Check
# ============================================================

def cross_check_expenses(
    ncf_rows: List[Dict[str, Any]],
    *,
    rows_606: List[Dict[str, Any]],
    rows_609: List[Dict[str, Any]],
    policy: Optional[ReportingPolicy] = None,
) -> List[Dict[str, Any]]:

    policy = policy or ReportingPolicy.default()

    index_606 = build_606_index(rows_606)
    index_609 = build_609_index(rows_609)

    enriched_rows: List[Dict[str, Any]] = []

    # An invoice is identified by (supplier RNC, NCF). The first occurrence
    # counts; later rows with the same pair are flagged as duplicates.
    first_seen: Dict[Tuple[str, str], Tuple[int, Any]] = {}

    for position, row in enumerate(ncf_rows, start=1):
        result = cross_check_one_expense(
            row,
            index_606=index_606,
            index_609=index_609,
            policy=policy,
        )

        new_row = dict(row)

        new_row["_reported_in_606"] = result.reported_in_606
        new_row["_reported_in_609"] = result.reported_in_609
        new_row["_reporting_verified"] = result.reporting_verified
        new_row["_606_amount_matches"] = result.amount_matches_606
        new_row["_reporting_rationale"] = result.rationale

        key = (_norm_str(row.get("RNC")), _norm_str(row.get("Numero NCF")))
        new_row["_duplicate_of_row"] = None
        new_row["_duplicate_first_total"] = None

        if key[0] and key[1]:
            if key in first_seen:
                first_position, first_total = first_seen[key]
                new_row["_duplicate_of_row"] = first_position
                new_row["_duplicate_first_total"] = first_total
            else:
                first_seen[key] = (position, row.get("Total"))

        enriched_rows.append(new_row)

    return enriched_rows

    return enriched_rows
