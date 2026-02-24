"""
core/document_type_normalizer.py

This module normalizes Dominican fiscal receipt document types (NCF/e-CF) into
canonical categories used by the compliance engine.

Why normalization matters:
- The same business meaning exists in both physical and electronic receipts.
- We want a single policy table regardless of whether a supplier uses NCF or e-CF.
- This makes the system future-proof as the market migrates to e-CF adoption.

The validator should never branch on raw codes like "B01" vs "E31".
It should branch on canonical categories like "FISCAL_CREDIT_INVOICE".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass(frozen=True)
class CanonicalDocumentType:
    """
    Canonical categories used by the engine.

    We use strings (instead of Enum) to keep interoperability easy when
    serializing to JSON (LangGraph state, Streamlit sessions, logs).
    """
    FISCAL_CREDIT_INVOICE: str = "FISCAL_CREDIT_INVOICE"            # B01 / E31
    PURCHASE_FROM_NON_REGISTERED: str = "PURCHASE_FROM_NON_REGISTERED"  # B11 / E41
    MINOR_EXPENSE: str = "MINOR_EXPENSE"                            # B13 / E43
    SPECIAL_REGIME_NO_TAX: str = "SPECIAL_REGIME_NO_TAX"            # B14 / E44
    FOREIGN_PAYMENT: str = "FOREIGN_PAYMENT"                        # B17 / E47
    UNKNOWN: str = "UNKNOWN"


# Raw code -> Canonical type
CANONICAL_TYPE_MAP: Dict[str, str] = {
    "B01": CanonicalDocumentType.FISCAL_CREDIT_INVOICE,
    "E31": CanonicalDocumentType.FISCAL_CREDIT_INVOICE,

    "B11": CanonicalDocumentType.PURCHASE_FROM_NON_REGISTERED,
    "E41": CanonicalDocumentType.PURCHASE_FROM_NON_REGISTERED,

    "B13": CanonicalDocumentType.MINOR_EXPENSE,
    "E43": CanonicalDocumentType.MINOR_EXPENSE,

    "B14": CanonicalDocumentType.SPECIAL_REGIME_NO_TAX,
    "E44": CanonicalDocumentType.SPECIAL_REGIME_NO_TAX,

    "B17": CanonicalDocumentType.FOREIGN_PAYMENT,
    "E47": CanonicalDocumentType.FOREIGN_PAYMENT,
}


def normalize_document_type(raw_type: Any) -> str:
    """
    Normalize a raw document type from the NCF ledger.

    Inputs:
    - raw_type: can be None, string, or any value from CSV/XLSX parsing.

    Output:
    - Canonical document type string.

    Important:
    - If the type is unknown, return "UNKNOWN" and let policy decide whether
      to block or warn (MVP should not silently assume).
    """
    if raw_type is None:
        return CanonicalDocumentType.UNKNOWN

    code = str(raw_type).strip().upper()
    if not code:
        return CanonicalDocumentType.UNKNOWN

    return CANONICAL_TYPE_MAP.get(code, CanonicalDocumentType.UNKNOWN)