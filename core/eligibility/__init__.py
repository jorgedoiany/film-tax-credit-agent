"""Eligibility engine: decides which expenses are admitted for the incentive.

Independent prototype. Not an official DGCINE determination.
"""
from .catalog import Concept, load_catalog
from .engine import (
    EligibilityParams,
    build_threshold_rows,
    evaluate_project,
)
from .models import (
    AccountSummary,
    CapInfo,
    EligibilityReport,
    ExpenseDecision,
    ExpenseRecord,
    ProjectContext,
    Rejection,
)

__all__ = [
    "AccountSummary", "CapInfo", "Concept", "EligibilityParams",
    "EligibilityReport", "ExpenseDecision", "ExpenseRecord",
    "ProjectContext", "Rejection", "build_threshold_rows",
    "evaluate_project", "load_catalog",
]