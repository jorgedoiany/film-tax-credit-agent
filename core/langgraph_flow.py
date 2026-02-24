"""
core/langgraph_flow.py

Clean LangGraph wrapper for the Art. 39 compliance MVP.

Refactoring goals:
- No direct state mutation
- No silent parsing failures
- Deterministic validation
- Explicit input structure validation
- Clear audit trace handling
"""

from __future__ import annotations

from typing import Dict, Any, TypedDict, List, Mapping
from langgraph.graph import StateGraph, END

from .report_crosscheck import cross_check_expenses, ReportingPolicy
from .threshold_validator import ThresholdEngine, FXPolicy, build_validated_rows_from_validator_output
from .participation_validator import ParticipationEngine, CrewRow
from .credit_calculator import CreditCalculator, CreditComputationInputs


# ============================================================
# State Definition
# ============================================================

class Art39State(TypedDict):
    ncf_rows: List[Dict[str, Any]]
    rows_606: List[Dict[str, Any]]
    rows_609: List[Dict[str, Any]]
    cast_crew_rows: List[Dict[str, Any]]

    applicant_id: str
    project_id: str
    fiscal_year: int
    fx_to_usd: Dict[str, float]

    waiver_override_present: bool
    resident_flags: Dict[str, bool]

    acquisition_fiscal_year: int
    transferee_isr_due_by_year_usd: Dict[int, float]

    enriched_ncf_rows: List[Dict[str, Any]]
    expense_validation_outputs: List[Dict[str, Any]]
    threshold_result: Dict[str, Any]
    participation_result: Dict[str, Any]
    credit_result: Dict[str, Any]

    compliance_status: Dict[str, Any]
    audit_trace: List[str]


# ============================================================
# Utility Helpers
# ============================================================

def _append_audit(state: Art39State, message: str) -> List[str]:
    """
    Return a new audit trace list with appended message.
    """
    current = list(state.get("audit_trace", []))
    current.append(message)
    return current


def _validate_required_keys(
        data: Mapping[str, Any],
        required: List[str],
        context: str,
) -> None:
    """
    Raise explicit error if required keys are missing.
    """
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required field '{key}' in {context}.")


def _parse_amount(value: Any) -> float:
    """
    Strict amount parser.
    Raises error if invalid instead of silently failing.
    """
    if value is None:
        raise ValueError("Amount value is None.")

    raw = str(value).replace(",", "").replace(" ", "").strip()
    try:
        return float(raw)
    except Exception as e:
        raise ValueError(f"Invalid numeric amount: {value}") from e


def _normalize_currency(value: str) -> str:
    return str(value).strip().upper()


# ============================================================
# Nodes
# ============================================================

def validate_input_node(state: Art39State) -> Art39State:
    """
    Ensure required top-level inputs exist before processing.
    """
    audit = _append_audit(state, "ValidateInputNode: Validating initial state structure.")

    required_keys = [
        "ncf_rows",
        "rows_606",
        "rows_609",
        "cast_crew_rows",
        "applicant_id",
        "project_id",
        "fiscal_year",
        "fx_to_usd",
    ]

    _validate_required_keys(state, required_keys, "initial state")

    return {**state, "audit_trace": audit}


def cross_check_node(state: Art39State) -> Art39State:
    audit = _append_audit(state, "CrossCheckNode: Running 606/609 reconciliation.")

    enriched = cross_check_expenses(
        state["ncf_rows"],
        rows_606=state["rows_606"],
        rows_609=state["rows_609"],
        policy=ReportingPolicy.default(),
    )

    return {
        **state,
        "enriched_ncf_rows": enriched,
        "audit_trace": audit,
    }


def validate_expenses_node(state: Art39State) -> Art39State:
    """
    Deterministic expense validation layer.

    Rules enforced:
    - Reporting must be verified
    - Must appear in 606
    - Gross amount must be > 0
    - If document type is B17 or E47 (foreign payment),
      it must also appear in 609
    """

    audit = _append_audit(state, "ValidateExpensesNode: Applying deterministic validation.")

    validated_rows = []

    for row in state["enriched_ncf_rows"]:
        _validate_required_keys(row, ["Total", "Tipo de Documento"], "NCF row")

        gross = _parse_amount(row["Total"])
        currency = _normalize_currency(row.get("Currency", "DOP"))
        doc_type = _normalize_currency(row["Tipo de Documento"])

        reporting_verified = row.get("_reporting_verified") is True
        reported_606 = row.get("_reported_in_606") is True
        reported_609 = row.get("_reported_in_609") is True

        # Base validation conditions
        valid = (
            reporting_verified
            and reported_606
            and gross > 0
        )

        # Foreign payment rule enforcement (B17 / E47)
        if doc_type in {"B17", "E47"}:
            if not reported_609:
                valid = False

        validated_rows.append({
            "expense_id": row.get("Numero NCF", "UNSPECIFIED"),
            "applicant_id": state["applicant_id"],
            "project_id": state["project_id"],
            "fiscal_year": state["fiscal_year"],
            "validatable": valid,
            "validated_amount": gross if valid else 0.0,
            "currency": currency,
        })

    return {
        **state,
        "expense_validation_outputs": validated_rows,
        "audit_trace": audit,
    }


def threshold_node(state: Art39State) -> Art39State:
    audit = _append_audit(state, "ThresholdNode: Evaluating USD 500k requirement.")

    fx_policy = FXPolicy(
        to_usd_rate={_normalize_currency(k): float(v) for k, v in state["fx_to_usd"].items()}
    )

    engine = ThresholdEngine(fx_policy=fx_policy)

    validated_rows = build_validated_rows_from_validator_output(
        state["expense_validation_outputs"]
    )

    result = engine.evaluate_single_project(
        rows=validated_rows,
        applicant_id=state["applicant_id"],
        project_id=state["project_id"],
        fiscal_year=state["fiscal_year"],
    )

    return {
        **state,
        "threshold_result": {
            "eligible_to_request": result.eligible_to_request,
            "total_validated_usd": result.total_validated_usd,
            "rationale": result.rationale,
        },
        "audit_trace": audit,
    }


def participation_node(state: Art39State) -> Art39State:
    audit = _append_audit(state, "ParticipationNode: Computing Dominican participation.")

    engine = ParticipationEngine()

    crew_rows = [
        CrewRow(
            position_job_title=r.get("Position/Job Title", ""),
            name=r.get("Name", ""),
            sirecine=r.get("SIRECINE (code or number)"),
            tax_id=r.get("Tax ID (code or number)"),
            id_number_passport=r.get("ID Number/Passport"),
            nationality=r.get("Nationality", ""),
            department=r.get("Department"),
            phone=r.get("Phone"),
            email=r.get("Email"),
        )
        for r in state["cast_crew_rows"]
    ]

    result = engine.evaluate(
        crew_rows,
        waiver_override_present=state.get("waiver_override_present", False),
        resident_flags=state.get("resident_flags", {}),
    )

    return {
        **state,
        "participation_result": {
            "compliant": result.compliant,
            "ratio": result.participation_ratio,
            "rationale": result.rationale,
        },
        "audit_trace": audit,
    }


def credit_node(state: Art39State) -> Art39State:
    audit = _append_audit(state, "CreditNode: Computing 25% transferable credit.")

    calculator = CreditCalculator()

    acquisition_year = state.get("acquisition_fiscal_year", state["fiscal_year"])

    inputs = CreditComputationInputs(
        total_validated_spend_usd=state["threshold_result"]["total_validated_usd"],
        acquisition_fiscal_year=acquisition_year,
        transferee_isr_due_by_year_usd=state.get("transferee_isr_due_by_year_usd"),
    )

    result = calculator.compute(inputs)

    return {
        **state,
        "credit_result": {
            "gross_credit_usd": result.gross_credit_usd,
            "warnings": result.warnings,
        },
        "audit_trace": audit,
    }


def summary_node(state: Art39State) -> Art39State:
    audit = _append_audit(state, "SummaryNode: Building final compliance decision.")

    threshold_ok = state["threshold_result"]["eligible_to_request"]
    participation_ok = state["participation_result"]["compliant"]

    compliance_status = {
        "threshold_met": threshold_ok,
        "participation_met": participation_ok,
        "eligible_to_request": threshold_ok and participation_ok,
        "validated_spend_usd": state["threshold_result"]["total_validated_usd"],
        "gross_credit_usd": state["credit_result"]["gross_credit_usd"],
    }

    return {
        **state,
        "compliance_status": compliance_status,
        "audit_trace": audit,
    }


# ============================================================
# Graph Builder
# ============================================================

def build_art39_graph():
    graph = StateGraph(Art39State)

    graph.add_node("ValidateInputNode", validate_input_node)
    graph.add_node("CrossCheckNode", cross_check_node)
    graph.add_node("ValidateExpensesNode", validate_expenses_node)
    graph.add_node("ThresholdNode", threshold_node)
    graph.add_node("ParticipationNode", participation_node)
    graph.add_node("CreditNode", credit_node)
    graph.add_node("SummaryNode", summary_node)

    graph.set_entry_point("ValidateInputNode")

    graph.add_edge("ValidateInputNode", "CrossCheckNode")
    graph.add_edge("CrossCheckNode", "ValidateExpensesNode")
    graph.add_edge("ValidateExpensesNode", "ThresholdNode")
    graph.add_edge("ThresholdNode", "ParticipationNode")
    graph.add_edge("ParticipationNode", "CreditNode")
    graph.add_edge("CreditNode", "SummaryNode")
    graph.add_edge("SummaryNode", END)

    return graph.compile()