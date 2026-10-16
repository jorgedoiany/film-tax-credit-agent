"""
core/langgraph_flow.py

Clean LangGraph wrapper for the Art. 39 compliance MVP.

Refactoring goals:
- No direct state mutation
- No silent parsing failures
- Deterministic validation
- Explicit input structure validation
- Clear audit trace handling
- Optional Legal RAG + Local LLM reasoning (Ollama) without overriding deterministic results
"""

from __future__ import annotations

import re
from typing import Dict, Any, TypedDict, List, Mapping, Optional
from langgraph.graph import StateGraph, END

from .report_crosscheck import cross_check_expenses, ReportingPolicy
from .threshold_validator import ThresholdEngine, FXPolicy, build_validated_rows_from_validator_output
from .participation_validator import ParticipationEngine, CrewRow
from .credit_calculator import CreditCalculator, CreditComputationInputs


# ============================================================
# State Definition
# ============================================================

class Art39State(TypedDict):
    # Raw inputs
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

    # Optional: AI layer input/output
    legal_question: str
    legal_analysis: str
    legal_retrieved_chunk_ids: List[str]

    # NEW: Citation enforcement outputs
    legal_citation_valid: bool
    legal_citation_warnings: List[str]

    # Intermediate outputs
    enriched_ncf_rows: List[Dict[str, Any]]
    expense_validation_outputs: List[Dict[str, Any]]
    threshold_result: Dict[str, Any]
    participation_result: Dict[str, Any]
    credit_result: Dict[str, Any]

    # Final outputs
    compliance_status: Dict[str, Any]
    audit_trace: List[str]


# ============================================================
# Utility Helpers
# ============================================================

def _append_audit(state: Mapping[str, Any], message: str) -> List[str]:
    """
    Return a new audit trace list with appended message.

    Why we copy:
    - Avoid accidental shared list references across state transitions.
    - LangGraph nodes must return new dictionaries; audit_trace should behave similarly.
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

    Why strict:
    - In compliance systems, missing fields should be loud failures.
    - Silent fallbacks create "false compliance" which is dangerous.
    """
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required field '{key}' in {context}.")


def _parse_amount(value: Any) -> float:
    """
    Strict amount parser.
    Raises error if invalid instead of silently failing.

    Why strict:
    - We want deterministic failures for bad data.
    - Silencing errors leads to incorrect validated spend / credit estimates.
    """
    if value is None:
        raise ValueError("Amount value is None.")

    raw = str(value).replace(",", "").replace(" ", "").strip()
    try:
        return float(raw)
    except Exception as e:
        raise ValueError(f"Invalid numeric amount: {value}") from e


def _normalize_currency(value: Any) -> str:
    """
    Normalize currency codes to uppercase strings.

    Why:
    - Inputs may be 'dop', 'DOP ', etc.
    - We normalize once to avoid inconsistent logic.
    """
    return str(value).strip().upper()


# ============================================================
# Nodes
# ============================================================

def validate_input_node(state: Art39State) -> Art39State:
    """
    Ensure required top-level inputs exist before processing.

    Also initializes audit_trace if missing to prevent downstream failures.
    """
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

    # Ensure audit_trace exists from the start (prevents KeyError & mutable pitfalls).
    base_trace = list(state.get("audit_trace", []))
    if not base_trace:
        base_trace = []

    audit = base_trace + ["ValidateInputNode: Validating initial state structure."]

    # Initialize AI fields defensively so downstream code can rely on presence.
    # (These are optional at runtime, but explicit initialization reduces confusion.)
    return {
        **state,
        "audit_trace": audit,
        "legal_analysis": state.get("legal_analysis", ""),
        "legal_retrieved_chunk_ids": state.get("legal_retrieved_chunk_ids", []),
        "legal_citation_valid": state.get("legal_citation_valid", True),
        "legal_citation_warnings": state.get("legal_citation_warnings", []),
    }


def legal_reasoning_node(state: Art39State) -> Art39State:
    """
    Optional AI layer:
    - Retrieves legal context from Article 39.
    - Uses Ollama local LLM to draft a legal explanation.
    - Enforces citation integrity:
        - Only chunk IDs from retrieved context are allowed.
        - Detects and flags invalid citations.

    IMPORTANT:
    - This node does NOT decide compliance.
    - It does NOT modify validated spend, thresholds, or credit calculations.
    - It only adds explanatory context to the state for UI / reporting purposes.
    """
    audit = _append_audit(
        state,
        "LegalReasoningNode: Retrieving Article 39 context + generating local legal analysis."
    )

    question = state.get("legal_question", "").strip()
    if not question:
        # No question means: skip AI reasoning, keep deterministic pipeline intact.
        return {
            **state,
            "audit_trace": audit,
            "legal_analysis": "",
            "legal_retrieved_chunk_ids": [],
            "legal_citation_valid": True,
            "legal_citation_warnings": [],
        }

    # Imported lazily: the RAG stack (torch, sentence-transformers) is optional
    # and only needed when a legal_question is provided.
    from .legal_rag import LegalRAG
    from .local_llm import LocalLLM

    rag = LegalRAG("data/legal_sources/article_39_structured.txt")
    llm = LocalLLM(model="mistral")  # Keep model configurable if you wish.

    chunks = rag.retrieve(question, k=3)
    retrieved_ids = [c.chunk_id for c in chunks]

    context = "\n\n".join([f"{c.chunk_id}:\n{c.content}" for c in chunks])

    # Prompt is strict on sources and citations.
    allowed_chunks = ", ".join(retrieved_ids)

    prompt = f"""
You are a legal analyst specializing in Dominican Film Law (Article 39).

STRICT RULES:
1. You may ONLY use the legal context provided below.
2. You may ONLY cite chunk IDs from this list:
   [{allowed_chunks}]
3. If the answer is not fully contained in the provided chunks,
   explicitly state: "Not found in provided context."
4. Do NOT invent new chunk IDs.
5. Do NOT reference any paragraph not included in the retrieved chunks.

LEGAL CONTEXT:
{context}

QUESTION:
{question}

RESPONSE FORMAT:

Legal Interpretation:
[Your explanation]

Citations:
[List of chunk IDs used, exactly as provided above]

Practical Compliance Implications:
[Concrete explanation of what the production must do]
""".strip()

    analysis = llm.generate(prompt).strip()

    # ------------------------------------------------------------
    # Citation Validation Layer (Option A)
    # ------------------------------------------------------------
    # We enforce that any cited chunk IDs in the output must be
    # a subset of retrieved_ids. If not, we flag it.
    #
    # Why:
    # - Even with strong prompts, LLMs can invent references.
    # - In Legal-Tech, invented citations are unacceptable.
    # - We keep the deterministic pipeline independent from this.
    found_ids = re.findall(r"\bart39_[a-zA-Z0-9_]+\b", analysis)
    invalid_ids = [cid for cid in found_ids if cid not in retrieved_ids]

    citation_valid = len(invalid_ids) == 0
    citation_warnings: List[str] = []

    if not citation_valid:
        citation_warnings.append(
            f"Invalid citation(s) detected: {invalid_ids}. Allowed IDs: {retrieved_ids}"
        )
        audit = _append_audit(
            {"audit_trace": audit},
            "LegalReasoningNode: WARNING - invalid citations detected in LLM output."
        )

    return {
        **state,
        "legal_analysis": analysis,
        "legal_retrieved_chunk_ids": retrieved_ids,
        "legal_citation_valid": citation_valid,
        "legal_citation_warnings": citation_warnings,
        "audit_trace": audit,
    }


def cross_check_node(state: Art39State) -> Art39State:
    """
    Deterministic reconciliation between:
    - NCF ledger rows
    - DGII 606 export
    - DGII 609 export

    Output is an enriched version of the NCF ledger rows.
    """
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

    Rules enforced (MVP):
    - Reporting must be verified
    - Must appear in 606
    - Gross amount must be > 0
    - If document type is B17 or E47 (foreign payment),
      it must also appear in 609

    Why here (not in cross-check):
    - Cross-check computes facts (reported_in_606/609, verified).
    - Validation decides whether the expense is currently "validatable".
    """
    audit = _append_audit(state, "ValidateExpensesNode: Applying deterministic validation.")

    validated_rows: List[Dict[str, Any]] = []

    for row in state.get("enriched_ncf_rows", []):
        _validate_required_keys(row, ["Total", "Tipo de Documento"], "NCF row")

        gross = _parse_amount(row["Total"])
        currency = _normalize_currency(row.get("Currency", "DOP"))
        doc_type = _normalize_currency(row["Tipo de Documento"])

        reporting_verified = row.get("_reporting_verified") is True
        reported_606 = row.get("_reported_in_606") is True
        reported_609 = row.get("_reported_in_609") is True

        # Base conditions (local purchases must be in 606 and verification must be possible)
        valid = reporting_verified and reported_606 and gross > 0

        # Foreign payment enforcement
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
    """
    Evaluate minimum spend threshold (USD 500,000) under Art. 39 paragraph II.

    IMPORTANT:
    - This node uses deterministic FX conversion.
    - It uses outputs from ValidateExpensesNode (validated_amount + currency).
    """
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
    """
    Compute Dominican/resident participation ratio from Cast & Crew list.

    IMPORTANT:
    - "Participation of Dominican personnel" refers to crew/staff positions,
      NOT suppliers/vendors.
    """
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
    """
    Compute gross transferable credit (25% of validated spend) and produce warnings.

    Note:
    - Usability/caps are handled by CreditCalculator, not by the LLM.
    """
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
    """
    Combine deterministic outcomes into final compliance status.
    """
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
    """
    Build and compile the Art. 39 compliance graph.

    Flow:
    ValidateInput -> LegalReasoning -> CrossCheck -> ValidateExpenses -> Threshold
    -> Participation -> Credit -> Summary -> END
    """
    graph = StateGraph(Art39State)

    graph.add_node("ValidateInputNode", validate_input_node)
    graph.add_node("LegalReasoningNode", legal_reasoning_node)
    graph.add_node("CrossCheckNode", cross_check_node)
    graph.add_node("ValidateExpensesNode", validate_expenses_node)
    graph.add_node("ThresholdNode", threshold_node)
    graph.add_node("ParticipationNode", participation_node)
    graph.add_node("CreditNode", credit_node)
    graph.add_node("SummaryNode", summary_node)

    graph.set_entry_point("ValidateInputNode")

    graph.add_edge("ValidateInputNode", "LegalReasoningNode")
    graph.add_edge("LegalReasoningNode", "CrossCheckNode")
    graph.add_edge("CrossCheckNode", "ValidateExpensesNode")
    graph.add_edge("ValidateExpensesNode", "ThresholdNode")
    graph.add_edge("ThresholdNode", "ParticipationNode")
    graph.add_edge("ParticipationNode", "CreditNode")
    graph.add_edge("CreditNode", "SummaryNode")
    graph.add_edge("SummaryNode", END)

    return graph.compile()