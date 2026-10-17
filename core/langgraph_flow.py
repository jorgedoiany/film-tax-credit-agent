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

from datetime import date
from .eligibility.engine import evaluate_project
from .eligibility.models import ExpenseRecord, ProjectContext, Rejection

import re
from typing import Dict, Any, TypedDict, List, Mapping, Optional, Tuple
from langgraph.graph import StateGraph, END
from dataclasses import asdict

from .report_crosscheck import cross_check_expenses, ReportingPolicy
from .threshold_validator import (
    ThresholdEngine,
    ThresholdPolicy,
    FXPolicy,
    build_validated_rows_from_validator_output,
)
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

    # Optional: "foreign" or "dominican". Art. 39 Paragraph III applies the
    # Dominican participation requirement to foreign productions only.
    production_origin: str

    # Optional: enables the DGCINE validation guide rules (caps, cash payments,
    # ITBIS matrix, Art. 168 exclusions). Keys: approved_budget_total,
    # budget_currency, permit_start, permit_end (ISO dates).
    project_context: Dict[str, Any]
    guide_rules_evaluated: bool
    eligibility_summary: Dict[str, Any]

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
    - Must appear in 606, with the same amount as the ledger
    - Gross amount must be > 0
    - Must not be a duplicate of an earlier row (same supplier RNC + NCF)
    - If document type is B17 or E47 (foreign payment),
      it must also appear in 609

    Each output row carries `exclusion_reasons` (empty when validatable) so the
    user can see exactly why an expense was excluded.
    """
    audit = _append_audit(state, "ValidateExpensesNode: Applying deterministic validation.")

    validated_rows: List[Dict[str, Any]] = []

    for position, row in enumerate(state.get("enriched_ncf_rows", []), start=1):
        _validate_required_keys(row, ["Total", "Tipo de Documento"], "NCF row")

        gross = _parse_amount(row["Total"])
        currency = _normalize_currency(row.get("Currency", "DOP"))
        doc_type = _normalize_currency(row["Tipo de Documento"])

        reporting_verified = row.get("_reporting_verified") is True
        reported_606 = row.get("_reported_in_606") is True
        reported_609 = row.get("_reported_in_609") is True
        amount_matches_606 = row.get("_606_amount_matches")
        duplicate_of = row.get("_duplicate_of_row")

        reasons: List[str] = []

        if not reporting_verified:
            reasons.append("Reporting could not be verified (missing RNC/NCF or required 609 data).")
        if not reported_606:
            reasons.append("Not found in 606.")
        elif amount_matches_606 is not True:
            reasons.append("Amount differs from the 606 report.")
        if gross <= 0:
            reasons.append("Amount must be greater than zero.")
        if doc_type in {"B17", "E47"} and not reported_609:
            reasons.append("Foreign payment not found in 609.")
        if duplicate_of is not None:
            reasons.append(
                f"Duplicate of row {duplicate_of} (same RNC and NCF; "
                f"this row Total={row['Total']}, row {duplicate_of} "
                f"Total={row.get('_duplicate_first_total')})."
            )

        valid = not reasons

        validated_rows.append({
            "expense_id": row.get("Numero NCF", "UNSPECIFIED"),
            "row_number": position,
            "applicant_id": state["applicant_id"],
            "project_id": state["project_id"],
            "fiscal_year": state["fiscal_year"],
            "validatable": valid,
            "validated_amount": gross if valid else 0.0,
            "currency": currency,
            "exclusion_reasons": reasons,
        })

    return {
        **state,
        "expense_validation_outputs": validated_rows,
        "audit_trace": audit,
    }

def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as e:
        raise ValueError(f"Invalid ISO date for '{field}': {value}") from e


def eligibility_node(state: Art39State) -> Art39State:
    """
    Apply the DGCINE validation guide rules (JRD-G43) to the expenses that the
    606/609 cross-check and basic validation produced.

    Only runs when `project_context` is provided; otherwise the flow behaves as
    before and the summary says the guide rules were not evaluated.

    - The cross-check owns the NCF, duplicate and 606/609 rules; its rejections
      enter the engine as "pre-rejected" and do not count toward the 6% / 3% caps.
    - Each expense is identified internally by its row position, so repeated
      NCFs cannot overwrite each other.
    """
    audit = _append_audit(state, "EligibilityNode: Applying DGCINE validation guide rules.")

    context = state.get("project_context")
    if not context:
        return {
            **state,
            "guide_rules_evaluated": False,
            "eligibility_summary": {},
            "audit_trace": audit,
        }

    _validate_required_keys(
        context,
        ["approved_budget_total", "budget_currency", "permit_start", "permit_end"],
        "project_context",
    )

    project = ProjectContext(
        project_id=state["project_id"],
        applicant_id=state["applicant_id"],
        fiscal_year=state["fiscal_year"],
        approved_budget_total=_parse_amount(context["approved_budget_total"]),
        budget_currency=_normalize_currency(context["budget_currency"]),
        permit_start=_parse_date(context["permit_start"], "project_context.permit_start"),
        permit_end=_parse_date(context["permit_end"], "project_context.permit_end"),
    )

    outputs = state["expense_validation_outputs"]
    expenses: List[ExpenseRecord] = []
    prerejected: Dict[str, Tuple[Rejection, ...]] = {}

    for index, (row, out) in enumerate(zip(state["enriched_ncf_rows"], outputs), start=1):
        context_label = f"NCF row {index}"
        _validate_required_keys(row, ["date", "cost", "itbis", "concept_code"], context_label)

        engine_id = f"row-{index}"
        total = _parse_amount(row["Total"])

        expenses.append(ExpenseRecord(
            expense_id=engine_id,
            project_id=state["project_id"],
            applicant_id=state["applicant_id"],
            fiscal_year=state["fiscal_year"],
            date=_parse_date(row["date"], f"{context_label} date"),
            account=str(row.get("account", "UNSPECIFIED")),
            ncf=row.get("Numero NCF"),
            supplier=str(row.get("supplier", "")),
            rnc=str(row.get("RNC", "")),
            details="",
            cost=_parse_amount(row["cost"]),
            itbis=_parse_amount(row["itbis"]),
            total=total,
            payment_method=row.get("payment_method"),
            concept_code=str(row["concept_code"]),
            currency=out["currency"],
            supplier_sirecine=row.get("supplier_sirecine") is True,
        ))

        if not out["validatable"]:
            prerejected[engine_id] = tuple(
                Rejection(
                    "REPORTING_CROSSCHECK",
                    reason,
                    "Cross-check against DGII 606/609 (Guide JRD-G43, 8.7-8.8)",
                    total,
                )
                for reason in out["exclusion_reasons"]
            )

    fx_policy = FXPolicy(
        to_usd_rate={_normalize_currency(k): float(v) for k, v in state["fx_to_usd"].items()}
    )

    report = evaluate_project(
        expenses,
        project,
        fx_policy,
        prerejected=prerejected,
        skip_ncf_rules=True,
    )
    decisions = {d.expense_id: d for d in report.decisions}

    updated: List[Dict[str, Any]] = []
    for index, out in enumerate(outputs, start=1):
        decision = decisions[f"row-{index}"]

        reasons = list(out["exclusion_reasons"])
        for rejection in decision.rejections:
            if rejection.message not in reasons:
                reasons.append(rejection.message)

        counts = decision.status != "rejected"

        updated.append({
            **out,
            "validatable": counts,
            "validated_amount": decision.admitted_cost if counts else 0.0,
            "exclusion_reasons": reasons,
            "eligibility": {
                "status": decision.status,
                "admitted_amount": decision.admitted_cost,
                "excluded_amount": round(decision.cost - decision.admitted_cost, 2),
                "rejections": [asdict(r) for r in decision.rejections],
            },
        })

    summary = {
        "currency": report.currency,
        "executed_total": report.executed_total,
        "admitted_total": report.admitted_total,
        "rejected_total": report.rejected_total,
        "rejected_by_primary_rule": report.rejected_by_primary_rule,
        "caps": {name: asdict(cap) for name, cap in report.caps.items()},
        "unconverted_row_numbers": [
            int(expense_id.split("-")[1]) for expense_id in report.unconverted_expense_ids
        ],
    }

    return {
        **state,
        "expense_validation_outputs": updated,
        "guide_rules_evaluated": True,
        "eligibility_summary": summary,
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

    policy = ThresholdPolicy()
    engine = ThresholdEngine(fx_policy=fx_policy, threshold_policy=policy)

    validated_rows = build_validated_rows_from_validator_output(
        state["expense_validation_outputs"]
    )

    result = engine.evaluate_single_project(
        rows=validated_rows,
        applicant_id=state["applicant_id"],
        project_id=state["project_id"],
        fiscal_year=state["fiscal_year"],
    )

    shortfall = max(0.0, policy.minimum_usd - result.total_validated_usd)

    return {
        **state,
        "threshold_result": {
            "eligible_to_request": result.eligible_to_request,
            "total_validated_usd": result.total_validated_usd,
            "minimum_usd": policy.minimum_usd,
            "shortfall_usd": round(shortfall, 2),
            "counted_expenses": result.counted_expenses,
            "excluded_expenses": result.excluded_expenses,
            "rationale": result.rationale,
        },
        "audit_trace": audit,
    }


def participation_node(state: Art39State) -> Art39State:
    """
    Compute Dominican/resident participation ratio from Cast & Crew list.

    Art. 39 Paragraph III applies the minimum participation to FOREIGN
    productions. For a Dominican production the requirement does not apply
    and does not block the request.

    IMPORTANT:
    - "Participation of Dominican personnel" refers to crew/staff positions,
      NOT suppliers/vendors.
    - If production_origin is missing, "foreign" is assumed (strict) and a
      warning is returned.
    """
    audit = _append_audit(state, "ParticipationNode: Computing Dominican participation.")

    warnings: List[str] = []
    origin_raw = state.get("production_origin")

    if origin_raw is None or not str(origin_raw).strip():
        origin = "foreign"
        warnings.append(
            "production_origin not provided; assuming a foreign production, "
            "so the Art. 39 Paragraph III participation requirement is applied."
        )
    else:
        origin = str(origin_raw).strip().lower()
        if origin not in {"foreign", "dominican"}:
            raise ValueError(
                f"Invalid production_origin '{origin_raw}'. Use 'foreign' or 'dominican'."
            )

    applicable = origin == "foreign"

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

    rationale = list(result.rationale)
    if not applicable:
        rationale.append(
            "Production origin is Dominican: the Paragraph III participation "
            "requirement does not apply."
        )

    return {
        **state,
        "participation_result": {
            "applicable": applicable,
            # True when the requirement is met OR does not apply.
            "compliant": result.compliant if applicable else True,
            "ratio": result.participation_ratio,
            "minimum_required_ratio": result.minimum_required_ratio,
            "dominican_or_resident_count": result.dominican_or_resident_count,
            "total_personnel_count": result.total_personnel_count,
            "waiver_override_used": result.waiver_override_used,
            "issues": [asdict(i) for i in result.issues],
            "warnings": warnings,
            "rationale": rationale,
        },
        "audit_trace": audit,
    }


def credit_node(state: Art39State) -> Art39State:
    """
    Compute gross transferable credit (25% of validated spend) and produce warnings.

    Note:
    - Usability/caps are handled by CreditCalculator, not by the LLM.
    - This is an informational calculation on validated spend; whether the
      credit can be requested is decided in SummaryNode.
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
            "year_plan": [asdict(p) for p in result.year_plan],
            "warnings": result.warnings,
            "rationale": result.rationale,
        },
        "audit_trace": audit,
    }


def summary_node(state: Art39State) -> Art39State:
    """
    Combine deterministic outcomes into the final compliance status.

    - gross_credit_usd: informational, 25% of validated spend.
    - requestable_credit_usd: the same amount only if the request is eligible,
      otherwise 0.0.
    - excluded_expenses: every expense excluded entirely, with reasons.
    - partially_admitted_expenses: expenses admitted only in part (for example
      producer fees above the 6% cap), with admitted and excluded amounts.
    - fx_excluded_expenses: validated expenses left out of the threshold
      because no FX rate was provided (a configuration issue, not a spend issue).
    - guide_rules_evaluated: False when no project_context was provided, so
      the DGCINE guide rules were not applied.
    - blocking_reasons: short list explaining why the request is not eligible.
    """
    audit = _append_audit(state, "SummaryNode: Building final compliance decision.")

    threshold = state["threshold_result"]
    participation = state["participation_result"]
    outputs = state["expense_validation_outputs"]

    threshold_ok = threshold["eligible_to_request"]
    participation_ok = participation["compliant"]
    eligible = threshold_ok and participation_ok

    gross_credit = state["credit_result"]["gross_credit_usd"]

    excluded_expenses = [
        {
            "expense_id": row["expense_id"],
            "row_number": row["row_number"],
            "reasons": row["exclusion_reasons"],
        }
        for row in outputs
        if not row["validatable"]
    ]

    partially_admitted_expenses = [
        {
            "expense_id": row["expense_id"],
            "row_number": row["row_number"],
            "admitted_amount": row["eligibility"]["admitted_amount"],
            "excluded_amount": row["eligibility"]["excluded_amount"],
            "reasons": row["exclusion_reasons"],
        }
        for row in outputs
        if row.get("eligibility", {}).get("status") == "partial"
    ]

    fx_excluded = [
        e for e in threshold["excluded_expenses"]
        if "missing FX rate" in e["reason"]
    ]

    blocking_reasons: List[str] = []

    if not threshold_ok:
        blocking_reasons.append(
            f"Validated spend USD {threshold['total_validated_usd']:,.2f} is below the "
            f"USD {threshold['minimum_usd']:,.2f} minimum "
            f"(short by USD {threshold['shortfall_usd']:,.2f})."
        )

    if fx_excluded:
        blocking_reasons.append(
            f"{len(fx_excluded)} validated expense(s) were left out because no FX rate "
            "was provided for their currency; the validated spend may be understated."
        )

    if not participation_ok:
        blocking_reasons.append(
            f"Dominican/resident participation {participation['ratio']:.1%} is below the "
            f"required {participation['minimum_required_ratio']:.0%}."
        )

    guide_evaluated = state.get("guide_rules_evaluated", False)

    warnings: List[str] = list(participation["warnings"])
    if not guide_evaluated:
        warnings.append(
            "project_context not provided; the exclusion rules of the DGCINE validation "
            "guide (budget caps, cash payments, ITBIS, Art. 168 exclusions) were not "
            "evaluated, so the validated spend may include expenses the guide excludes."
        )

    compliance_status = {
        "threshold_met": threshold_ok,
        "participation_met": participation_ok,
        "participation_applicable": participation["applicable"],
        "guide_rules_evaluated": guide_evaluated,
        "eligible_to_request": eligible,
        "validated_spend_usd": threshold["total_validated_usd"],
        "gross_credit_usd": gross_credit,
        "requestable_credit_usd": gross_credit if eligible else 0.0,
        "blocking_reasons": blocking_reasons,
        "excluded_expenses": excluded_expenses,
        "partially_admitted_expenses": partially_admitted_expenses,
        "fx_excluded_expenses": fx_excluded,
        "warnings": warnings,
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
    graph.add_node("EligibilityNode", eligibility_node)

    graph.set_entry_point("ValidateInputNode")

    graph.add_edge("ValidateInputNode", "LegalReasoningNode")
    graph.add_edge("LegalReasoningNode", "CrossCheckNode")
    graph.add_edge("CrossCheckNode", "ValidateExpensesNode")
    graph.add_edge("ValidateExpensesNode", "EligibilityNode")
    graph.add_edge("EligibilityNode", "ThresholdNode")
    graph.add_edge("ThresholdNode", "ParticipationNode")
    graph.add_edge("ParticipationNode", "CreditNode")
    graph.add_edge("CreditNode", "SummaryNode")
    graph.add_edge("SummaryNode", END)

    return graph.compile()