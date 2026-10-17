"""
run_local_test.py

Manual deterministic + RAG test runner for the Art. 39 LangGraph pipeline.

Run with:
    python run_local_test.py

Make sure:
    ollama serve
is running if using LegalReasoningNode.
"""

from __future__ import annotations

from typing import cast
from core.langgraph_flow import build_art39_graph, Art39State


def main() -> None:
    # ------------------------------------------------------------
    # 1) Fake Minimal Input Data
    # ------------------------------------------------------------

    ncf_rows = [
        {
            "Numero NCF": "B0100000001",
            "RNC": "101234567",
            "Total": "40000000",  # DOP
            "Tipo de Documento": "B01",
        }
    ]

    rows_606 = [
        {
            "RNC o Cédula": "101234567",
            "NCF": "B0100000001",
            "Total Monto Facturado": "40000000",
        }
    ]

    rows_609 = []  # Not needed for this test

    cast_crew_rows = [
        {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
        {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
        {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
    ]

    # ------------------------------------------------------------
    # 2) Build Initial State
    # ------------------------------------------------------------

    initial_state = cast(Art39State, {
        "ncf_rows": ncf_rows,
        "rows_606": rows_606,
        "rows_609": rows_609,
        "cast_crew_rows": cast_crew_rows,

        # Legal question triggers RAG + Ollama reasoning
        "legal_question": "Explain the relationship between Article 39 and Article 999",

        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,

        # FX rate (MVP hardcoded)
        "fx_to_usd": {"DOP": 1 / 58.5, "USD": 1.0},

        # Participation settings
        "production_origin": "foreign",  # "foreign" or "dominican" (Art. 39 Paragraph III)
        "waiver_override_present": False,
        "resident_flags": {},

        # Credit usage
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},

        # Always initialize
        "audit_trace": [],
    })

    # ------------------------------------------------------------
    # 3) Build Graph and Run
    # ------------------------------------------------------------

    graph = build_art39_graph()

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        print("\n⚠️ ERROR DURING EXECUTION")
        print(e)
        return

    # ------------------------------------------------------------
    # 4) Print Deterministic Results
    # ------------------------------------------------------------

    print("\n=== COMPLIANCE STATUS ===")
    print(result["compliance_status"])

    print("\n=== THRESHOLD RESULT ===")
    print(result["threshold_result"])

    print("\n=== PARTICIPATION RESULT ===")
    print(result["participation_result"])

    print("\n=== CREDIT RESULT ===")
    print(result["credit_result"])

    # ------------------------------------------------------------
    # 5) Print AI Layer (if available)
    # ------------------------------------------------------------

    if result.get("legal_analysis"):
        print("\n=== LEGAL ANALYSIS (LOCAL LLM) ===")
        print(result["legal_analysis"])

    if result.get("legal_retrieved_chunk_ids"):
        print("\n=== RETRIEVED LEGAL CHUNKS ===")
        print(result["legal_retrieved_chunk_ids"])

    if result.get("legal_citation_valid") is not None:
        print("\n=== CITATION VALIDATION ===")
        print("Valid:", result["legal_citation_valid"])

    if result.get("legal_citation_warnings"):
        print("Warnings:", result["legal_citation_warnings"])

    # ------------------------------------------------------------
    # 6) Audit Trace
    # ------------------------------------------------------------

    print("\n=== AUDIT TRACE ===")
    for line in result["audit_trace"]:
        print("-", line)


if __name__ == "__main__":
    main()