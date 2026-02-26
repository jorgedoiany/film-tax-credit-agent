"""
ui/streamlit_app.py

Simple UI for Art. 39 Compliance Engine (v1.1.0)

This is a thin presentation layer.
All compliance logic lives in core/.
"""

import streamlit as st
from core.langgraph_flow import build_art39_graph


st.set_page_config(page_title="Art. 39 Compliance Engine", layout="wide")

st.title("🎬 Article 39 – Film Tax Credit Compliance")
st.markdown("Hybrid deterministic + explainable AI compliance engine.")

st.divider()

st.subheader("Run Sample Evaluation")

if st.button("Run Demo Scenario"):

    graph = build_art39_graph()

    initial_state = {
        "ncf_rows": [
            {
                "Numero NCF": "B0100000001",
                "RNC": "101234567",
                "Total": "40000000",
                "Tipo de Documento": "B01",
            }
        ],
        "rows_606": [
            {
                "RNC o Cédula": "101234567",
                "NCF": "B0100000001",
                "Total Monto Facturado": "40000000",
            }
        ],
        "rows_609": [],
        "cast_crew_rows": [
            {"Position/Job Title": "Gaffer", "Name": "Juan Perez", "Nationality": "Dominican"},
            {"Position/Job Title": "Director", "Name": "John Smith", "Nationality": "American"},
            {"Position/Job Title": "Producer", "Name": "Maria Lopez", "Nationality": "Dominican"},
        ],
        "legal_question": "Can multiple projects be accumulated to reach the threshold?",
        "applicant_id": "101234567",
        "project_id": "MOVIE-001",
        "fiscal_year": 2026,
        "fx_to_usd": {"DOP": 1 / 58.5, "USD": 1.0},
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {2026: 500_000},
        "audit_trace": [],
    }

    result = graph.invoke(initial_state)

    st.success("Evaluation Completed")

    st.subheader("Compliance Status")
    st.json(result["compliance_status"])

    st.subheader("Legal Analysis")
    st.write(result.get("legal_analysis", "No legal analysis generated."))