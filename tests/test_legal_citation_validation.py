"""
tests/test_legal_citation_validation.py

Deterministic tests for the citation enforcement layer in LegalReasoningNode.
The RAG retriever and the local LLM are replaced by fakes, so these tests need
neither torch/sentence-transformers nor a running Ollama server.
"""

import sys
import types
from typing import cast

from core.langgraph_flow import build_art39_graph, Art39State


class _FakeChunk:
    def __init__(self, chunk_id: str, content: str) -> None:
        self.chunk_id = chunk_id
        self.content = content


def _install_fakes(monkeypatch, llm_response: str) -> None:
    """Replace core.legal_rag and core.local_llm with in-memory fakes."""
    fake_rag = types.ModuleType("core.legal_rag")
    fake_llm = types.ModuleType("core.local_llm")

    class FakeRAG:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, query, k=3):
            return [
                _FakeChunk("art39_para_ii", "Threshold text."),
                _FakeChunk("art39_para_iii", "Participation text."),
            ]

    class FakeLLM:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, prompt: str) -> str:
            return llm_response

    fake_rag.LegalRAG = FakeRAG
    fake_llm.LocalLLM = FakeLLM

    monkeypatch.setitem(sys.modules, "core.legal_rag", fake_rag)
    monkeypatch.setitem(sys.modules, "core.local_llm", fake_llm)


def _state(question: str) -> Art39State:
    return cast(Art39State, {
        "ncf_rows": [],
        "rows_606": [],
        "rows_609": [],
        "cast_crew_rows": [],
        "legal_question": question,
        "applicant_id": "TEST",
        "project_id": "TEST",
        "fiscal_year": 2026,
        "fx_to_usd": {"USD": 1.0},
        "waiver_override_present": False,
        "resident_flags": {},
        "acquisition_fiscal_year": 2026,
        "transferee_isr_due_by_year_usd": {},
        "audit_trace": [],
    })


def test_flags_invented_citation(monkeypatch):
    _install_fakes(
        monkeypatch,
        "Citations:\nart39_para_ii\nart39_imaginary_z",
    )

    result = build_art39_graph().invoke(_state("What does paragraph Z say?"))

    assert result["legal_citation_valid"] is False
    assert len(result["legal_citation_warnings"]) == 1
    assert "art39_imaginary_z" in result["legal_citation_warnings"][0]


def test_accepts_retrieved_citations(monkeypatch):
    _install_fakes(
        monkeypatch,
        "Citations:\nart39_para_ii\nart39_para_iii",
    )

    result = build_art39_graph().invoke(_state("What is the minimum spend?"))

    assert result["legal_citation_valid"] is True
    assert result["legal_citation_warnings"] == []
    assert result["legal_retrieved_chunk_ids"] == ["art39_para_ii", "art39_para_iii"]


def test_ai_layer_is_skipped_without_question():
    # No fakes installed: with an empty question the RAG must not even be imported.
    result = build_art39_graph().invoke(_state(""))

    assert result["legal_analysis"] == ""
    assert result["legal_citation_valid"] is True
    assert result["legal_retrieved_chunk_ids"] == []
