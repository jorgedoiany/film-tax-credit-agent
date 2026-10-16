# Film Tax Credit Agent

> A deterministic Art. 39 compliance pipeline with optional local legal retrieval and explanation for film productions seeking the Dominican Republic's transferable tax credit.

## Status
- Implemented: a LangGraph pipeline that reconciles NCF rows with DGII 606/609 rows, validates basic reporting and amount conditions, checks the USD 500,000 threshold, evaluates Dominican/resident crew participation, estimates gross credit, and returns a compliance summary with an audit trace.
- Implemented: an optional Article 39 retrieval and local LLM explanation layer with checks that cited chunk IDs were among those retrieved.
- Not implemented: ingestion of source documents, DGII or DGCINE service integrations, and automatic verification of the evidence flags described by the domain models; the pipeline receives structured rows and explicit values.
- Not implemented: accumulated multi-project threshold evaluation in the LangGraph flow (the threshold engine exposes an accumulation method, but the flow calls single-project evaluation).
- TODO: supported and tested Python version.
- TODO: tested hardware requirements.

## What it does
This repository implements an early compliance-checking pipeline for the Dominican film incentive in Article 39 of Law 108-10. Given structured production expense rows, DGII 606/609 report rows, crew data, and exchange-rate inputs, it checks selected reporting, spend-threshold, participation, and credit-calculation conditions to help a production assess its request. It provides calculations and explanations, not a DGCINE validation or legal determination.

## Architecture
- The LangGraph flow runs the deterministic checks and calculations in sequence; the optional RAG/LLM step retrieves Article 39 text and produces an explanation, but does not change deterministic results.
- `legal_question` controls whether the graph performs retrieval and LLM generation; when absent or empty, that node skips the AI work.

| module in core/ | responsibility |
|---|---|
| `__init__.py` | Empty package initializer. |
| `credit_calculator.py` | Calculates gross credit from validated USD spend and, when tax-due inputs are provided, plans yearly use subject to the configured cap and carry-forward window. |
| `document_type_normalizer.py` | Maps listed NCF/e-CF codes to canonical document categories, returning `UNKNOWN` for other values. |
| `langgraph_flow.py` | Defines and connects the Art. 39 graph, including reporting cross-checks, basic expense validation, threshold and participation checks, credit calculation, and summary output. |
| `legal_rag.py` | Parses marked legal chunks, creates or loads local sentence-transformer embeddings, and retrieves the top matching chunks. |
| `local_llm.py` | Sends a prompt to the local Ollama generate endpoint and returns its response. |
| `models.py` | Defines shared expense, evidence, withholding, and validation dataclasses and enums. |
| `participation_validator.py` | Computes a Dominican-or-resident crew participation ratio using nationality aliases and optional explicit residency flags, with an optional waiver override. |
| `report_crosscheck.py` | Matches NCF rows to 606 and, for foreign-payment types, 609 rows, and records reporting verification facts and rationale. |
| `threshold_validator.py` | Converts validated expenses to USD using explicit rates and evaluates the threshold for one project or accumulated projects. |

## Requirements
- Python version: TODO: the repository does not specify a supported or tested Python version.
- Install the packages pinned in `requirements.txt`, including `torch` and `sentence-transformers`.
- When `legal_question` is non-empty, the code calls an Ollama server at `http://localhost:11434/api/generate` using model name `mistral`. The `run_local_test.py` sample supplies a legal question, so that run requires this local service.
- The RAG code constructs `SentenceTransformer` with model name `all-MiniLM-L6-v2`. TODO: document model download/cache setup and tested hardware requirements.

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   On Windows, activate it with `.venv\Scripts\activate`.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Keep the legal source at `data/legal_sources/article_39_structured.txt`; the RAG node refers to this repository-relative path. For the sample's non-empty legal question, start an Ollama server with the `mistral` model available. TODO: exact Ollama installation and model setup instructions; TODO: exact `all-MiniLM-L6-v2` model setup requirements.

## Usage
Run the local sample from the repository root:

```bash
python run_local_test.py
```

It invokes the graph with hard-coded sample NCF, 606, and cast/crew rows, then prints deterministic results, any legal analysis and citation output, and the audit trace. The script's sample includes a legal question and therefore requires the local RAG model and Ollama setup described above.

## Tests
Run the tests from the repository root:

```bash
python -m pytest
```

`tests/test_art39_golden.py` exercises a compliant case, a threshold failure, and a B17 expense missing its required 609 match through the graph. `tests/test_legal_citation_validation.py` checks that citation-validity output is present and that invalid citations, if reported, have warnings. `tests/test_expense_validator.py` is empty and currently covers no tests.

## Legal sources
- `data/legal_sources/article_39_structured.txt` is read by `LegalRAG` in the graph's legal-reasoning node. Its `[CHUNK_ID: ...]` sections and article/section/topic markers are parsed for retrieval and citation checking.
- `data/legal_sources/article_39.txt` contains plain Article 39 text, but repository code does not reference it.

## Known limitations
- Expense validation in the graph checks reporting verification, a 606 match, a positive gross amount, and a 609 match for B17/E47 rows. It does not perform the additional source-document or evidence checks suggested by the domain model.
- The graph evaluates a single project for the threshold; accumulated multi-project evaluation is not wired into the graph.
- Participation uses a default 25% minimum and nationality-based counting; non-Dominican residency counts only when explicit residency flags are supplied. The graph applies this participation check to the provided crew list.
- Credit-use planning depends on supplied transferee ISR due amounts. The calculator documents that it does not model the national 10% ISR cap or IT-1/asset-tax offsets in detail.
- The graph uses the structured Article 39 source; it does not connect to DGCINE or DGII services.

## Roadmap
- TODO: Specify supported and tested Python versions.
- TODO: Document local Ollama and `all-MiniLM-L6-v2` setup.
- TODO: Specify tested hardware requirements.
- TODO: Add tests for expense validation.
- TODO: Implement and test additional integrations or validation rules as requirements are defined.

## License
Proprietary. All rights reserved. See LICENSE.
