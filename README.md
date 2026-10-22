# Film Tax Credit Agent

> **Note:** This repository is under active development. The latest code and data are available on the `develop` branch.

> MVP: a deterministic Art. 39 compliance pipeline with optional local legal retrieval and explanation for film productions seeking the Dominican Republic's transferable tax credit.

## Status
- Implemented: a LangGraph pipeline that reconciles NCF rows with DGII 606/609 rows, validates basic reporting and amount conditions, applies the DGCINE validation guide rules (when `project_context` is provided), checks the USD 500,000 threshold on the admitted amount, evaluates Dominican/resident crew participation for foreign productions, estimates gross credit, and returns a compliance summary with per-expense exclusion reasons and an audit trace.
- Implemented: an expense is excluded when it is missing from the 606, when its amount differs from the 606, or when it duplicates an earlier row (same supplier RNC and NCF).
- Implemented (ported from the Alexa+ hackathon repo): exclusion rules of the guide JRD-G43 for cash payments, the 6% producer-fee and 3% development-cost caps (partial admission of the excess), financial charges, marketing and distribution, fixed assets, alcohol, tobacco, and the ITBIS matrix with the B14/E44 exemptions.
- Implemented: an optional Article 39 retrieval and local LLM explanation layer with citations restricted to retrieved chunks: cited chunk IDs outside the retrieved set are flagged as invalid (they are detected, not removed).
- Not implemented: guide rules that need human review or external checks (unrelated expenses, valid NCF lookup on the DGII site, incomplete supporting documents, withholdings, unsigned checks, illegible invoices).
- Not implemented: ingestion of source documents, DGII or DGCINE service integrations; the pipeline receives structured rows and explicit values.
- Not implemented: accumulated multi-project threshold evaluation in the LangGraph flow, and fiscal year per expense (one fiscal year is applied to all rows).

## What it does

This repository implements an early compliance-checking pipeline for the Dominican film incentive in Article 39 of Law 108-10. Given structured production expense rows, DGII 606/609 report rows, crew data, and exchange-rate inputs, it checks selected reporting, spend-threshold, participation, and credit-calculation conditions to help a production assess its request. It provides calculations and explanations, not a DGCINE validation or legal determination.

## Architecture

- The LangGraph flow runs the deterministic checks and calculations in sequence; the optional RAG/LLM step retrieves Article 39 text and produces an explanation, but does not change deterministic results.
- `legal_question` controls whether the graph performs retrieval and LLM generation; when absent or empty, that node skips the AI work.
- When `project_context` is provided, `EligibilityNode` runs after expense validation. The 606/609 cross-check owns the NCF, duplicate and reporting rules; its rejections enter the engine as pre-rejected and do not count toward the budget caps. When it is not provided, the guide rules are skipped and the result says so.

| module in core/               | responsibility                                                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                 | Empty package initializer.                                                                                                                                                      |
| `credit_calculator.py`        | Calculates gross credit from validated USD spend and, when tax-due inputs are provided, plans yearly use subject to the configured cap and carry-forward window.                |
| `document_type_normalizer.py` | Maps listed NCF/e-CF codes to canonical document categories, returning `UNKNOWN` for other values.                                                                              |
| `langgraph_flow.py`           | Defines and connects the Art. 39 graph, including reporting cross-checks, basic expense validation, threshold and participation checks, credit calculation, and summary output. |
| `legal_rag.py`                | Parses marked legal chunks, creates or loads local sentence-transformer embeddings, and retrieves the top matching chunks.                                                      |
| `local_llm.py`                | Sends a prompt to the local Ollama generate endpoint and returns its response.                                                                                                  |
| `models.py`                   | Defines shared expense, evidence, withholding, and validation dataclasses and enums.                                                                                            |
| `participation_validator.py`  | Computes a Dominican-or-resident crew participation ratio using nationality aliases and optional explicit residency flags, with an optional waiver override.                    |
| `report_crosscheck.py`        | Matches NCF rows to 606 and, for foreign-payment types, 609 rows, and records reporting verification facts and rationale.                                                       |
| `threshold_validator.py`      | Converts validated expenses to USD using explicit rates and evaluates the threshold for one project or accumulated projects.                                                    |
| `eligibility/` | Applies the guide JRD-G43 exclusion rules per expense, with Decimal arithmetic, partial admission and a rejection reason with legal reference (`engine.py`, `models.py`, `catalog.py`, closed concept catalog in `concepts.json`). Ported from the Alexa+ hackathon repo. |

## Requirements
- Python: the deterministic engine and its tests were verified with Python 3.13.2 (macOS Intel) and Python 3.14.7 (macOS Apple Silicon). TODO: verify other Python versions.
- The deterministic pipeline needs only `requirements.txt` (LangGraph and requests). It does not need torch.
- The optional legal RAG layer needs `requirements-rag.txt` (`torch`, `sentence-transformers`, `transformers`, `numpy`). It was verified end to end on an Apple Silicon Mac (Python 3.14.7, torch 2.10.0, sentence-transformers 5.2.3). `torch==2.10.0` has no wheels for Intel Macs, so the RAG layer cannot be installed there.
- When `legal_question` is non-empty, the code calls an Ollama server at `http://localhost:11434/api/generate` using model name `mistral`. The `run_local_test.py` sample supplies a legal question, so that run requires this local service plus the RAG dependencies. Verified with Ollama 0.17.0 and `mistral:latest`.
- The RAG code constructs `SentenceTransformer` with model name `all-MiniLM-L6-v2`. On the first run it is downloaded from the Hugging Face Hub (unauthenticated requests produce a warning; setting `HF_TOKEN` is optional). Chunk embeddings are cached under `data/vector_store/`, which is git-ignored.

## Setup

1. Create and activate a virtual environment:

```bash
   python3 -m venv .venv
   source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate`. 2. Install the engine and test dependencies:

```bash
   python -m pip install -r requirements-dev.txt
```

3. Optional, only for the legal RAG layer:

```bash
   python -m pip install -r requirements-rag.txt
```

4. Keep the legal source at `data/legal_sources/article_39_structured.txt`; the RAG node refers to this repository-relative path. For a non-empty legal question, install [Ollama](https://ollama.com), make sure the `mistral` model is available (`ollama list`; if missing, `ollama pull mistral`), and start the server with `ollama serve`.

## Usage

Run the local sample from the repository root:

```bash
python run_local_test.py
```

It invokes the graph with hard-coded sample NCF, 606, and cast/crew rows, then prints deterministic results, any legal analysis and citation output, and the audit trace. The script's sample includes a legal question and therefore requires the local RAG model and Ollama setup described above.

## Input and output
Optional input `production_origin` (`"foreign"` or `"dominican"`): Art. 39 Paragraph III applies the Dominican participation minimum to foreign productions. If omitted, `"foreign"` is assumed and a warning is returned. Any other value raises a `ValueError`.

Optional input `project_context` enables the guide rules:

| field | meaning |
|---|---|
| `approved_budget_total` | total approved budget, the base of the 6% and 3% caps |
| `budget_currency` | currency of that budget |
| `permit_start`, `permit_end` | validity of the filming permit (ISO dates) |

When `project_context` is present, every `ncf_rows` row also needs `date` (ISO), `cost` (without ITBIS), `itbis` and `concept_code` (from `core/eligibility/concepts.json`). Optional: `payment_method` (`Transfer`, `Check` or `Cash`; a missing or unaccepted method is rejected), `account`, `supplier` and `supplier_sirecine`. A missing required field raises a `ValueError` naming the row and the field. These names follow the Alexa+ hackathon connector and have not been checked against a real producer ledger.

`compliance_status` includes:

| field | meaning |
|---|---|
| `eligible_to_request` | threshold met and participation met (or not applicable) |
| `guide_rules_evaluated` | `False` when `project_context` was not provided |
| `gross_credit_usd` | informational credit on validated spend |
| `requestable_credit_usd` | the gross credit if eligible, otherwise 0.0 |
| `blocking_reasons` | plain-language reasons the request is not eligible |
| `excluded_expenses` | each fully excluded expense, with its row number and reasons |
| `partially_admitted_expenses` | expenses admitted only in part, with admitted and excluded amounts |
| `fx_excluded_expenses` | validated expenses left out because no FX rate was provided |
| `warnings` | for example, missing `production_origin` or `project_context` |

`eligibility_summary` (only when the guide rules ran) reports executed, admitted and rejected totals in the budget currency, rejected totals by rule, and the 6% and 3% caps with claimed and excess amounts. `threshold_result`, `participation_result` and `credit_result` also return their detail (counted and excluded expenses, participation issues and counts, yearly credit plan and rationale). All results are plain dictionaries and lists, serializable to JSON.

## Tests

Install `requirements-dev.txt` and run from the repository root:

```bash
python -m pytest
```

The tests do not require torch, a running Ollama server, or network access (the RAG and LLM are replaced by fakes). 36/36 tests passing across 7 test files, covering:
- `tests/test_art39_golden.py`: a compliant case, a threshold failure, and a B17 expense missing its required 609 match.
- `tests/test_crosscheck_amount_and_duplicates.py`: 606 amount mismatch, the 0.01 tolerance, duplicate invoices, and the same NCF from different suppliers.
- `tests/test_summary_detail.py`: requestable credit, blocking reasons, foreign vs. Dominican participation, missing FX rate, per-expense exclusion reasons, and JSON serializability.
- `tests/test_legal_citation_validation.py`: detection of invented citations using a fake retriever and LLM.
- `tests/test_eligibility_port.py`, `tests/test_eligibility_adaptation.py`: the ported engine (cash payment, 6% cap, payments abroad, upstream rejections, skipped NCF rules).
- `tests/test_eligibility_flow.py`: the eligibility node inside the flow (guide rules on and off, cash, fee cap, missing 606, payment abroad with 609, missing field, JSON serializability).

`tests/test_expense_validator.py` is empty and currently covers no tests. The real RAG and Ollama path is not covered by automated tests.

## Legal sources

- `data/legal_sources/article_39_structured.txt` is read by `LegalRAG` in the graph's legal-reasoning node. Its `[CHUNK_ID: ...]` sections and article/section/topic markers are parsed for retrieval and citation checking.
- `data/legal_sources/article_39.txt` contains plain Article 39 text, but repository code does not reference it.
- Both files reproduce Article 39 of Law 108-10 as amended by Law 82-13. Official legislative texts are excluded from copyright protection under Dominican Law 65-00 on Copyright, so they are not covered by the proprietary license below; the chunk markup in the structured file is part of this repository. They are a working transcription, not an official version: check against the Gaceta Oficial text before relying on them.
- The repository contains no real DGCINE files or production data. All RNC, NCF, names and amounts in tests and samples are synthetic.

## Known limitations
- The classification in `concepts.json` (Art. 168 category and ITBIS regime per concept) and the rule references come from the Alexa+ hackathon repo and have not been checked against Decree 370-11.
- B17/E47 payments abroad are admitted when they appear in the 609, skipping the ITBIS matrix. Art. 39 refers to spending made in the Dominican Republic, so this needs confirmation against the guide.
- Without `project_context` the guide rules are not applied: reported spend that the guide would exclude (for example cash payments or fees above the cap) still counts toward the threshold. The result flags this with `guide_rules_evaluated: False`.
- The engine does not check that `cost + itbis` equals `total`.
- Only receipt types B01, E31, B14, E44, B17 and E47 are evaluated by the guide rules.
- The 606 amount tolerance is 0.01 and has not been validated against real DGII data.
- The graph evaluates a single project for the threshold; accumulated multi-project evaluation is not wired into the graph.
- One fiscal year is applied to all expenses; it is not derived from each expense date.
- A failed participation check blocks the whole request for foreign productions. Art. 39 Paragraph III may limit its effect to personnel expenses; this is not modeled and has not been confirmed.
- The waiver flag (Paragraph IV) is a boolean without the reduced percentage or the resolution reference.
- Participation uses a default 25% minimum and nationality-based counting; non-Dominican residency counts only when explicit residency flags are supplied.
- Credit-use planning depends on supplied transferee ISR due amounts. The calculator documents that it does not model the national 10% ISR cap or IT-1/asset-tax offsets in detail.
- Cross-check and threshold money is handled as floating-point numbers; only the eligibility engine uses Decimal.
- The graph uses the structured Article 39 source; it does not connect to DGCINE or DGII services.
- A model answer with no citations is treated as valid by the citation check.

## Roadmap
- TODO: Reject expenses where `cost + itbis` differs from `total`.
- TODO: Confirm the treatment of payments abroad (B17/E47) and the concept classification against the guide and Decree 370-11.
- TODO: Validate the input column names against a real producer ledger.
- TODO: Derive the fiscal year from each expense.
- TODO: Add the waiver details (reduced percentage and resolution reference).
- TODO: Use Decimal for money across the whole pipeline.
- TODO: Share the eligibility engine with the Alexa+ hackathon repo as a package instead of a copy.
- TODO: Evaluate the quality of the local LLM answers beyond a single question (the citation check validates chunk IDs, not whether the content is faithful to the article).
- TODO: Add tests for expense validation or remove the empty test file.

## License

Proprietary. All rights reserved. See LICENSE. This does not apply to the legal texts in `data/legal_sources/` (see Legal sources).
