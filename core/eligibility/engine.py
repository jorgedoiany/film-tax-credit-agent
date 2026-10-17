"""Eligibility rules, in order:

 0. Technical validity (amount > 0, known concept, FX rate available)
 1. NCF present
 2. Duplicate NCF (same NCF + same RNC): the first one is kept
 3. Payment method (cash or missing is rejected)
 4. ITBIS matrix
 5. Art. 168, full exclusion (c, d, fixed assets, alcohol, tobacco)
 6. Art. 168, proportional excess (a: 6%, b: 3%) over what is still admitted

The USD 500,000 threshold (rule 7) is evaluated by ThresholdEngine through
build_threshold_rows(); this module does not reimplement it.

Rules 1-4 are DGCINE requirements (article to be confirmed).
Independent prototype. Not an official DGCINE determination.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .catalog import Concept, load_catalog
from .models import (
    AccountSummary, CapInfo, EligibilityReport, ExpenseDecision,
    ExpenseRecord, FXConverter, ProjectContext, Rejection,
)

CENT = Decimal("0.01")

STANDARD_TYPES = {"B01", "E31"}
SPECIAL_TYPES = {"B14", "E44"}

REF_DGCINE = "DGCINE requirement (article to be confirmed)"
REF_188 = "Decree 370-11, Art. 188"
REF_189 = "Decree 370-11, Art. 189"
REF_191 = "Decree 370-11, Art. 191"
REF_PROTOTYPE = "Prototype limitation"

EXCESS_RULES = {"ART168_A_EXCESS", "ART168_B_EXCESS"}

# Art. 168 category -> (rule, message, legal reference)
TOTAL_EXCLUSIONS = {
    "financial": ("ART168_C_FINANCIAL",
                  "Excluded: financial charges and performance bonds/guarantees.",
                  "Art. 168 (c)"),
    "marketing": ("ART168_D_MARKETING",
                  "Excluded: marketing and distribution costs of the work.",
                  "Art. 168 (d)"),
    "fixed_assets": ("ART168_FIXED_ASSETS",
                     "Excluded: purchase of fixed assets.",
                     "Art. 168, final paragraph"),
    "alcohol": ("ART168_ALCOHOL",
                "Excluded: purchase of alcoholic beverages.",
                "Art. 168, final paragraph"),
    "tobacco": ("ART168_TOBACCO",
                "Excluded: purchase of cigarettes.",
                "Art. 168, final paragraph"),
}


@dataclass(frozen=True)
class EligibilityParams:
    """Documented parameters. Change them here, in one place."""
    itbis_rate: float = 0.18
    itbis_tolerance: float = 1.00          # +/- 1 peso for rounding
    producer_fee_cap: float = 0.06         # Art. 168 (a)
    development_cap: float = 0.03          # Art. 168 (b)
    allowed_payment_methods: Tuple[str, ...] = ("transfer", "check")


def _d(x) -> Decimal:
    return Decimal(str(x))


def _q(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def _amount(e) -> Decimal:
    """Amount of an expense for every total: cost plus ITBIS."""
    return _d(e.total) if e.total is not None else Decimal(0)


def _rej(rule: str, msg: str, ref: str, e: ExpenseRecord) -> Rejection:
    return Rejection(rule, msg, ref, float(_q(_amount(e))))


def _ncf_type(ncf: str) -> str:
    return ncf.strip().upper()[:3]


# ---------------------------------------------------------------- rules

def _check_technical(e, concept, fx) -> List[Rejection]:
    out: List[Rejection] = []
    if e.cost is None or e.cost <= 0:
        out.append(_rej("INVALID_AMOUNT",
                        "Non-positive amount (credit/debit notes are not evaluated in this prototype).",
                        REF_PROTOTYPE, e))
    if concept is None:
        out.append(_rej("UNKNOWN_CONCEPT",
                        f"Concept '{e.concept_code}' is not in the catalog.",
                        REF_PROTOTYPE, e))
    if fx.convert_to_usd(1.0, e.currency) is None:
        out.append(_rej("FX_MISSING",
                        f"No USD conversion rate for currency {e.currency}.",
                        REF_PROTOTYPE, e))
    return out


def _check_ncf_present(e) -> List[Rejection]:
    if not e.ncf or not str(e.ncf).strip():
        return [_rej("NCF_MISSING", "The expense has no NCF.", REF_DGCINE, e)]
    return []


def _check_duplicate(e, seen: Set[Tuple[str, str]]) -> List[Rejection]:
    if not e.ncf or not str(e.ncf).strip():
        return []
    key = (str(e.ncf).strip().upper(), "".join(ch for ch in str(e.rnc) if ch.isdigit()))
    if key in seen:
        return [_rej("NCF_DUPLICATE",
                     "Duplicate NCF: another expense already has the same NCF and the same RNC.",
                     REF_DGCINE, e)]
    seen.add(key)
    return []


def _check_payment(e, params) -> List[Rejection]:
    method = (e.payment_method or "").strip().lower()
    if method == "cash":
        return [_rej("PAYMENT_CASH",
                     "Cash payment: only checks or bank transfers are accepted.",
                     REF_DGCINE, e)]
    if method not in params.allowed_payment_methods:
        return [_rej("PAYMENT_UNREGISTERED",
                     "Payment method missing or not accepted.",
                     REF_DGCINE, e)]
    return []


def _check_itbis(e, concept: Concept, project: ProjectContext, params) -> List[Rejection]:
    if not e.ncf or not str(e.ncf).strip():
        return []  # already rejected by NCF_MISSING
    ncf_type = _ncf_type(e.ncf)
    itbis = _d(e.itbis)
    cost = _d(e.cost)
    expected = cost * _d(params.itbis_rate)
    tol = _d(params.itbis_tolerance)
    is_zero = abs(itbis) < _d("0.005")
    regime = concept.itbis_regime

    if ncf_type in STANDARD_TYPES:
        if abs(itbis - expected) <= tol:
            return []
        if is_zero:
            if regime == "dgii_exempt":
                return []
            if regime == "art189_cine":
                return [_rej("ITBIS_ZERO_NO_EXEMPTION",
                             "Zero ITBIS without a valid exemption: this is an Art. 189 service, "
                             "so an NCF type B14/E44 should have been used to invoice it without ITBIS.",
                             f"{REF_DGCINE}; {REF_189}", e)]
            return [_rej("ITBIS_ZERO_NO_EXEMPTION",
                         "Zero ITBIS with no exemption applicable to this concept; "
                         "it must carry 18% ITBIS.",
                         REF_DGCINE, e)]
        return [_rej("ITBIS_MISMATCH",
                     f"ITBIS differs from {params.itbis_rate * 100:g}% of the cost.",
                     REF_DGCINE, e)]

    if ncf_type in SPECIAL_TYPES:
        out: List[Rejection] = []
        if not is_zero:
            out.append(_rej("B14_ITBIS_NOT_ZERO",
                            "B14/E44 receipt with non-zero ITBIS: inconsistent.",
                            REF_DGCINE, e))
        if regime == "art191_excluded":
            out.append(_rej("B14_NOT_APPLICABLE",
                            "Exemption not applicable: the concept is excluded by Art. 191.",
                            REF_191, e))
        elif regime != "art189_cine":
            out.append(_rej("B14_NOT_APPLICABLE",
                            "Exemption not applicable: the concept is not listed in Art. 189.",
                            REF_189, e))
        else:
            if not e.supplier_sirecine:
                out.append(_rej("B14_NOT_SIRECINE",
                                "Supplier is not registered in the SIRECINE tax registry of "
                                "cinematographic agents and suppliers.",
                                f"{REF_188} (a)", e))
            if not (project.permit_start <= e.date <= project.permit_end):
                out.append(_rej("B14_PERMIT_INVALID",
                                "Expense date falls outside the validity of the Single Filming Permit.",
                                f"{REF_188} (b)", e))
        return out

    return [_rej("NCF_TYPE_UNSUPPORTED",
                 f"Receipt type {ncf_type} is not evaluated by this prototype "
                 "(only B01, E31, B14, E44).",
                 REF_PROTOTYPE, e)]


def _check_total_exclusion(e, concept: Concept) -> List[Rejection]:
    hit = TOTAL_EXCLUSIONS.get(concept.art168)
    if hit is None:
        return []
    rule, msg, ref = hit
    return [_rej(rule, msg, ref, e)]


# ---------------------------------------------------------------- engine

def evaluate_project(
    expenses: Sequence[ExpenseRecord],
    project: ProjectContext,
    fx: FXConverter,
    *,
    catalog: Optional[Dict[str, Concept]] = None,
    params: Optional[EligibilityParams] = None,
) -> EligibilityReport:
    catalog = catalog or load_catalog()
    params = params or EligibilityParams()

    rate_b = fx.convert_to_usd(1.0, project.budget_currency)
    budget_usd = fx.convert_to_usd(project.approved_budget_total, project.budget_currency)
    if not rate_b or budget_usd is None:
        raise ValueError(f"No USD rate for the budget currency ({project.budget_currency}).")
    rate_b_d = _d(rate_b)

    ordered = sorted(expenses, key=lambda x: (x.date, x.expense_id))
    seen: Set[Tuple[str, str]] = set()
    rejs: Dict[str, List[Rejection]] = {}

    for e in ordered:
        concept = catalog.get(e.concept_code)
        r: List[Rejection] = []
        r += _check_technical(e, concept, fx)
        r += _check_ncf_present(e)
        r += _check_duplicate(e, seen)
        r += _check_payment(e, params)
        if concept is not None:
            r += _check_itbis(e, concept, project, params)
            r += _check_total_exclusion(e, concept)
        rejs[e.expense_id] = r

    # Rule 6: proportional excess over what is still admitted
    caps: Dict[str, CapInfo] = {}
    specs = (
        ("producer_fees", params.producer_fee_cap, "ART168_A_EXCESS", "Art. 168 (a)", "producer fees"),
        ("development", params.development_cap, "ART168_B_EXCESS", "Art. 168 (b)", "development expenses"),
    )
    for cat, pct, rule, ref, label in specs:
        alive = [e for e in ordered if not rejs[e.expense_id] and catalog[e.concept_code].art168 == cat]
        claimed = sum((_d(fx.convert_to_usd(float(_amount(e)), e.currency)) for e in alive), Decimal(0))
        cap = _d(pct) * _d(budget_usd)
        excess = max(Decimal(0), claimed - cap)
        ratio = (excess / claimed) if claimed > 0 else Decimal(0)
        caps[cat] = CapInfo(
            cap=float(_q(cap / rate_b_d)),
            claimed=float(_q(claimed / rate_b_d)),
            excess=float(_q(excess / rate_b_d)),
        )
        if ratio > 0:
            for e in alive:
                amount = _q(_amount(e) * ratio)
                rejs[e.expense_id].append(Rejection(
                    rule,
                    f"Exceeds {pct * 100:g}% of the total budget for {label}: "
                    f"only the excess portion is excluded.",
                    ref, float(amount)))

    # Decisions
    decisions: List[ExpenseDecision] = []
    for e in ordered:
        r = rejs[e.expense_id]
        cost = _q(_amount(e))
        full = [x for x in r if x.rule not in EXCESS_RULES]
        if full:
            status, admitted = "rejected", Decimal(0)
        elif r:
            status = "partial"
            admitted = cost - sum((_d(x.amount) for x in r), Decimal(0))
        else:
            status, admitted = "admitted", cost
        decisions.append(ExpenseDecision(
            e.expense_id, status, float(cost), float(_q(admitted)), tuple(r)))

    # Totals in the budget currency
    def to_budget(amount: Decimal, currency: str) -> Optional[Decimal]:
        usd = fx.convert_to_usd(float(amount), currency)
        return None if usd is None else _d(usd) / rate_b_d

    by_acc: Dict[str, List[Decimal]] = {}
    by_rule: Dict[str, Decimal] = {}
    unconverted: List[str] = []
    executed_t = admitted_t = Decimal(0)
    for e, d in zip(ordered, decisions):
        ex = to_budget(_d(d.cost), e.currency)
        ad = to_budget(_d(d.admitted_cost), e.currency)
        if ex is None or ad is None:
            unconverted.append(e.expense_id)
            continue
        acc = by_acc.setdefault(e.account, [Decimal(0), Decimal(0)])
        acc[0] += ex
        acc[1] += ad
        executed_t += ex
        admitted_t += ad
        if d.primary_rule and ex != ad:
            by_rule[d.primary_rule] = by_rule.get(d.primary_rule, Decimal(0)) + (ex - ad)

    return EligibilityReport(
        project_id=project.project_id,
        currency=project.budget_currency,
        decisions=tuple(decisions),
        by_account=tuple(AccountSummary(a, float(_q(v[0])), float(_q(v[1])))
                         for a, v in sorted(by_acc.items())),
        executed_total=float(_q(executed_t)),
        admitted_total=float(_q(admitted_t)),
        rejected_total=float(_q(executed_t - admitted_t)),
        rejected_by_primary_rule={k: float(_q(v)) for k, v in by_rule.items()},
        caps=caps,
        unconverted_expense_ids=tuple(unconverted),
    )


def build_threshold_rows(
    report: EligibilityReport,
    expenses: Sequence[ExpenseRecord],
    row_cls=None,
):
    """Turn admitted expenses into rows for ThresholdEngine.

    Only admitted or partially admitted rows are passed on (validated=True,
    validated_amount = admitted cost). Rejected rows are not sent, so
    ThresholdEngine never writes its own reason ("not validated by Expense
    Validation Engine"), which would be misleading.
    """
    if row_cls is None:
        from core.threshold_validator import ValidatedExpenseRow as row_cls  # type: ignore
    by_id = {e.expense_id: e for e in expenses}
    rows = []
    for d in report.decisions:
        if d.status == "rejected":
            continue
        e = by_id[d.expense_id]
        rows.append(row_cls(
            expense_id=e.expense_id,
            applicant_id=e.applicant_id,
            project_id=e.project_id,
            fiscal_year=e.fiscal_year,
            validated_amount=d.admitted_cost,
            currency=e.currency,
            validated=True,
        ))
    return rows