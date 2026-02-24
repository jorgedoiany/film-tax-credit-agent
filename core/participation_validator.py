"""
core/participation_validator.py

Participation Engine for Art. 39 (Transferable Tax Credit - DR Film Incentives).

Legal concept:
- "Participation of Dominican personnel" refers to people working on the project
  as crew/cast/production staff (positions within the production), not suppliers.
- The Dominican Film Commission requires a "Cast & Crew List" spreadsheet that
  enumerates individuals who worked on the project, which is the cleanest and most
  defensible input for participation computation.

Input document (Cast & Crew List) columns (provided by user):
- Position/Job Title
- Name
- SIRECINE (code or number)
- Tax ID (code or number)
- ID Number/Passport
- Nationality
- Department
- Phone
- Email

MVP constraint:
- The Cast & Crew List includes NATIONALITY but does not explicitly encode
  residency status for non-Dominicans.
- Therefore, in the MVP we compute the Dominican participation ratio using
  nationality only.
- The design includes an override for a DGCINE waiver/resolution (Paragraph IV).

Current rule:
- For foreign productions, minimum Dominican/resident participation is 25%
  after the 6th year of the law's effect.
- We treat this as the default (configurable) threshold.

This engine is deterministic and auditable:
- It never guesses nationality.
- It never infers "resident" unless an explicit flag is provided.
- It produces a rationale trace suitable for compliance reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class CrewRow:
    """
    Canonical representation of one line in the Cast & Crew List.

    We keep the structure close to the spreadsheet to reduce mapping errors.
    For MVP, these fields are sufficient to compute participation deterministically.
    """
    position_job_title: str
    name: str
    sirecine: Optional[str]
    tax_id: Optional[str]
    id_number_passport: Optional[str]
    nationality: str
    department: Optional[str]
    phone: Optional[str]
    email: Optional[str]


@dataclass(frozen=True)
class ParticipationPolicy:
    """
    Configurable participation threshold policy.

    - minimum_ratio: default 0.25 (25%)
    - dominican_nationality_aliases: values treated as Dominican nationality
      after normalization.

    Why aliases?
    - Real spreadsheets can contain "Dominican", "Dominicana", "RD",
      "Dominican Republic", etc.
    - We normalize and match against a controlled alias set.
    """
    minimum_ratio: float = 0.25
    dominican_nationality_aliases: Tuple[str, ...] = (
        "DOMINICAN",
        "DOMINICANA",
        "DOMINICANO",
        "DOMINICAN REPUBLIC",
        "REP. DOMINICANA",
        "REPUBLICA DOMINICANA",
        "R.D.",
        "RD",
        "D.R.",
        "DR",
    )


@dataclass(frozen=True)
class ParticipationIssue:
    """
    A structured issue for compliance auditing.

    severity:
      - "BLOCKER": prevents participation computation or compliance
      - "WARNING": requires attention but does not block computation
    """
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ParticipationResult:
    """
    Result of the participation evaluation.
    """
    compliant: bool
    total_personnel_count: int
    dominican_or_resident_count: int
    participation_ratio: float
    minimum_required_ratio: float
    waiver_override_used: bool
    issues: List[ParticipationIssue]
    rationale: List[str]


# ============================================================
# Helpers
# ============================================================

def _norm_str(value: Any) -> str:
    """
    Normalize for stable comparisons.

    - Strip whitespace
    - Uppercase
    - Handle None safely
    """
    if value is None:
        return ""
    return str(value).strip().upper()


def _is_dominican_nationality(nationality: str, policy: ParticipationPolicy) -> bool:
    """
    Determine whether a nationality value should be treated as Dominican.

    We do NOT infer residency here. Residency must be provided explicitly.
    """
    n = _norm_str(nationality)
    if not n:
        return False

    # Exact match against known aliases (after normalization)
    return n in {a.upper() for a in policy.dominican_nationality_aliases}


# ============================================================
# Engine
# ============================================================

class ParticipationEngine:
    """
    ParticipationEngine computes the Dominican personnel participation ratio.

    Inputs:
    - crew_rows: list of CrewRow extracted from Cast & Crew List
    - waiver_override_present: True if there is a DGCINE resolution reducing the requirement
      (Paragraph IV of Art. 39). If True, we treat the participation rule as satisfied.

    Optional advanced input:
    - resident_flags: mapping from an identity key to explicit residency status.
      This allows counting non-Dominicans with Dominican residency when the production
      has documentary evidence. For MVP, this mapping is optional and can be empty.

    Deterministic identity key strategy:
    - Prefer ID Number/Passport if present
    - Otherwise, fall back to SIRECINE code
    - Otherwise, fall back to (Name + Email) composite
    """
    def __init__(self, policy: Optional[ParticipationPolicy] = None) -> None:
        self._policy = policy or ParticipationPolicy()

    def evaluate(
        self,
        crew_rows: List[CrewRow],
        *,
        waiver_override_present: bool = False,
        resident_flags: Optional[Dict[str, bool]] = None,
    ) -> ParticipationResult:
        issues: List[ParticipationIssue] = []
        rationale: List[str] = []

        resident_flags = resident_flags or {}

        # --------------------------------------------------------
        # 0) Basic data quality checks
        # --------------------------------------------------------
        if not crew_rows:
            issues.append(self._blocker(
                "CREW_LIST_EMPTY",
                "Cast & Crew List is empty; cannot compute participation."
            ))
            rationale.append("Rejected: no personnel data available.")
            return ParticipationResult(
                compliant=False,
                total_personnel_count=0,
                dominican_or_resident_count=0,
                participation_ratio=0.0,
                minimum_required_ratio=self._policy.minimum_ratio,
                waiver_override_used=False,
                issues=issues,
                rationale=rationale,
            )

        rationale.append(f"Total Cast & Crew rows received: {len(crew_rows)}.")
        rationale.append(
            "Participation is computed from individuals in the Cast & Crew List "
            "(crew/cast/production staff), not suppliers/vendors."
        )

        # --------------------------------------------------------
        # 1) Compute counts
        # --------------------------------------------------------
        total_count = 0
        dom_or_res_count = 0
        missing_nationality = 0

        for row in crew_rows:
            total_count += 1

            nat = _norm_str(row.nationality)
            if not nat:
                missing_nationality += 1
                continue

            if _is_dominican_nationality(nat, self._policy):
                dom_or_res_count += 1
                continue

            # Residency is not inferable from nationality alone.
            # If explicit residency evidence exists, it can be passed in resident_flags.
            identity_key = self._build_identity_key(row)
            if identity_key and resident_flags.get(identity_key) is True:
                dom_or_res_count += 1

        if missing_nationality > 0:
            issues.append(self._warning(
                "MISSING_NATIONALITY_VALUES",
                f"{missing_nationality} rows have missing nationality; they cannot be counted as Dominican/resident."
            ))
            rationale.append(
                f"Warning: {missing_nationality} rows missing nationality. "
                "They remain in the denominator but not in the Dominican/resident numerator."
            )

        # Guard against division by zero (should not happen because crew_rows not empty)
        ratio = dom_or_res_count / total_count if total_count > 0 else 0.0

        rationale.append(f"Dominican/resident personnel count: {dom_or_res_count}.")
        rationale.append(f"Total personnel count (denominator): {total_count}.")
        rationale.append(f"Participation ratio: {ratio:.4f}.")
        rationale.append(f"Minimum required ratio (policy): {self._policy.minimum_ratio:.2%}.")

        # --------------------------------------------------------
        # 2) Apply waiver override (Paragraph IV)
        # --------------------------------------------------------
        if waiver_override_present:
            rationale.append(
                "Waiver override present: DGCINE resolution reduces the minimum Dominican participation requirement "
                "(Art. 39 Paragraph IV). Participation condition treated as satisfied."
            )
            return ParticipationResult(
                compliant=True,
                total_personnel_count=total_count,
                dominican_or_resident_count=dom_or_res_count,
                participation_ratio=round(ratio, 6),
                minimum_required_ratio=self._policy.minimum_ratio,
                waiver_override_used=True,
                issues=issues,
                rationale=rationale,
            )

        # --------------------------------------------------------
        # 3) Determine compliance
        # --------------------------------------------------------
        compliant = ratio >= self._policy.minimum_ratio
        if compliant:
            rationale.append("Result: participation requirement met.")
        else:
            shortfall = self._policy.minimum_ratio - ratio
            issues.append(self._blocker(
                "PARTICIPATION_BELOW_MINIMUM",
                f"Participation ratio below minimum requirement by {shortfall:.4f}."
            ))
            rationale.append("Result: participation requirement NOT met (no waiver override provided).")

        return ParticipationResult(
            compliant=compliant,
            total_personnel_count=total_count,
            dominican_or_resident_count=dom_or_res_count,
            participation_ratio=round(ratio, 6),
            minimum_required_ratio=self._policy.minimum_ratio,
            waiver_override_used=False,
            issues=issues,
            rationale=rationale,
        )

    # --------------------------------------------------------
    # Identity key helper
    # --------------------------------------------------------

    @staticmethod
    def _build_identity_key(row: CrewRow) -> str:
        """
        Build a deterministic identity key for optional residency lookups.

        Priority:
        1) ID Number/Passport
        2) SIRECINE
        3) Name + Email composite

        Why not just email?
        - Some spreadsheets can have missing emails.
        - ID/Passport is more stable when available.
        """
        id_doc = _norm_str(row.id_number_passport)
        if id_doc:
            return f"ID:{id_doc}"

        sirecine = _norm_str(row.sirecine)
        if sirecine:
            return f"SIRECINE:{sirecine}"

        name = _norm_str(row.name)
        email = _norm_str(row.email)
        if name and email:
            return f"NAME_EMAIL:{name}|{email}"

        # If we cannot build a stable key, return empty string.
        return ""

    # --------------------------------------------------------
    # Issue helpers
    # --------------------------------------------------------

    @staticmethod
    def _blocker(code: str, message: str) -> ParticipationIssue:
        return ParticipationIssue(code=code, severity="BLOCKER", message=message)

    @staticmethod
    def _warning(code: str, message: str) -> ParticipationIssue:
        return ParticipationIssue(code=code, severity="WARNING", message=message)
    