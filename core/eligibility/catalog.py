from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ART168_VALUES = {
    "none", "producer_fees", "development", "financial",
    "marketing", "fixed_assets", "alcohol", "tobacco",
}
ITBIS_REGIMES = {"standard", "dgii_exempt", "art189_cine", "art191_excluded"}

DEFAULT_PATH = Path(__file__).with_name("concepts.json")


@dataclass(frozen=True)
class Concept:
    code: str
    label: str
    art168: str
    itbis_regime: str


def load_catalog(path: Optional[Path] = None) -> Dict[str, Concept]:
    data = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    catalog: Dict[str, Concept] = {}
    for item in data["concepts"]:
        c = Concept(item["code"], item["label"], item["art168"], item["itbis_regime"])
        if c.art168 not in ART168_VALUES:
            raise ValueError(f"Concept {c.code}: invalid art168 value ({c.art168})")
        if c.itbis_regime not in ITBIS_REGIMES:
            raise ValueError(f"Concept {c.code}: invalid ITBIS regime ({c.itbis_regime})")
        if c.code in catalog:
            raise ValueError(f"Duplicate concept: {c.code}")
        catalog[c.code] = c
    return catalog