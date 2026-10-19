"""Pemuat leksikon sentimen (word -> skor -1..+1) dari resources/lexicon/."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_DIR = PROJECT_ROOT / "resources" / "lexicon"
DEFAULT_LEXICON = "seed_lexicon_v0.tsv"


def load_lexicon(name: str = DEFAULT_LEXICON) -> Dict[str, float]:
    path = Path(name) if Path(name).is_absolute() else LEXICON_DIR / name
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        lexicon = {row["word"].strip().lower(): float(row["score"]) for row in reader}
    for word, score in lexicon.items():
        if not -1.0 <= score <= 1.0:
            raise ValueError(f"Skor leksikon di luar [-1, 1]: {word}={score}")
    return lexicon
