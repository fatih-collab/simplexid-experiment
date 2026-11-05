"""Daftar kata negasi/amplifier/downtoner terpadu (resources/cues/), dipakai scoring & tagging."""

from __future__ import annotations

from pathlib import Path

CUE_DIR = Path(__file__).resolve().parents[2] / "resources" / "cues"


def _load(name: str) -> frozenset:
    return frozenset(
        w.strip().lower() for w in (CUE_DIR / name).read_text(encoding="utf-8").splitlines() if w.strip()
    )


NEGATION_WORDS = _load("negation.txt")
AMPLIFIER_WORDS = _load("amplifier.txt")
DOWNTONER_WORDS = _load("downtoner.txt")
INTENSITY_WORDS = AMPLIFIER_WORDS | DOWNTONER_WORDS
ALL_CUES = NEGATION_WORDS | INTENSITY_WORDS
