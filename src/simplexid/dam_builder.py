"""
Membangun kamus Domain-Aspect Mapping dari HoASA (H4.4).

HoASA memberi label kategori per ULASAN, jadi asosiasi kata -> kategori diestimasi
dari kemunculan bersama: P(kategori disebut | kata ada di ulasan), dibandingkan
peluang dasar kategori itu (lift).

Baris HoASA yang teksnya identik dengan kalimat TermA berlabel (dev maupun final)
dikeluarkan agar tidak ada kebocoran ke data evaluasi.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, List, Optional, Set

import pandas as pd

HOASA_CATEGORIES = ["ac", "air_panas", "bau", "general", "kebersihan",
                    "linen", "service", "sunrise_meal", "tv", "wifi"]
NOT_MENTIONED = "neut"
DAM_PATH = Path(__file__).resolve().parents[2] / "resources" / "dam" / "dam_hotel_v1.json"
CONTEXT_MIN_LIFT = 1.5
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass
class BuiltDam:
    opinion_categories: Dict[str, List[str]] = field(default_factory=dict)  # kata -> kandidat, terkuat dulu
    aspect_association: Dict[str, Dict[str, float]] = field(default_factory=dict)  # kata aspek -> kategori -> lift
    default_category: Optional[str] = None  # untuk kata leksikon di luar kamus

    def candidates(self, forms: Iterable[str]) -> List[str]:
        for form in forms:
            if form in self.opinion_categories:
                return self.opinion_categories[form]
        return []

    def similarity(self, category: str, aspect_text: str) -> float:
        words = aspect_text.lower().replace("_", " ").split()
        return max((self.aspect_association.get(w, {}).get(category, 0.0) for w in words), default=0.0)

    def to_json(self) -> dict:
        return {"opinion_categories": self.opinion_categories,
                "aspect_association": self.aspect_association,
                "default_category": self.default_category}


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def usable_reviews(hoasa_df: pd.DataFrame, terma_df: pd.DataFrame) -> pd.DataFrame:
    """Ulasan HoASA berlabel, tanpa yang identik dengan kalimat TermA berlabel (train/validation)."""
    excluded = set(terma_df.loc[terma_df["source_split"].isin(["train", "validation"]), "sentence"].map(normalize_text))
    labeled = hoasa_df[hoasa_df["source_split"].isin(["train", "validation"])]
    return labeled[~labeled["sentence"].map(normalize_text).isin(excluded)]


def association_table(reviews: pd.DataFrame, words: AbstractSet[str]):
    """(jumlah ulasan per kata, jumlah per (kata, kategori), peluang dasar kategori)."""
    word_counts: Counter = Counter()
    joint: Dict[str, Counter] = defaultdict(Counter)
    category_counts: Counter = Counter()
    for row in reviews.itertuples(index=False):
        mentioned = [c for c in HOASA_CATEGORIES if getattr(row, c) != NOT_MENTIONED]
        category_counts.update(mentioned)
        for word in set(TOKEN_RE.findall(str(row.sentence).lower())) & words:
            word_counts[word] += 1
            joint[word].update(mentioned)
    base = {c: category_counts[c] / len(reviews) for c in HOASA_CATEGORIES} if len(reviews) else {}
    return word_counts, joint, base


def build_dam(reviews: pd.DataFrame, opinion_words: AbstractSet[str], aspect_words: AbstractSet[str],
              min_count: int = 3, min_lift: float = 1.2, default_category: Optional[str] = None) -> BuiltDam:
    counts, joint, base = association_table(reviews, opinion_words)
    opinion_categories: Dict[str, List[str]] = {}
    for word, n in counts.items():
        kept = [(c, k) for c, k in joint[word].items()
                if k >= min_count and base[c] > 0 and (k / n) / base[c] >= min_lift]
        if kept:
            opinion_categories[word] = [c for c, _ in sorted(kept, key=lambda x: (-x[1], x[0]))]

    # Konteks hanya berarti jika kata aspek jauh lebih sering muncul dengan kategori itu
    # daripada biasanya; P mentah membuat kata umum ("kamar") mengarah ke kategori apa saja.
    counts, joint, base = association_table(reviews, aspect_words)
    aspect_association = {
        word: {c: round((k / n) / base[c], 4) for c, k in joint[word].items()
               if k >= min_count and base[c] > 0 and (k / n) / base[c] >= CONTEXT_MIN_LIFT}
        for word, n in counts.items()
    }
    return BuiltDam(opinion_categories, {w: a for w, a in aspect_association.items() if a}, default_category)


def write_dam(dam: BuiltDam, path: Path = DAM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dam.to_json(), ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def load_dam(path: Path = DAM_PATH) -> BuiltDam:
    data = json.loads(path.read_text(encoding="utf-8"))
    return BuiltDam(data["opinion_categories"], data["aspect_association"], data["default_category"])
