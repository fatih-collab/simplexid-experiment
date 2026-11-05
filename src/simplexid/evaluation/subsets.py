"""
Subset uji berbasis GOLD span (H4.0), tidak bergantung pada hasil parser, agar
efek R3/R4/SC tidak hanya diukur pada kalimat yang kebetulan di-parse benar.

- COORD_OPINION : dua pasangan berbagi aspek, kedua span opini hanya dipisah konektor.
- COORD_ASPECT  : dua pasangan berbagi opini, kedua span aspek hanya dipisah konektor.
- ASYM_MOD      : COORD_OPINION dengan negasi/intensifier hanya pada salah satu opini.

Ini tag otomatis; diverifikasi manusia di H5 sebelum dipakai untuk klaim final.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Set

from ..cues import ALL_CUES
from .gold_loader import EvalGold, GoldPairSpan

CONNECTORS = {"dan", "serta", "atau", "&", "+", ",", "/", "tapi", "tetapi", "namun", "tp", "dgn"}
COORD_OPINION = "COORD_OPINION"
COORD_ASPECT = "COORD_ASPECT"
ASYM_MOD = "ASYM_MOD"


def _joined(tokens: List[str], first, second) -> bool:
    left, right = sorted((first, second))
    between = [t.lower() for t in tokens[left[1] + 1:right[0]]]
    return 1 <= len(between) <= 2 and all(t in CONNECTORS for t in between)


def _cues(tokens: List[str], span) -> Set[str]:
    return {t.lower() for t in tokens[span[0]:span[1] + 1]} & ALL_CUES


def pair_tags(tokens: List[str], pairs: List[GoldPairSpan]) -> Dict[str, Set[str]]:
    tags: Dict[str, Set[str]] = {p.pair_id: set() for p in pairs}
    for p, q in combinations(pairs, 2):
        if p.aspect_span == q.aspect_span and p.opinion_span != q.opinion_span:
            if _joined(tokens, p.opinion_span, q.opinion_span):
                labels = {COORD_OPINION}
                if _cues(tokens, p.opinion_span) != _cues(tokens, q.opinion_span):
                    labels.add(ASYM_MOD)
                tags[p.pair_id] |= labels
                tags[q.pair_id] |= labels
        elif p.opinion_span == q.opinion_span and p.aspect_span != q.aspect_span:
            if _joined(tokens, p.aspect_span, q.aspect_span):
                tags[p.pair_id].add(COORD_ASPECT)
                tags[q.pair_id].add(COORD_ASPECT)
    return tags


def gold_tags(gold: EvalGold) -> Dict[str, Set[str]]:
    """{pair_id: tag} untuk seluruh universe gold."""
    out: Dict[str, Set[str]] = {}
    for sid in gold.sentence_ids:
        out.update(pair_tags(gold.tokens[sid], gold.pairs[sid]))
    return out


def sentences_with_tag(gold: EvalGold, tags: Dict[str, Set[str]], tag: str) -> List[str]:
    return [sid for sid in gold.sentence_ids if any(tag in tags[p.pair_id] for p in gold.pairs[sid])]
