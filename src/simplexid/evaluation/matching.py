"""
Format prediksi dan pencocokan prediksi <-> gold (H3.3).

Tiga tingkat pencocokan pasangan aspek-opini:
- strict  : span aspek DAN span opini sama persis dengan gold.
- head    : head token prediksi berada di dalam span gold (kedua sisi). METRIK UTAMA,
            karena rules memilih head token sedangkan gold TermA berupa frasa.
- overlap : span prediksi beririsan dengan span gold (kedua sisi).

Pencocokan one-to-one (maximum bipartite matching): satu prediksi hanya boleh
membenarkan satu gold, dan sebaliknya.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LEVELS = ("strict", "head", "overlap")

Span = Tuple[int, int]


@dataclass
class PredPair:
    aspect_span: Optional[Span]     # None untuk aspek implisit (R8)
    aspect_head: Optional[int]
    opinion_span: Span
    opinion_head: int
    rule: str
    score: float = 0.0
    implicit_category: Optional[str] = None

    @property
    def is_implicit(self) -> bool:
        return self.aspect_span is None

    def to_dict(self) -> dict:
        return {
            "aspect_span": list(self.aspect_span) if self.aspect_span else None,
            "aspect_head": self.aspect_head,
            "opinion_span": list(self.opinion_span),
            "opinion_head": self.opinion_head,
            "rule": self.rule,
            "score": round(float(self.score), 6),
            "implicit_category": self.implicit_category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PredPair":
        return cls(
            aspect_span=tuple(d["aspect_span"]) if d["aspect_span"] else None,
            aspect_head=d["aspect_head"],
            opinion_span=tuple(d["opinion_span"]),
            opinion_head=d["opinion_head"],
            rule=d["rule"],
            score=d["score"],
            implicit_category=d["implicit_category"],
        )


def from_extraction_results(results) -> List[PredPair]:
    """Konversi pipeline.ExtractionResult -> PredPair (span = head token saja untuk saat ini)."""
    out = []
    for r in results:
        p = r.pair
        explicit = p.aspect_idx >= 0
        out.append(PredPair(
            aspect_span=(p.aspect_idx, p.aspect_idx) if explicit else None,
            aspect_head=p.aspect_idx if explicit else None,
            opinion_span=(p.opinion_idx, p.opinion_idx),
            opinion_head=p.opinion_idx,
            rule=p.rule,
            score=r.factors.score_final,
            implicit_category=None if explicit else p.aspect_text,
        ))
    return out


def _inside(i: int, span: Span) -> bool:
    return span[0] <= i <= span[1]


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def span_match(level: str, pred_span: Span, pred_head: int, gold_span: Span) -> bool:
    if level == "strict":
        return tuple(pred_span) == tuple(gold_span)
    if level == "head":
        return _inside(pred_head, gold_span)
    if level == "overlap":
        return _overlaps(pred_span, gold_span)
    raise ValueError(f"level tidak dikenal: {level}")


def pair_match(level: str, pred: PredPair, gold_aspect: Span, gold_opinion: Span) -> bool:
    if pred.is_implicit:
        return False
    return (span_match(level, pred.aspect_span, pred.aspect_head, gold_aspect)
            and span_match(level, pred.opinion_span, pred.opinion_head, gold_opinion))


def max_bipartite(n_left: int, n_right: int, edge: Callable[[int, int], bool]) -> Dict[int, int]:
    """Maximum matching (augmenting path, Kuhn). Mengembalikan {left_index: right_index}."""
    adj = [[j for j in range(n_right) if edge(i, j)] for i in range(n_left)]
    match_right: Dict[int, int] = {}

    def try_assign(i: int, seen: set) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_right or try_assign(match_right[j], seen):
                match_right[j] = i
                return True
        return False

    for i in range(n_left):
        try_assign(i, set())
    return {i: j for j, i in match_right.items()}


def match_pairs(level: str, preds: Sequence[PredPair], golds: Sequence) -> Dict[int, int]:
    """{index_pred: index_gold} untuk prediksi eksplisit yang cocok dengan gold (GoldPairSpan)."""
    return max_bipartite(
        len(preds), len(golds),
        lambda i, j: pair_match(level, preds[i], golds[j].aspect_span, golds[j].opinion_span),
    )
