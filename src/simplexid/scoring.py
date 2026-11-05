"""
Perhitungan skor sentimen: Neg x Int x Lexicon.

Dua dimensi yang diuji ablation:
- distance_weighted=False (statis): hanya modifier yang menempel LANGSUNG pada
  opini, efeknya tetap (setara SO-CAL/Polanyi-Zaenen).
- distance_weighted=True (DW): modifier dicari sampai jarak dependensi
  NEG_DISTANCE_THRESHOLD lewat rantai modifier, dan efek intensifier meluruh
  terhadap jarak.

- Scope consistency: pasangan hasil R4 berbagi aspek tetapi punya token opini
  sendiri. Dengan SC faktor dihitung dari opininya sendiri; tanpa SC faktor
  DIWARISI dari opini sumber (modifier_source_idx). Pasangan R3 berbagi token
  opini yang sama dengan sumbernya, jadi SC hanya berpengaruh pada R4.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config
from .cues import ALL_CUES, AMPLIFIER_WORDS, INTENSITY_WORDS, NEGATION_WORDS
from .parser import ParsedSentence, Token

# DW hanya menelusuri rantai modifier ("tidak terlalu bersih"). Relasi lain (amod, compound,
# obj, conj, ...) terbukti di dev menyeberang ke frasa lain ("bau tidak sedap" -> negasi keliru).
TRAVERSED_RELATIONS = {"advmod"}
# Naik ke head hanya lewat komplemen verbal ("tidak bisa dikunci").
UPWARD_RELATIONS = {"xcomp"}


@dataclass
class SentimentFactors:
    neg: float
    intensifier: float
    swn_score: float
    distance_neg: Optional[int] = None
    distance_int: Optional[int] = None

    @property
    def score_final(self) -> float:
        return self.neg * self.intensifier * self.swn_score


def get_swn_score(opinion_lemma: str, swn_lexicon: dict) -> float:
    """Skor leksikon sebuah kata (-1..+1); 0.0 jika tidak ada."""
    return swn_lexicon.get(opinion_lemma.lower(), 0.0)


def token_score(token: Token, swn_lexicon: dict) -> float:
    """Cari bentuk permukaan dulu (lemma UDPipe kadang keliru), lalu lemma."""
    for form in (token.text.lower(), token.lemma.lower()):
        if form in swn_lexicon:
            return swn_lexicon[form]
    return 0.0


def _cue_form(token: Token) -> Optional[str]:
    for form in (token.text.lower(), token.lemma.lower()):
        if form in ALL_CUES:
            return form
    return None


def direct_modifiers(sent: ParsedSentence, idx: int) -> List[Tuple[Token, int]]:
    return [(t, 1) for t in sent.tokens if t.head_idx == idx and t.idx != idx and t.dep == "advmod"]


def reachable_modifiers(sent: ParsedSentence, idx: int, max_distance: int) -> List[Tuple[Token, int]]:
    """Token kata-isyarat dalam jarak <= max_distance dari idx, dengan jaraknya."""
    children = {t.idx: [] for t in sent.tokens}
    for t in sent.tokens:
        if t.head_idx != t.idx:
            children[t.head_idx].append(t)

    found, seen = [], {idx}
    queue = deque([(idx, 0)])
    while queue:
        cur, d = queue.popleft()
        if d == max_distance:
            continue
        neighbours = [c for c in children[cur] if c.dep in TRAVERSED_RELATIONS]
        tok = sent.tokens[cur]
        if tok.dep in UPWARD_RELATIONS and tok.head_idx != cur:
            neighbours.append(sent.tokens[tok.head_idx])
        for n in neighbours:
            if n.idx in seen:
                continue
            seen.add(n.idx)
            queue.append((n.idx, d + 1))
            if _cue_form(n):
                found.append((n, d + 1))
    return found


def compute_factors(sent: ParsedSentence, opinion_idx: int, swn_lexicon: dict,
                    distance_weighted: bool,
                    modifier_source_idx: Optional[int] = None) -> SentimentFactors:
    """
    Hitung Neg, Int, dan skor leksikon untuk satu token opini.

    modifier_source_idx: jika diberikan, Neg/Int diambil dari token tsb (mode
    pewarisan tanpa scope consistency); skor leksikon tetap milik opini sendiri.
    """
    source = opinion_idx if modifier_source_idx is None else modifier_source_idx
    if distance_weighted:
        modifiers = reachable_modifiers(sent, source, config.NEG_DISTANCE_THRESHOLD)
    else:
        modifiers = direct_modifiers(sent, source)

    neg_val, int_val = 1.0, 1.0
    d_neg, d_int = None, None
    for m, d in sorted(modifiers, key=lambda x: x[1]):
        form = _cue_form(m)
        if form in NEGATION_WORDS and d_neg is None:
            neg_val, d_neg = -1.0, d
        elif form in INTENSITY_WORDS and d_int is None:
            base = 1.8 if form in AMPLIFIER_WORDS else 0.6
            d_int = d
            int_val = 1.0 + (base - 1.0) * math.exp(-config.INT_DECAY_LAMBDA * d) if distance_weighted else base

    swn = token_score(sent.tokens[opinion_idx], swn_lexicon)
    if swn == 0.0 and neg_val < 0:
        # Predikat dinegasi di luar leksikon ("tidak berfungsi") -> negatif lemah, bukan netral.
        swn = config.NEGATED_UNKNOWN_SCORE
    return SentimentFactors(neg=neg_val, intensifier=int_val, swn_score=swn,
                            distance_neg=d_neg, distance_int=d_int)
