"""
Perhitungan skor sentimen: formula statis (baseline) dan berbobot-jarak
(distance-weighted, revisi), plus scope consistency untuk pasangan R3/R4.
"""

import math
from dataclasses import dataclass
from typing import Optional

from parser import ParsedSentence, dependency_distance, find_dependents
import config

NEGATION_WORDS = {"tidak", "bukan", "belum", "tak", "jangan"}
AMPLIFIER_WORDS = {"sangat", "sekali", "banget", "amat", "super"}
DOWNTONER_WORDS = {"agak", "cukup", "lumayan", "sedikit"}


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
    """
    Ambil skor sentimen dasar dari leksikon (mis. SentiWordNet Bahasa Indonesia
    hasil terjemahan/adaptasi, atau leksikon lain yang Anda pakai).
    swn_lexicon: dict kata -> skor real (-1..+1). Default 0.0 jika tidak ada.
    """
    return swn_lexicon.get(opinion_lemma.lower(), 0.0)


def compute_factors(sent: ParsedSentence, opinion_idx: int, swn_lexicon: dict,
                     distance_weighted: bool) -> SentimentFactors:
    """
    Hitung Neg dan Int untuk SATU token opini, berdasarkan token yang
    terhubung padanya via relasi advmod/neg pada pohon dependensi.

    Jika distance_weighted=False: perilaku setara varian V6 (statis, presence-based).
    Jika distance_weighted=True: perilaku varian V7/V9 (fungsi jarak, lihat
    Bagian "Perhitungan Skor Sentimen" pada draf paten).
    """
    opinion_tok = sent.tokens[opinion_idx]
    modifiers = find_dependents(sent, opinion_idx, "advmod") + \
                find_dependents(sent, opinion_idx, "neg")

    neg_val = 1.0
    d_neg = None
    int_val = 1.0
    d_int = None

    for m in modifiers:
        lemma = m.lemma.lower()
        d = dependency_distance(sent, opinion_idx, m.idx) or 1

        if lemma in NEGATION_WORDS:
            if distance_weighted:
                if d <= config.NEG_DISTANCE_THRESHOLD:
                    neg_val = -1.0
                    d_neg = d
            else:
                neg_val = -1.0
                d_neg = d

        elif lemma in AMPLIFIER_WORDS or lemma in DOWNTONER_WORDS:
            base = 1.8 if lemma in AMPLIFIER_WORDS else 0.6
            d_int = d
            if distance_weighted:
                int_val = 1.0 + (base - 1.0) * math.exp(-config.INT_DECAY_LAMBDA * d)
            else:
                int_val = base

    swn = get_swn_score(opinion_tok.lemma, swn_lexicon)
    return SentimentFactors(neg=neg_val, intensifier=int_val, swn_score=swn,
                             distance_neg=d_neg, distance_int=d_int)


def apply_scope_consistency(sent: ParsedSentence, pairs_with_factors: list,
                             swn_lexicon: dict, distance_weighted: bool,
                             enabled: bool):
    """
    Untuk pasangan yang berasal dari R3/R4 (ekspansi konjungtif), hitung ULANG
    Neg/Int secara independen dari jalur dependensi pasangan tsb sendiri --
    BUKAN mewarisi dari pasangan asal.

    Jika enabled=False (varian V8): sengaja BIARKAN pewarisan (bug lama)
    utuh untuk keperluan perbandingan ablation -- lihat catatan di
    run_experiment.py soal cara memanggil fungsi ini dua kali (enabled=False
    vs True) pada pipeline yang sama.
    """
    if not enabled:
        return pairs_with_factors  # sengaja tidak mengubah apapun (mewarisi pasangan asal)

    for entry in pairs_with_factors:
        pair = entry["pair"]
        if pair.rule in ("R3", "R4"):
            factors = compute_factors(sent, pair.opinion_idx, swn_lexicon, distance_weighted)
            entry["factors"] = factors
    return pairs_with_factors
