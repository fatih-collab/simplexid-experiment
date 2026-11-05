"""Ablation H4: tiap varian harus benar-benar berbeda dan tiap mekanisme bisa berpengaruh."""

import dataclasses

from simplexid import config
from simplexid.evaluation.gold_loader import GoldPairSpan
from simplexid.evaluation.subsets import ASYM_MOD, COORD_ASPECT, COORD_OPINION, pair_tags
from simplexid.parser import ParsedSentence, Token
from simplexid.pipeline import run_pipeline
from simplexid.rules import conj_source_opinion
from simplexid.scoring import compute_factors


def mock(spec):
    return ParsedSentence(tokens=[Token(idx=i, text=t, lemma=t, pos=p, dep=d, head_idx=h)
                                  for i, (t, p, d, h) in enumerate(spec)])


# "sinyal sangat lemah dan tidak stabil"
ASYM = mock([
    ("sinyal", "NOUN", "root", 0), ("sangat", "ADV", "advmod", 2), ("lemah", "ADJ", "amod", 0),
    ("dan", "CCONJ", "cc", 5), ("tidak", "PART", "advmod", 5), ("stabil", "ADJ", "conj", 2),
])
LEX = {"lemah": -0.4, "stabil": 0.4, "bersih": 0.5}


def _flags(v):
    return tuple(getattr(v, f.name) for f in dataclasses.fields(v) if f.name not in ("name", "description"))


def test_every_variant_has_unique_flags():
    flags = [_flags(v) for v in config.VARIANTS.values()]
    assert len(set(flags)) == len(flags)


def test_comparisons_reference_existing_variants():
    for a, b, _, targets in config.KEY_COMPARISONS:
        assert a in config.VARIANTS and b in config.VARIANTS and targets


def _stabil(variant):
    res = run_pipeline(None, config.VARIANTS[variant], "hotel", LEX, parsed_sentence=ASYM)
    return next(r for r in res if r.pair.opinion_idx == 5)


def test_scope_consistency_changes_r4_pair_factors():
    with_sc, without_sc = _stabil("A6"), _stabil("A5")
    assert with_sc.pair.rule == without_sc.pair.rule == "R4"
    # SC: "stabil" punya negasinya sendiri -> negatif
    assert with_sc.factors.neg == -1.0 and with_sc.factors.score_final < 0
    # Tanpa SC: mewarisi "sangat" dari "lemah" dan kehilangan negasi -> positif
    assert without_sc.factors.neg == 1.0 and without_sc.factors.intensifier > 1.0
    assert without_sc.factors.score_final > 0


def test_inheritance_keeps_own_lexicon_score():
    assert _stabil("A5").factors.swn_score == 0.4


def test_conj_source_opinion():
    assert conj_source_opinion(ASYM, 5) == 2
    assert conj_source_opinion(ASYM, 2) is None


def test_distance_weighting_reaches_indirect_negation():
    # "kamar tidak terlalu bersih": tidak -> terlalu -> bersih (jarak 2)
    sent = mock([
        ("kamar", "NOUN", "root", 0), ("tidak", "PART", "advmod", 2), ("terlalu", "ADV", "advmod", 3),
        ("bersih", "ADJ", "amod", 0),
    ])
    static = compute_factors(sent, 3, LEX, distance_weighted=False)
    weighted = compute_factors(sent, 3, LEX, distance_weighted=True)
    assert static.neg == 1.0
    assert weighted.neg == -1.0 and weighted.distance_neg == 2


def test_distance_weighting_does_not_cross_conjunction():
    assert compute_factors(ASYM, 2, LEX, distance_weighted=True).neg == 1.0
    assert compute_factors(ASYM, 5, LEX, distance_weighted=True).intensifier == 1.0


def test_intensifier_decays_only_in_distance_weighted_mode():
    assert compute_factors(ASYM, 2, LEX, distance_weighted=False).intensifier == 1.8
    assert 1.0 < compute_factors(ASYM, 2, LEX, distance_weighted=True).intensifier < 1.8


def test_gold_subset_tags():
    tokens = ["sinyal", "sangat", "lemah", "dan", "tidak", "stabil"]
    pairs = [GoldPairSpan("p1", (0, 0), (1, 2)), GoldPairSpan("p2", (0, 0), (4, 5))]
    assert pair_tags(tokens, pairs)["p1"] == {COORD_OPINION, ASYM_MOD}

    tokens = ["wifi", "dan", "tv", "rusak"]
    pairs = [GoldPairSpan("p1", (0, 0), (3, 3)), GoldPairSpan("p2", (2, 2), (3, 3))]
    assert pair_tags(tokens, pairs)["p2"] == {COORD_ASPECT}


def test_gold_subset_needs_only_connector_between_spans():
    tokens = ["kamar", "bersih", "tapi", "ac", "nya", "rusak"]
    pairs = [GoldPairSpan("p1", (0, 0), (1, 1)), GoldPairSpan("p2", (0, 0), (5, 5))]
    assert pair_tags(tokens, pairs)["p1"] == set()
