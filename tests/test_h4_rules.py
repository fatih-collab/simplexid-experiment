"""Perubahan rules H4.2-H4.3: tiap opsi diuji hidup vs mati pada kalimat tiruan."""

import json

import pandas as pd
import pytest

from simplexid.aspect_vocab import count_aspect_heads, fold_of, vocabulary
from simplexid.config import VARIANTS
from simplexid.parser import ParsedSentence, Token
from simplexid.pipeline import run_pipeline
from simplexid.rules import (
    BASE_RULES,
    DEFAULT_RULES,
    RuleOptions,
    expand_r3,
    expand_r4,
    extract_explicit_pairs,
)


def mock(spec):
    return ParsedSentence(tokens=[Token(idx=i, text=t, lemma=t, pos=p, dep=d, head_idx=h)
                                  for i, (t, p, d, h) in enumerate(spec)])


def keys(pairs):
    return {(p.aspect_idx, p.opinion_idx) for p in pairs}


LEX = {"kotor": -0.6, "bersih": 1.0, "dingin": -0.2, "oke": 0.6}


def test_passive_subject():
    # "pintu dikunci"
    sent = mock([("pintu", "NOUN", "nsubj:pass", 1), ("dikunci", "VERB", "root", 1)])
    assert keys(extract_explicit_pairs(sent, BASE_RULES)) == set()
    assert keys(extract_explicit_pairs(sent, RuleOptions(passive_subject=True))) == {(0, 1)}


def test_acl_relative():
    # "makanan memuaskan"
    sent = mock([("makanan", "NOUN", "root", 0), ("memuaskan", "VERB", "acl", 0)])
    assert keys(extract_explicit_pairs(sent, BASE_RULES)) == set()
    assert keys(extract_explicit_pairs(sent, RuleOptions(acl_relative=True))) == {(0, 1)}


def test_extended_aspect_pos():
    # "AC dingin", dengan "AC" diberi POS X
    sent = mock([("AC", "X", "root", 0), ("dingin", "ADJ", "amod", 0)])
    assert keys(extract_explicit_pairs(sent, BASE_RULES)) == set()
    assert keys(extract_explicit_pairs(sent, RuleOptions(extended_aspect_pos=True))) == {(0, 1)}


def test_lexicon_opinion_recovers_mistagged_words():
    # "kamar kotor" dengan "kotor" diberi POS NOUN/compound
    sent = mock([("kamar", "NOUN", "root", 0), ("kotor", "NOUN", "compound", 0)])
    assert keys(extract_explicit_pairs(sent, BASE_RULES, LEX)) == set()
    assert keys(extract_explicit_pairs(sent, RuleOptions(lexicon_opinion=True), LEX)) == {(0, 1)}


def test_lexicon_opinion_ignores_words_outside_lexicon():
    # "kamar mandi": "mandi" bukan opini
    sent = mock([("kamar", "NOUN", "root", 0), ("mandi", "NOUN", "compound", 0)])
    assert keys(extract_explicit_pairs(sent, RuleOptions(lexicon_opinion=True), LEX)) == set()


def test_phrase_spans_cover_compound_and_modifier():
    # "kamar mandi kurang bersih"
    sent = mock([
        ("kamar", "NOUN", "root", 0), ("mandi", "NOUN", "compound", 0),
        ("kurang", "ADV", "advmod", 3), ("bersih", "ADJ", "amod", 0),
    ])
    pair = extract_explicit_pairs(sent, RuleOptions(phrase_spans=True), LEX)[0]
    assert pair.aspect_span == (0, 1) and pair.aspect_text == "kamar mandi"
    assert pair.opinion_span == (2, 3) and pair.opinion_text == "kurang bersih"
    assert (pair.aspect_idx, pair.opinion_idx) == (0, 3)


def test_subjective_verbs_filter_keeps_lexicon_or_modified_verbs():
    options = RuleOptions(subjective_verbs=True)
    # "kami pesan kamar": verba netral -> R2 dibuang
    neutral = mock([("tamu", "NOUN", "nsubj", 1), ("pesan", "VERB", "root", 1), ("kamar", "NOUN", "obj", 1)])
    assert keys(extract_explicit_pairs(neutral, options, LEX)) == set()
    # "air tidak mengalir": verba di luar leksikon tetapi dinegasi -> dipertahankan
    negated = mock([("air", "NOUN", "nsubj", 2), ("tidak", "PART", "advmod", 2), ("mengalir", "VERB", "root", 2)])
    assert keys(extract_explicit_pairs(negated, options, LEX)) == {(0, 2)}


def test_distribution_guard_r3_skips_conjunct_with_own_opinion():
    # "kamar bersih , ac dingin": conj antar-klausa, "bersih" tidak menilai "ac"
    sent = mock([
        ("kamar", "NOUN", "root", 0), ("bersih", "ADJ", "amod", 0), (",", "PUNCT", "punct", 0),
        ("ac", "NOUN", "conj", 0), ("dingin", "ADJ", "amod", 3),
    ])
    pairs = extract_explicit_pairs(sent, BASE_RULES)
    assert (3, 1) in keys(expand_r3(sent, pairs, BASE_RULES))
    assert (3, 1) not in keys(expand_r3(sent, pairs, RuleOptions(distribution_guard=True)))


def test_distribution_guard_r4_skips_conjunct_with_own_subject():
    # "kamar bersih dan ac dingin": "dingin" punya subjek sendiri
    sent = mock([
        ("kamar", "NOUN", "root", 0), ("bersih", "ADJ", "amod", 0), ("dan", "CCONJ", "cc", 4),
        ("ac", "NOUN", "nsubj", 4), ("dingin", "ADJ", "conj", 1),
    ])
    pairs = [p for p in extract_explicit_pairs(sent, BASE_RULES) if p.opinion_idx == 1]
    assert (0, 4) in keys(expand_r4(sent, pairs, BASE_RULES))
    assert (0, 4) not in keys(expand_r4(sent, pairs, RuleOptions(distribution_guard=True)))


def test_aspect_vocabulary_filter():
    sent = mock([("hari", "NOUN", "root", 0), ("bersih", "ADJ", "amod", 0)])
    options = RuleOptions(aspect_vocabulary=True)
    assert keys(extract_explicit_pairs(sent, options, LEX, frozenset({"kamar"}))) == set()
    assert keys(extract_explicit_pairs(sent, options, LEX, frozenset({"hari"}))) == {(0, 1)}


def test_pipeline_uses_default_rules():
    sent = mock([("kamar", "NOUN", "root", 0), ("kotor", "NOUN", "compound", 0)])
    res = run_pipeline(None, VARIANTS["A0"], "hotel", LEX, parsed_sentence=sent,
                       aspect_vocabulary=frozenset({"kamar"}))
    assert [r.pair.rule for r in res] == ["R0"] and res[0].factors.score_final == -0.6
    assert DEFAULT_RULES.aspect_vocabulary is True


def _terma():
    def row(sid, split, tokens, aspects):
        return {"sentence_id": sid, "source_split": split, "aspects": json.dumps(aspects),
                "tokens": json.dumps(tokens)}
    span = {"start": 0, "end": 0, "text": "x"}
    return pd.DataFrame([
        row("terma_train_00000", "train", ["kamar"], [span]),
        row("terma_train_00001", "train", ["kamar"], [span]),
        row("terma_train_00002", "train", ["wifi"], [span]),
        row("terma_validation_00000", "validation", ["sarapan"], [span]),
    ])


def _parses():
    words = {"terma_train_00000": "kamar", "terma_train_00001": "kamar", "terma_train_00002": "wifi",
             "terma_validation_00000": "sarapan"}
    return {sid: mock([(w, "NOUN", "root", 0)]) for sid, w in words.items()}


def test_aspect_vocabulary_never_uses_final_split():
    counts = count_aspect_heads(_terma(), _parses())
    assert counts == {"kamar": 2, "wifi": 1}
    assert vocabulary(counts, min_count=2) == frozenset({"kamar"})


def test_aspect_vocabulary_excludes_held_out_fold():
    counts = count_aspect_heads(_terma(), _parses(), exclude=["terma_train_00000"])
    assert counts == {"kamar": 1, "wifi": 1}


def test_fold_of_is_stable():
    assert fold_of("terma_train_00000") == fold_of("terma_train_00000")
    assert 0 <= fold_of("terma_train_00000") < 5
