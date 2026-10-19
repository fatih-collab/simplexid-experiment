"""Test harness evaluasi H3.3 dengan contoh kecil yang dihitung manual."""

import json

import pandas as pd
import pytest

from simplexid.evaluation.gold_loader import (
    SOURCE_AUTO,
    SOURCE_NO_SPANS,
    SOURCE_REVIEWED,
    STRATUM_EMPTY,
    STRATUM_SIMPLE,
    GoldPairSpan,
    build_eval_gold,
)
from simplexid.evaluation.matching import PredPair, match_pairs
from simplexid.evaluation.metrics import (
    Counts,
    classification_report,
    extraction_counts,
    implicit_items,
    per_rule_precision,
    polarity_items,
    polarity_of,
    summarize_implicit,
)
from simplexid.evaluation.run_record import FrozenConfigError, check_frozen, freeze
from simplexid.evaluation.significance import holm, mcnemar_exact, paired_bootstrap_f1
from simplexid.gold import GoldValidationError


def pred(a, o, rule="R0", score=0.5, a_head=None, o_head=None):
    return PredPair(aspect_span=a, aspect_head=a[0] if a_head is None else a_head,
                    opinion_span=o, opinion_head=o[0] if o_head is None else o_head,
                    rule=rule, score=score)


# gold: "ac nya kurang dingin" -> aspek [0,1], opini [2,3]
GOLD = [GoldPairSpan("s__A1__O1", (0, 1), (2, 3))]


def test_strict_requires_exact_spans():
    assert match_pairs("strict", [pred((0, 1), (2, 3))], GOLD) == {0: 0}
    assert match_pairs("strict", [pred((0, 0), (3, 3))], GOLD) == {}


def test_head_match_accepts_head_inside_gold_span():
    # rules memilih head token: "ac" dan "dingin"
    assert match_pairs("head", [pred((0, 0), (3, 3))], GOLD) == {0: 0}
    # head opini di luar span gold -> tidak cocok
    assert match_pairs("head", [pred((0, 0), (4, 4))], GOLD) == {}


def test_overlap_match():
    assert match_pairs("overlap", [pred((1, 2), (3, 5))], GOLD) == {0: 0}


def test_matching_is_one_to_one():
    # dua prediksi untuk satu gold -> hanya satu TP
    preds = [pred((0, 0), (3, 3)), pred((1, 1), (2, 2))]
    assert len(match_pairs("head", preds, GOLD)) == 1


def test_bipartite_matching_finds_maximum():
    # p0 cocok dengan g0 & g1, p1 hanya dengan g0 -> greedy salah, maksimum = 2
    golds = [GoldPairSpan("g0", (0, 0), (2, 3)), GoldPairSpan("g1", (0, 0), (2, 2))]
    preds = [pred((0, 0), (2, 2)), pred((0, 0), (3, 3))]
    assert len(match_pairs("head", preds, golds)) == 2


def test_implicit_predictions_never_match_explicit_gold():
    implicit = PredPair(None, None, (3, 3), 3, "R8", implicit_category="ac")
    assert match_pairs("head", [implicit], GOLD) == {}


def test_counts_prf_hand_computed():
    # tp=2 fp=1 fn=1 -> P=2/3 R=2/3 F1=2/3
    r = Counts(2, 1, 1).prf()
    assert (r["precision"], r["recall"], r["f1"]) == (0.6667, 0.6667, 0.6667)


def test_polarity_threshold():
    assert polarity_of(0.06) == "POSITIVE"
    assert polarity_of(-0.06) == "NEGATIVE"
    assert polarity_of(0.0) == "NEUTRAL"


def test_classification_report_counts_missing_as_wrong():
    rep = classification_report(["POSITIVE", "NEGATIVE", "POSITIVE"], ["POSITIVE", None, "NEGATIVE"])
    assert rep["accuracy"] == pytest.approx(0.3333, abs=1e-4)


# ------------------------------------------------------------------ gold loader

def _terma_row(sid, tokens, aspects, opinions):
    return {
        "sentence_id": sid, "source_split": "train" if "train" in sid else "validation",
        "sentence": " ".join(tokens), "tokens": json.dumps(tokens),
        "aspects": json.dumps(aspects), "opinions": json.dumps(opinions),
        "n_aspects": len(aspects), "n_opinions": len(opinions),
    }


def _fixture():
    sp = lambda s, e, t: {"start": s, "end": e, "text": t}  # noqa: E731
    terma = pd.DataFrame([
        # reviewed complex
        _terma_row("terma_train_00000", ["kamar", "bersih", "dan", "nyaman"],
                   [sp(0, 0, "kamar")], [sp(1, 1, "bersih"), sp(3, 3, "nyaman")]),
        # auto 1A1O
        _terma_row("terma_train_00001", ["ac", "dingin"], [sp(0, 0, "ac")], [sp(1, 1, "dingin")]),
        # no aspect -> empty gold
        _terma_row("terma_train_00002", ["mantap", "!"], [], [sp(0, 0, "mantap")]),
        # unreviewed complex -> excluded
        _terma_row("terma_train_00003", ["wifi", "dan", "tv", "rusak"],
                   [sp(0, 0, "wifi"), sp(2, 2, "tv")], [sp(3, 3, "rusak")]),
        # final, reviewed
        _terma_row("terma_validation_00000", ["air", "panas"], [sp(0, 0, "air")], [sp(1, 1, "panas")]),
    ])
    review = pd.DataFrame([
        {"sentence_id": "terma_train_00000", "source_split": "train", "accepted_pairs": "A1-O1|A1-O2"},
        {"sentence_id": "terma_validation_00000", "source_split": "validation", "accepted_pairs": "A1-O1"},
    ])
    gold_pairs = pd.DataFrame([
        {"pair_id": "terma_train_00000__A1__O1", "sentence_id": "terma_train_00000", "source_split": "train",
         "aspect_start": 0, "aspect_end": 0, "opinion_start": 1, "opinion_end": 1},
        {"pair_id": "terma_train_00000__A1__O2", "sentence_id": "terma_train_00000", "source_split": "train",
         "aspect_start": 0, "aspect_end": 0, "opinion_start": 3, "opinion_end": 3},
        {"pair_id": "terma_validation_00000__A1__O1", "sentence_id": "terma_validation_00000",
         "source_split": "validation", "aspect_start": 0, "aspect_end": 0, "opinion_start": 1, "opinion_end": 1},
    ])
    auto = pd.DataFrame([{"sentence_id": "terma_train_00001", "source_split": "train",
                          "accepted_pairs": "A1-O1", "annotation_status": "AUTO"}])
    return terma, review, gold_pairs, auto


def test_dev_universe_sources_and_exclusions():
    terma, review, gold_pairs, auto = _fixture()
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)

    assert gold.sentence_ids == ["terma_train_00000", "terma_train_00001", "terma_train_00002"]
    assert gold.source == {"terma_train_00000": SOURCE_REVIEWED, "terma_train_00001": SOURCE_AUTO,
                           "terma_train_00002": SOURCE_NO_SPANS}
    assert gold.stratum["terma_train_00001"] == STRATUM_SIMPLE
    assert gold.stratum["terma_train_00002"] == STRATUM_EMPTY
    assert gold.excluded["unreviewed_complex"] == 1


def test_auto_status_reviewed_changes_source_and_version():
    terma, review, gold_pairs, auto = _fixture()
    before = build_eval_gold("dev", terma, review, gold_pairs, auto)
    auto.loc[0, "annotation_status"] = "REVIEWED"
    after = build_eval_gold("dev", terma, review, gold_pairs, auto)

    assert after.source["terma_train_00001"] == SOURCE_REVIEWED
    assert before.version != after.version


def test_auto_none_gives_empty_gold_and_ambiguous_is_excluded():
    terma, review, gold_pairs, auto = _fixture()
    auto.loc[0, "accepted_pairs"] = "NONE"
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)
    assert gold.pairs["terma_train_00001"] == []

    auto.loc[0, "accepted_pairs"] = "AMBIGUOUS"
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)
    assert "terma_train_00001" not in gold.sentence_ids


def test_auto_sheet_is_ignored_for_final():
    terma, review, gold_pairs, auto = _fixture()
    gold = build_eval_gold("final", terma, review, gold_pairs, auto)
    assert gold.sentence_ids == ["terma_validation_00000"]
    assert set(gold.source.values()) == {SOURCE_REVIEWED}


def test_final_requires_every_spanned_sentence_reviewed():
    terma, review, gold_pairs, auto = _fixture()
    with pytest.raises(GoldValidationError):
        build_eval_gold("final", terma, review.iloc[:1], gold_pairs, auto)


def test_extraction_counts_fp_on_empty_gold_sentence():
    terma, review, gold_pairs, auto = _fixture()
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)
    preds = {
        "terma_train_00000": [pred((0, 0), (1, 1)), pred((0, 0), (2, 2), rule="R4")],  # 1 TP, 1 FP, 1 FN
        "terma_train_00001": [pred((0, 0), (1, 1))],                                    # 1 TP
        "terma_train_00002": [pred((0, 0), (1, 1))],                                    # 1 FP
    }
    counts = extraction_counts(gold, preds, "head")
    total = sum(counts.values(), Counts())
    assert (total.tp, total.fp, total.fn) == (2, 2, 1)
    rules = per_rule_precision(gold, preds, "head")
    assert rules["R0"] == {"predicted": 3, "correct": 2, "precision": 0.6667}
    assert rules["R4"]["correct"] == 0


def test_polarity_items_modes():
    terma, review, gold_pairs, auto = _fixture()
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)
    polarity = pd.DataFrame([
        {"pair_id": "terma_train_00000__A1__O1", "polarity_scope": "DEV_POLARITY",
         "polarity_stratum": "NORMAL", "gold_polarity": "POSITIVE"},
        {"pair_id": "terma_train_00000__A1__O2", "polarity_scope": "DEV_POLARITY",
         "polarity_stratum": "NORMAL", "gold_polarity": "POSITIVE"},
    ])
    preds = {"terma_train_00000": [pred((0, 0), (1, 1), score=-0.4)]}   # extracted O1 only, wrong sign
    items = polarity_items(gold, preds, polarity, oracle_score_fn=lambda *a: 0.5)

    assert items["pred_end_to_end"].iloc[0] == "NEGATIVE"
    assert pd.isna(items["pred_end_to_end"].iloc[1])     # O2 tidak terekstraksi
    assert items["pred_oracle_pair"].tolist() == ["POSITIVE", "POSITIVE"]


def test_implicit_detection_and_category():
    terma, review, gold_pairs, auto = _fixture()
    gold = build_eval_gold("dev", terma, review, gold_pairs, auto)
    implicit = pd.DataFrame([
        {"implicit_id": "terma_train_00002__O1", "sentence_id": "terma_train_00002",
         "implicit_scope": "DEV_IMPLICIT", "opinion_start": 0, "opinion_end": 0,
         "decision": "IMPLICIT", "gold_aspect_category": "general"},
    ])
    preds = {"terma_train_00002": [PredPair(None, None, (0, 0), 0, "R8", implicit_category="service")]}
    summary = summarize_implicit(implicit_items(gold, preds, implicit), baseline_category="general")

    assert summary["detection"]["tp"] == 1
    assert summary["category"]["accuracy"] == 0.0
    assert summary["majority_baseline"]["accuracy"] == 1.0


# ------------------------------------------------------------------ significance

def test_bootstrap_identical_systems_zero_delta():
    c = {f"s{i}": Counts(1, i % 2, 0) for i in range(30)}
    r = paired_bootstrap_f1(c, c, n_resamples=200)
    assert r["delta"] == 0.0 and r["ci95"] == [0.0, 0.0]


def test_bootstrap_is_deterministic_and_detects_improvement():
    a = {f"s{i}": Counts(0, 1, 1) for i in range(50)}
    b = {f"s{i}": Counts(1, 0, 0) for i in range(50)}
    r1 = paired_bootstrap_f1(a, b, n_resamples=500)
    r2 = paired_bootstrap_f1(a, b, n_resamples=500)
    assert r1 == r2
    assert r1["delta"] == 1.0 and r1["p_value"] == 0.0


def test_mcnemar_hand_computed():
    # A benar 0 kali sendirian, B benar 6 kali sendirian -> p = 2 * 0.5^6 = 0.03125
    r = mcnemar_exact([False] * 6 + [True] * 4, [True] * 10)
    assert (r["a_only_correct"], r["b_only_correct"]) == (0, 6)
    assert r["p_value"] == pytest.approx(0.0312, abs=1e-4)


def test_holm_hand_computed():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


# ------------------------------------------------------------------ freeze lock

def test_freeze_lock_detects_code_change(tmp_path):
    (tmp_path / "src" / "simplexid").mkdir(parents=True)
    code = tmp_path / "src" / "simplexid" / "rules.py"
    code.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(FrozenConfigError):
        check_frozen(tmp_path)
    freeze(tmp_path, note="test")
    check_frozen(tmp_path)

    code.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="code_sha256"):
        check_frozen(tmp_path)
