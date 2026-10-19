import json

import pandas as pd
import pytest

from simplexid.oracle import (
    SOURCE_AUTO_1A1O,
    SOURCE_REVIEWED,
    build_dev_gold,
    dependency_path,
    pair_pattern,
    span_head,
    summarize,
    oracle_records,
)
from simplexid.parse_cache import ParseCacheError, build_parse_cache, load_parse_cache
from simplexid.parser import ParsedSentence, Token, get_parser


def mock_sentence(spec):
    """spec: list of (text, pos, dep, head_idx); index = urutan (0-based)."""
    return ParsedSentence(tokens=[
        Token(idx=i, text=t, lemma=t, pos=p, dep=d, head_idx=h)
        for i, (t, p, d, h) in enumerate(spec)
    ])


# "sinyal sangat lemah dan tidak stabil"
SENT = mock_sentence([
    ("sinyal", "NOUN", "root", 0),
    ("sangat", "ADV", "advmod", 2),
    ("lemah", "ADJ", "amod", 0),
    ("dan", "CCONJ", "cc", 5),
    ("tidak", "PART", "advmod", 5),
    ("stabil", "ADJ", "conj", 2),
])


def test_span_head_picks_token_attached_outside_span():
    assert span_head(SENT, 4, 5) == 5        # "tidak stabil" -> stabil
    assert span_head(SENT, 1, 2) == 2        # "sangat lemah" -> lemah
    assert span_head(SENT, 0, 0) == 0


def test_dependency_path_directions():
    assert dependency_path(SENT, 0, 2) == ["vamod"]
    assert dependency_path(SENT, 0, 5) == ["vamod", "vconj"]
    assert dependency_path(SENT, 5, 0) == ["^conj", "^amod"]


def test_pair_pattern_includes_pos_and_same_token():
    assert pair_pattern(SENT, 0, 5) == "NOUN vamod vconj ADJ"
    assert pair_pattern(SENT, 2, 2) == "SAME_TOKEN"


def _terma():
    rows = []
    for sid, toks, aspects, opinions in [
        ("terma_train_00000", ["kamar", "bersih"],
         [{"start": 0, "end": 0, "text": "kamar"}], [{"start": 1, "end": 1, "text": "bersih"}]),
        ("terma_train_00001", ["ac", "dingin"],
         [{"start": 0, "end": 0, "text": "ac"}], [{"start": 1, "end": 1, "text": "dingin"}]),
        ("terma_validation_00000", ["wifi", "lemot"],
         [{"start": 0, "end": 0, "text": "wifi"}], [{"start": 1, "end": 1, "text": "lemot"}]),
    ]:
        rows.append({
            "sentence_id": sid, "source_split": sid.split("_")[1], "sentence": " ".join(toks),
            "tokens": json.dumps(toks), "aspects": json.dumps(aspects),
            "opinions": json.dumps(opinions), "n_aspects": 1, "n_opinions": 1,
        })
    return pd.DataFrame(rows)


def test_build_dev_gold_marks_sources_and_excludes_reviewed_auto():
    terma = _terma()
    gold_pairs = pd.DataFrame([{
        "sentence_id": "terma_train_00000", "source_split": "train",
        "aspect_text": "kamar", "aspect_start": 0, "aspect_end": 0,
        "opinion_text": "bersih", "opinion_start": 1, "opinion_end": 1,
    }])
    review = pd.DataFrame({"sentence_id": ["terma_train_00000"]})

    gold = build_dev_gold(terma, gold_pairs, review)

    assert dict(zip(gold["sentence_id"], gold["source"])) == {
        "terma_train_00000": SOURCE_REVIEWED,
        "terma_train_00001": SOURCE_AUTO_1A1O,
    }


def test_summarize_upper_bound_is_cumulative():
    gold = pd.DataFrame([
        {"sentence_id": "s", "source": SOURCE_REVIEWED, "aspect_text": "sinyal",
         "aspect_start": 0, "aspect_end": 0, "opinion_text": t,
         "opinion_start": s, "opinion_end": e}
        for t, s, e in [("sangat lemah", 1, 2), ("tidak stabil", 4, 5)]
    ])
    summary = summarize(oracle_records(gold, {"s": SENT}))

    assert summary["distance_distribution"] == {"1": 1, "2": 1}
    assert summary["recall_upper_bound_by_distance"]["<=1"] == 0.5
    assert summary["recall_upper_bound_by_distance"]["<=2"] == 1.0


def test_parse_cache_roundtrip_and_tamper_detection(tmp_path):
    terma = _terma()
    parser = get_parser()
    build_parse_cache(terma, tmp_path, parser)

    parses = load_parse_cache(tmp_path, terma_df=terma, model_path=parser.model_path)
    assert set(parses) == set(terma["sentence_id"])
    assert parses["terma_train_00000"].token_text() == ["kamar", "bersih"]

    changed = terma.copy()
    changed.loc[0, "tokens"] = json.dumps(["kamar", "kotor"])
    with pytest.raises(ParseCacheError):
        load_parse_cache(tmp_path, terma_df=changed)

    cache_file = tmp_path / "terma.conllu"
    cache_file.write_text(cache_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ParseCacheError):
        load_parse_cache(tmp_path)


def test_missing_cache_has_actionable_error(tmp_path):
    with pytest.raises(ParseCacheError, match="run_parse.py cache"):
        load_parse_cache(tmp_path)
