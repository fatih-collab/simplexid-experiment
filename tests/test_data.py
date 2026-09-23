from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplexid.gold import (  # noqa: E402
    GoldValidationError,
    build_gold_pairs,
    build_split_frames,
    parse_accepted_pairs,
    validate_gold_pairs,
    validate_review_log,
)


def _terma_fixture() -> pd.DataFrame:
    rows = []

    tokens1 = ["kamar", "bersih", "dan", "nyaman"]
    aspects1 = [{"start": 0, "end": 0, "text": "kamar"}]
    opinions1 = [
        {"start": 1, "end": 1, "text": "bersih"},
        {"start": 3, "end": 3, "text": "nyaman"},
    ]
    rows.append(
        {
            "sentence_id": "terma_train_00000",
            "source_split": "train",
            "sentence": " ".join(tokens1),
            "tokens": json.dumps(tokens1),
            "aspects": json.dumps(aspects1),
            "opinions": json.dumps(opinions1),
            "n_aspects": 1,
            "n_opinions": 2,
            "labels_available": True,
        }
    )

    tokens2 = ["wifi", "tidak", "stabil"]
    aspects2 = [{"start": 0, "end": 0, "text": "wifi"}]
    opinions2 = [{"start": 1, "end": 2, "text": "tidak stabil"}]
    rows.append(
        {
            "sentence_id": "terma_validation_00000",
            "source_split": "validation",
            "sentence": " ".join(tokens2),
            "tokens": json.dumps(tokens2),
            "aspects": json.dumps(aspects2),
            "opinions": json.dumps(opinions2),
            "n_aspects": 1,
            "n_opinions": 1,
            "labels_available": True,
        }
    )

    return pd.DataFrame(rows)


def _review_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id": "terma_train_00000",
                "source_split": "train",
                "sentence": "kamar bersih dan nyaman",
                "accepted_pairs": "A1-O1|A1-O2",
                "review_scope": "DEV_COMPLEX",
                "annotation_status": "REVIEWED",
            },
            {
                "sentence_id": "terma_validation_00000",
                "source_split": "validation",
                "sentence": "wifi tidak stabil",
                "accepted_pairs": "A1-O1",
                "review_scope": "FINAL_SIMPLE",
                "annotation_status": "REVIEWED",
            },
        ]
    )


def test_parse_accepted_pairs():
    assert parse_accepted_pairs("A1-O1|A2-O3") == [(0, 0), (1, 2)]
    assert parse_accepted_pairs("NONE") == []
    assert parse_accepted_pairs("AMBIGUOUS") == []


def test_parse_accepted_pairs_rejects_bad_format():
    with pytest.raises(GoldValidationError):
        parse_accepted_pairs("A1O1")


def test_validate_review_and_build_pairs():
    terma = _terma_fixture()
    review = _review_fixture()
    summary = validate_review_log(review, terma, require_reviewed=True)
    assert summary["expected_pair_count"] == 3

    gold = build_gold_pairs(review, terma)
    assert len(gold) == 3
    assert set(gold["relation_pattern"]) == {"ONE_TO_MANY", "ONE_TO_ONE"}

    pair_summary = validate_gold_pairs(gold, terma, review)
    assert pair_summary["status"] == "PASS"


def test_review_rejects_out_of_range_pair():
    terma = _terma_fixture()
    review = _review_fixture()
    review.loc[0, "accepted_pairs"] = "A2-O1"
    with pytest.raises(GoldValidationError):
        validate_review_log(review, terma, require_reviewed=True)


def test_review_requires_reviewed_status():
    terma = _terma_fixture()
    review = _review_fixture()
    review.loc[0, "annotation_status"] = "AI_REVIEWED"
    with pytest.raises(GoldValidationError):
        validate_review_log(review, terma, require_reviewed=True)


def test_split_builder_with_real_expected_sizes():
    rows = []
    for i in range(3000):
        rows.append(
            {
                "sentence_id": f"terma_train_{i:05d}",
                "source_split": "train",
                "sentence": f"train {i}",
                "labels_available": True,
                "n_aspects": 1,
                "n_opinions": 1,
            }
        )
    for i in range(1000):
        rows.append(
            {
                "sentence_id": f"terma_validation_{i:05d}",
                "source_split": "validation",
                "sentence": f"validation {i}",
                "labels_available": True,
                "n_aspects": 1,
                "n_opinions": 1,
            }
        )

    terma = pd.DataFrame(rows)
    development, final_test, metadata = build_split_frames(terma)
    assert len(development) == 3000
    assert len(final_test) == 1000
    assert metadata["final_test"]["locked"] is True
