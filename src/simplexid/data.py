from __future__ import annotations

import json
from typing import List, Dict

import pandas as pd
from datasets import load_dataset

import ast
import json
import re
import pandas as pd

HOASA_CATEGORIES = [
    "ac",
    "air_panas",
    "bau",
    "general",
    "kebersihan",
    "linen",
    "service",
    "sunrise_meal",
    "tv",
    "wifi",
]


# ============================================================
# DATASET LOADER
# ============================================================

def load_terma():
    return load_dataset(
        "indonlp/indonlu",
        "terma"
    )


def load_hoasa():
    return load_dataset(
        "indonlp/indonlu",
        "hoasa"
    )

def get_sequence_label_names(
    dataset,
    field: str,
) -> List[str]:

    feature = dataset["train"].features[field]

    if hasattr(feature, "feature"):
        feature = feature.feature

    if not hasattr(feature, "names"):
        raise ValueError(
            f"{field} bukan ClassLabel"
        )

    return list(feature.names)


def decode_sequence_labels(
    labels,
    label_names: List[str],
) -> List[str]:

    decoded = []

    for label in labels:

        if isinstance(label, str):
            decoded.append(label)

        else:
            decoded.append(
                label_names[int(label)]
            )

    return decoded

def iob_to_spans(
    tokens: List[str],
    labels: List[str],
    entity: str,
) -> List[Dict]:

    b_tag = f"B-{entity}"
    i_tag = f"I-{entity}"

    spans = []

    start = None

    for i, label in enumerate(labels):

        if label == b_tag:

            if start is not None:
                end = i - 1

                spans.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(
                        tokens[start:end + 1]
                    ),
                })

            start = i

        elif label == i_tag:

            # I-tag tanpa B sebelumnya.
            # Tetap kita recovery,
            # tetapi nanti dapat diaudit.
            if start is None:
                start = i

        else:

            if start is not None:

                end = i - 1

                spans.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(
                        tokens[start:end + 1]
                    ),
                })

                start = None

    if start is not None:

        end = len(tokens) - 1

        spans.append({
            "start": start,
            "end": end,
            "text": " ".join(
                tokens[start:end + 1]
            ),
        })

    return spans

def prepare_terma(
    dataset
) -> pd.DataFrame:

    label_names = get_sequence_label_names(
        dataset,
        "seq_label"
    )

    rows = []

    for split_name, split_data in dataset.items():

        # Official test labels IndoNLU dimask.
        labels_available = (
            split_name != "test"
        )

        for index, sample in enumerate(split_data):

            tokens = sample["tokens"]

            sentence_id = (
                f"terma_{split_name}_{index:05d}"
            )

            if labels_available:

                labels = decode_sequence_labels(
                    sample["seq_label"],
                    label_names,
                )

                aspects = iob_to_spans(
                    tokens,
                    labels,
                    "ASPECT",
                )

                opinions = iob_to_spans(
                    tokens,
                    labels,
                    "SENTIMENT",
                )

            else:

                labels = []
                aspects = []
                opinions = []

            rows.append({
                "sentence_id": sentence_id,
                "dataset": "terma",
                "source_split": split_name,

                "sentence":
                    " ".join(tokens),

                "tokens":
                    json.dumps(
                        tokens,
                        ensure_ascii=False
                    ),

                "labels":
                    json.dumps(
                        labels,
                        ensure_ascii=False
                    ),

                "aspects":
                    json.dumps(
                        aspects,
                        ensure_ascii=False
                    ),

                "opinions":
                    json.dumps(
                        opinions,
                        ensure_ascii=False
                    ),

                "n_aspects":
                    len(aspects)
                    if labels_available
                    else None,

                "n_opinions":
                    len(opinions)
                    if labels_available
                    else None,

                "labels_available":
                    labels_available,
            })

    return pd.DataFrame(rows)

def prepare_hoasa(
    dataset
) -> pd.DataFrame:

    category_labels = {}

    for category in HOASA_CATEGORIES:

        feature = (
            dataset["train"]
            .features[category]
        )

        category_labels[category] = (
            list(feature.names)
        )

    rows = []

    for split_name, split_data in dataset.items():

        labels_available = (
            split_name != "test"
        )

        for index, sample in enumerate(split_data):

            row = {
                "sentence_id":
                    f"hoasa_{split_name}_{index:05d}",

                "dataset": "hoasa",
                "source_split":
                    split_name,

                "sentence":
                    sample["sentence"],

                "labels_available":
                    labels_available,
            }

            for category in HOASA_CATEGORIES:

                if labels_available:

                    value = sample[category]

                    if isinstance(value, str):
                        decoded = value
                    else:
                        decoded = (
                            category_labels[
                                category
                            ][int(value)]
                        )

                    row[category] = decoded

                else:

                    row[category] = None

            rows.append(row)

    return pd.DataFrame(rows)

def generate_pair_candidates(
    terma_df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    data = terma_df[
        terma_df["labels_available"]
    ]

    for _, row in data.iterrows():

        aspects = json.loads(
            row["aspects"]
        )

        opinions = json.loads(
            row["opinions"]
        )

        if not aspects or not opinions:
            continue

        # 1 aspect + 1 opinion
        if (
            len(aspects) == 1
            and
            len(opinions) == 1
        ):

            combinations = [
                (
                    aspects[0],
                    opinions[0]
                )
            ]

            review_type = (
                "AUTO_HIGH_CONFIDENCE"
            )

        else:

            combinations = [
                (a, o)
                for a in aspects
                for o in opinions
            ]

            review_type = (
                "MANUAL_REVIEW_REQUIRED"
            )

        for aspect, opinion in combinations:

            rows.append({
                "sentence_id":
                    row["sentence_id"],

                "source_split":
                    row["source_split"],

                "sentence":
                    row["sentence"],

                "aspect_text":
                    aspect["text"],

                "aspect_start":
                    aspect["start"],

                "aspect_end":
                    aspect["end"],

                "opinion_text":
                    opinion["text"],

                "opinion_start":
                    opinion["start"],

                "opinion_end":
                    opinion["end"],

                "review_type":
                    review_type,
            })

    return pd.DataFrame(rows)

def build_sentence_review_table(
    terma_df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    labeled = terma_df[
        terma_df["labels_available"]
    ].copy()

    for _, row in labeled.iterrows():

        aspects = json.loads(
            row["aspects"]
        )

        opinions = json.loads(
            row["opinions"]
        )

        if not aspects or not opinions:
            continue

        # Hanya complex cases
        if (
            len(aspects) == 1
            and len(opinions) == 1
        ):
            continue

        aspect_labels = []

        for i, aspect in enumerate(
            aspects,
            start=1
        ):

            aspect_labels.append(
                f"A{i}={aspect['text']}"
            )

        opinion_labels = []

        for i, opinion in enumerate(
            opinions,
            start=1
        ):

            opinion_labels.append(
                f"O{i}={opinion['text']}"
            )

        rows.append({
            "sentence_id":
                row["sentence_id"],

            "source_split":
                row["source_split"],

            "sentence":
                row["sentence"],

            "aspects":
                " | ".join(
                    aspect_labels
                ),

            "opinions":
                " | ".join(
                    opinion_labels
                ),

            # Diisi annotator
            # contoh: A1-O1|A2-O2
            "accepted_pairs":
                "",

            "relation_type":
                "",

            "secondary_tags":
                "",

            "notes":
                "",
        })

    return pd.DataFrame(rows)

def _parse_span_list(value):
    """
    Membaca kolom aspects/opinions dari terma_processed.csv.
    Mendukung JSON string maupun Python-list string.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def build_gold_pairs_from_review(
    review_df: pd.DataFrame,
    terma_df: pd.DataFrame
) -> pd.DataFrame:

    required_review = {
        "sentence_id",
        "source_split",
        "accepted_pairs",
        "review_scope"
    }

    missing = required_review - set(review_df.columns)

    if missing:
        raise ValueError(
            f"Kolom review tidak lengkap: {missing}"
        )

    terma_map = (
        terma_df
        .set_index("sentence_id")
    )

    rows = []

    for _, review_row in review_df.iterrows():

        sentence_id = review_row["sentence_id"]

        if sentence_id not in terma_map.index:
            raise ValueError(
                f"sentence_id tidak ditemukan: {sentence_id}"
            )

        accepted = str(
            review_row["accepted_pairs"]
        ).strip()

        if (
            not accepted
            or accepted.upper() in {
                "NONE",
                "AMBIGUOUS",
                "NAN"
            }
        ):
            continue

        source_row = terma_map.loc[sentence_id]

        aspects = _parse_span_list(
            source_row["aspects"]
        )

        opinions = _parse_span_list(
            source_row["opinions"]
        )

        for pair in accepted.split("|"):

            pair = pair.strip()

            match = re.fullmatch(
                r"A(\d+)-O(\d+)",
                pair
            )

            if not match:
                raise ValueError(
                    f"Format pair invalid "
                    f"{sentence_id}: {pair}"
                )

            aspect_idx = int(match.group(1)) - 1
            opinion_idx = int(match.group(2)) - 1

            if not (
                0 <= aspect_idx < len(aspects)
            ):
                raise ValueError(
                    f"Aspect index invalid "
                    f"{sentence_id}: {pair}"
                )

            if not (
                0 <= opinion_idx < len(opinions)
            ):
                raise ValueError(
                    f"Opinion index invalid "
                    f"{sentence_id}: {pair}"
                )

            aspect = aspects[aspect_idx]
            opinion = opinions[opinion_idx]

            rows.append({
                "sentence_id":
                    sentence_id,

                "source_split":
                    review_row["source_split"],

                "review_scope":
                    review_row["review_scope"],

                "sentence":
                    source_row["sentence"],

                "aspect_text":
                    aspect["text"],

                "aspect_start":
                    aspect["start"],

                "aspect_end":
                    aspect["end"],

                "opinion_text":
                    opinion["text"],

                "opinion_start":
                    opinion["start"],

                "opinion_end":
                    opinion["end"],
            })

    gold = pd.DataFrame(rows)

    gold = gold.drop_duplicates(
        subset=[
            "sentence_id",
            "aspect_start",
            "aspect_end",
            "opinion_start",
            "opinion_end"
        ]
    ).reset_index(drop=True)

    return gold