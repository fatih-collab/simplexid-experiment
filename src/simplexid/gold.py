from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ontologi kategori implisit mengikuti kategori HoASA yang sudah dipakai di proyek.
IMPLICIT_CATEGORIES = [
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

VALID_POLARITIES = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
VALID_IMPLICIT_DECISIONS = {"IMPLICIT", "NOT_IMPLICIT", "AMBIGUOUS"}
SPECIAL_PAIR_VALUES = {"NONE", "AMBIGUOUS"}
PAIR_RE = re.compile(r"^A(\d+)-O(\d+)$")


class GoldValidationError(ValueError):
    """Raised when an H2 gold file violates the expected schema or references."""


def _is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def parse_json_list(value) -> list:
    """Parse a JSON-list column from processed CSV safely."""
    if isinstance(value, list):
        return value
    if _is_blank(value):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise GoldValidationError(f"Expected JSON list, got: {type(parsed).__name__}")
    return parsed


def parse_accepted_pairs(value) -> list[tuple[int, int]]:
    """
    Parse 'A1-O1|A2-O2' into zero-based (aspect_idx, opinion_idx) tuples.
    NONE/AMBIGUOUS produce an empty pair list.
    """
    if _is_blank(value):
        raise GoldValidationError("accepted_pairs kosong")

    text = str(value).strip()
    upper = text.upper()
    if upper in SPECIAL_PAIR_VALUES:
        return []

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for raw in text.split("|"):
        token = raw.strip()
        match = PAIR_RE.fullmatch(token)
        if not match:
            raise GoldValidationError(f"Format accepted_pairs invalid: {token!r}")

        pair = (int(match.group(1)) - 1, int(match.group(2)) - 1)
        if pair[0] < 0 or pair[1] < 0:
            raise GoldValidationError(f"Index A/O harus mulai dari 1: {token!r}")
        if pair in seen:
            raise GoldValidationError(f"Duplicate pair dalam satu row: {token!r}")

        seen.add(pair)
        pairs.append(pair)

    return pairs


def _ensure_unique_sentence_ids(df: pd.DataFrame, name: str) -> None:
    duplicated = df["sentence_id"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "sentence_id"].head(10).tolist()
        raise GoldValidationError(
            f"{name}: duplicate sentence_id ditemukan, contoh: {examples}"
        )


def validate_review_log(
    review_df: pd.DataFrame,
    terma_df: pd.DataFrame,
    *,
    require_reviewed: bool = True,
) -> dict:
    """Validate gold_review_log.csv against terma_processed.csv."""
    required_review = {
        "sentence_id",
        "source_split",
        "sentence",
        "accepted_pairs",
        "review_scope",
        "annotation_status",
    }
    required_terma = {
        "sentence_id",
        "source_split",
        "sentence",
        "tokens",
        "aspects",
        "opinions",
    }

    missing_review = required_review - set(review_df.columns)
    missing_terma = required_terma - set(terma_df.columns)
    if missing_review:
        raise GoldValidationError(
            f"gold_review_log.csv kurang kolom: {sorted(missing_review)}"
        )
    if missing_terma:
        raise GoldValidationError(
            f"terma_processed.csv kurang kolom: {sorted(missing_terma)}"
        )

    _ensure_unique_sentence_ids(review_df, "gold_review_log")
    _ensure_unique_sentence_ids(terma_df, "terma_processed")

    if review_df["accepted_pairs"].isna().any():
        ids = review_df.loc[review_df["accepted_pairs"].isna(), "sentence_id"].head(10).tolist()
        raise GoldValidationError(f"accepted_pairs kosong pada: {ids}")

    if require_reviewed:
        statuses = set(review_df["annotation_status"].fillna("").astype(str).str.strip())
        if statuses != {"REVIEWED"}:
            raise GoldValidationError(
                "annotation_status belum final. Semua row final harus bernilai REVIEWED. "
                f"Status yang ditemukan: {sorted(statuses)}"
            )

    terma_map = terma_df.set_index("sentence_id", drop=False)
    unknown = sorted(set(review_df["sentence_id"]) - set(terma_map.index))
    if unknown:
        raise GoldValidationError(f"sentence_id review tidak ada di TermA: {unknown[:10]}")

    expected_pair_count = 0
    special_count = 0

    for row in review_df.itertuples(index=False):
        source = terma_map.loc[row.sentence_id]
        if str(row.source_split) != str(source["source_split"]):
            raise GoldValidationError(
                f"source_split mismatch untuk {row.sentence_id}: "
                f"review={row.source_split}, terma={source['source_split']}"
            )

        aspects = parse_json_list(source["aspects"])
        opinions = parse_json_list(source["opinions"])
        accepted_text = str(row.accepted_pairs).strip()
        pairs = parse_accepted_pairs(accepted_text)

        if accepted_text.upper() in SPECIAL_PAIR_VALUES:
            special_count += 1
            continue

        for a_idx, o_idx in pairs:
            if a_idx >= len(aspects):
                raise GoldValidationError(
                    f"Aspect A{a_idx + 1} tidak ada pada {row.sentence_id}"
                )
            if o_idx >= len(opinions):
                raise GoldValidationError(
                    f"Opinion O{o_idx + 1} tidak ada pada {row.sentence_id}"
                )
        expected_pair_count += len(pairs)

    return {
        "review_rows": int(len(review_df)),
        "expected_pair_count": int(expected_pair_count),
        "special_rows": int(special_count),
        "status": "PASS",
    }


def _relation_pattern(pairs: list[tuple[int, int]]) -> str:
    """Structural pattern only; this is NOT a syntactic coordination label."""
    if not pairs:
        return "NO_PAIR"
    if len(pairs) == 1:
        return "ONE_TO_ONE"

    aspects = {a for a, _ in pairs}
    opinions = {o for _, o in pairs}

    if len(aspects) == 1 and len(opinions) > 1:
        return "ONE_TO_MANY"
    if len(aspects) > 1 and len(opinions) == 1:
        return "MANY_TO_ONE"
    return "MANY_TO_MANY"


def build_gold_pairs(review_df: pd.DataFrame, terma_df: pd.DataFrame) -> pd.DataFrame:
    """Convert reviewed sentence-level A/O links into pair-level explicit gold."""
    validate_review_log(review_df, terma_df, require_reviewed=False)
    terma_map = terma_df.set_index("sentence_id", drop=False)

    rows: list[dict] = []

    for review_row in review_df.itertuples(index=False):
        accepted_text = str(review_row.accepted_pairs).strip()
        pairs = parse_accepted_pairs(accepted_text)
        if accepted_text.upper() in SPECIAL_PAIR_VALUES:
            continue

        source = terma_map.loc[review_row.sentence_id]
        aspects = parse_json_list(source["aspects"])
        opinions = parse_json_list(source["opinions"])
        pattern = _relation_pattern(pairs)

        for a_idx, o_idx in pairs:
            aspect = aspects[a_idx]
            opinion = opinions[o_idx]
            rows.append(
                {
                    "pair_id": f"{review_row.sentence_id}__A{a_idx + 1}__O{o_idx + 1}",
                    "sentence_id": review_row.sentence_id,
                    "source_split": review_row.source_split,
                    "review_scope": review_row.review_scope,
                    "sentence": source["sentence"],
                    "aspect_id": f"A{a_idx + 1}",
                    "aspect_text": aspect["text"],
                    "aspect_start": int(aspect["start"]),
                    "aspect_end": int(aspect["end"]),
                    "opinion_id": f"O{o_idx + 1}",
                    "opinion_text": opinion["text"],
                    "opinion_start": int(opinion["start"]),
                    "opinion_end": int(opinion["end"]),
                    "relation_pattern": pattern,
                    # True syntactic relation tags are added after parser evidence / manual check.
                    "relation_type": "",
                    "secondary_tags": "",
                }
            )

    columns = [
        "pair_id",
        "sentence_id",
        "source_split",
        "review_scope",
        "sentence",
        "aspect_id",
        "aspect_text",
        "aspect_start",
        "aspect_end",
        "opinion_id",
        "opinion_text",
        "opinion_start",
        "opinion_end",
        "relation_pattern",
        "relation_type",
        "secondary_tags",
    ]
    gold = pd.DataFrame(rows, columns=columns)
    if not gold.empty:
        gold = gold.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
    return gold


def validate_gold_pairs(
    gold_df: pd.DataFrame,
    terma_df: pd.DataFrame,
    review_df: pd.DataFrame | None = None,
) -> dict:
    """Validate pair keys, span boundaries, and text reconstruction."""
    required = {
        "pair_id",
        "sentence_id",
        "source_split",
        "aspect_text",
        "aspect_start",
        "aspect_end",
        "opinion_text",
        "opinion_start",
        "opinion_end",
    }
    missing = required - set(gold_df.columns)
    if missing:
        raise GoldValidationError(f"gold_pairs.csv kurang kolom: {sorted(missing)}")

    if gold_df.empty:
        raise GoldValidationError("gold_pairs.csv kosong")

    if gold_df["pair_id"].duplicated().any():
        examples = gold_df.loc[gold_df["pair_id"].duplicated(keep=False), "pair_id"].head(10).tolist()
        raise GoldValidationError(f"Duplicate pair_id: {examples}")

    check_cols = [
        "pair_id",
        "sentence_id",
        "aspect_text",
        "aspect_start",
        "aspect_end",
        "opinion_text",
        "opinion_start",
        "opinion_end",
    ]
    if gold_df[check_cols].isna().any().any():
        raise GoldValidationError("Missing value ditemukan pada kolom inti gold_pairs")

    terma_map = terma_df.set_index("sentence_id", drop=False)
    unknown = sorted(set(gold_df["sentence_id"]) - set(terma_map.index))
    if unknown:
        raise GoldValidationError(f"sentence_id gold tidak ada di TermA: {unknown[:10]}")

    for row in gold_df.itertuples(index=False):
        source = terma_map.loc[row.sentence_id]
        tokens = parse_json_list(source["tokens"])

        if str(row.source_split) != str(source["source_split"]):
            raise GoldValidationError(f"source_split mismatch pada {row.pair_id}")

        a_start, a_end = int(row.aspect_start), int(row.aspect_end)
        o_start, o_end = int(row.opinion_start), int(row.opinion_end)
        if not (0 <= a_start <= a_end < len(tokens)):
            raise GoldValidationError(f"Aspect span invalid pada {row.pair_id}")
        if not (0 <= o_start <= o_end < len(tokens)):
            raise GoldValidationError(f"Opinion span invalid pada {row.pair_id}")

        aspect_text = " ".join(tokens[a_start : a_end + 1])
        opinion_text = " ".join(tokens[o_start : o_end + 1])
        if aspect_text != str(row.aspect_text):
            raise GoldValidationError(
                f"Aspect text/span mismatch {row.pair_id}: {row.aspect_text!r} != {aspect_text!r}"
            )
        if opinion_text != str(row.opinion_text):
            raise GoldValidationError(
                f"Opinion text/span mismatch {row.pair_id}: {row.opinion_text!r} != {opinion_text!r}"
            )

    if review_df is not None:
        summary = validate_review_log(review_df, terma_df, require_reviewed=False)
        if int(summary["expected_pair_count"]) != len(gold_df):
            raise GoldValidationError(
                "Jumlah gold_pairs tidak sama dengan jumlah accepted pair pada review: "
                f"expected={summary['expected_pair_count']}, actual={len(gold_df)}"
            )

    return {
        "gold_pairs": int(len(gold_df)),
        "duplicate_pair_id": 0,
        "missing_core_values": 0,
        "span_mismatch": 0,
        "status": "PASS",
    }


def build_split_frames(terma_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Freeze TermA train as development and validation as final local test."""
    required = {"sentence_id", "source_split", "sentence"}
    missing = required - set(terma_df.columns)
    if missing:
        raise GoldValidationError(f"terma_processed.csv kurang kolom: {sorted(missing)}")

    cols = [
        c
        for c in [
            "sentence_id",
            "source_split",
            "sentence",
            "n_aspects",
            "n_opinions",
            "labels_available",
        ]
        if c in terma_df.columns
    ]

    development = terma_df.loc[terma_df["source_split"] == "train", cols].copy()
    final_test = terma_df.loc[terma_df["source_split"] == "validation", cols].copy()

    if len(development) != 3000:
        raise GoldValidationError(
            f"Expected 3000 TermA train rows, found {len(development)}"
        )
    if len(final_test) != 1000:
        raise GoldValidationError(
            f"Expected 1000 TermA validation rows, found {len(final_test)}"
        )

    if set(development["sentence_id"]) & set(final_test["sentence_id"]):
        raise GoldValidationError("Development dan final_test overlap")

    metadata = {
        "development": {
            "source": "TermA train",
            "n_sentences": int(len(development)),
            "purpose": "rule development, debugging, and tuning",
            "locked": False,
        },
        "final_test": {
            "source": "TermA validation",
            "n_sentences": int(len(final_test)),
            "purpose": "locked final local evaluation",
            "locked": True,
        },
        "official_test": {
            "source": "TermA official test",
            "purpose": "prediction only; labels are not used for local gold metrics",
            "local_gold_metrics": False,
        },
        "random_seed": 42,
    }
    return development, final_test, metadata


def save_split_files(terma_df: pd.DataFrame, split_dir: str | Path) -> dict:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    development, final_test, metadata = build_split_frames(terma_df)

    development.to_csv(split_dir / "development.csv", index=False, encoding="utf-8-sig")
    final_test.to_csv(split_dir / "final_test.csv", index=False, encoding="utf-8-sig")
    with (split_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


_NEGATION_CUES = {
    "tidak",
    "tak",
    "bukan",
    "belum",
    "tanpa",
    "kurang",
    "gak",
    "ga",
    "nggak",
    "enggak",
}
_INTENSIFIER_CUES = {
    "sangat",
    "banget",
    "sekali",
    "amat",
    "terlalu",
    "agak",
    "cukup",
    "lumayan",
}


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    words = set(str(text).lower().split())
    return bool(words & set(cues))


def _assign_polarity_stratum(row: pd.Series) -> str:
    text = f"{row.get('sentence', '')} {row.get('opinion_text', '')}".lower()
    if _contains_any(text, _NEGATION_CUES):
        return "NEGATION"
    if _contains_any(text, _INTENSIFIER_CUES):
        return "INTENSIFIER"
    if str(row.get("relation_pattern", "")) != "ONE_TO_ONE":
        return "STRUCTURAL_MULTI"
    return "NORMAL"


def _stratified_sample(
    df: pd.DataFrame,
    n_total: int,
    *,
    seed: int,
) -> pd.DataFrame:
    if n_total <= 0 or df.empty:
        return df.head(0).copy()
    if len(df) <= n_total:
        return df.copy()

    strata = ["NEGATION", "INTENSIFIER", "STRUCTURAL_MULTI", "NORMAL"]
    base = n_total // len(strata)
    remainder = n_total % len(strata)
    chosen = []
    used_index = set()

    for i, stratum in enumerate(strata):
        group = df[df["polarity_stratum"] == stratum]
        quota = base + (1 if i < remainder else 0)
        n = min(quota, len(group))
        if n:
            sample = group.sample(n=n, random_state=seed + i)
            chosen.append(sample)
            used_index.update(sample.index.tolist())

    result = pd.concat(chosen, axis=0) if chosen else df.head(0).copy()
    remaining_needed = n_total - len(result)
    if remaining_needed > 0:
        pool = df.loc[~df.index.isin(used_index)]
        n = min(remaining_needed, len(pool))
        if n:
            result = pd.concat(
                [result, pool.sample(n=n, random_state=seed + 100)],
                axis=0,
            )
    return result.copy()


def build_polarity_template(
    gold_pairs_df: pd.DataFrame,
    *,
    final_sample_size: int = 600,
    dev_sample_size: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a fixed, stratified polarity annotation subset.

    TermA supplies sentiment spans, not a positive/negative/neutral class.
    Therefore polarity is evaluated on a manually reviewed subset rather than
    silently assigning heuristic labels to every pair. Sampling is deterministic
    and stratified by negation, intensifier, structural multi-relation, and normal cases.
    """
    if gold_pairs_df.empty:
        raise GoldValidationError("gold_pairs_df kosong")

    df = gold_pairs_df.copy()
    df["polarity_stratum"] = df.apply(_assign_polarity_stratum, axis=1)
    cue_text = (df["sentence"].fillna("") + " " + df["opinion_text"].fillna("")).str.lower()
    df["has_negation_cue"] = cue_text.apply(lambda x: _contains_any(x, _NEGATION_CUES))
    df["has_intensifier_cue"] = cue_text.apply(lambda x: _contains_any(x, _INTENSIFIER_CUES))

    final_pool = df[df["source_split"] == "validation"].copy()
    dev_pool = df[df["source_split"] == "train"].copy()

    final_rows = _stratified_sample(
        final_pool, final_sample_size, seed=seed
    )
    final_rows["polarity_scope"] = "FINAL_POLARITY"

    dev_rows = _stratified_sample(
        dev_pool, dev_sample_size, seed=seed + 1000
    )
    dev_rows["polarity_scope"] = "DEV_POLARITY"

    result = pd.concat([final_rows, dev_rows], ignore_index=True)
    result = result.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
    result["gold_polarity"] = ""
    result["annotation_status"] = ""
    result["notes"] = ""

    cols = [
        "pair_id",
        "sentence_id",
        "source_split",
        "polarity_scope",
        "polarity_stratum",
        "sentence",
        "aspect_text",
        "opinion_text",
        "relation_pattern",
        "has_negation_cue",
        "has_intensifier_cue",
        "gold_polarity",
        "annotation_status",
        "notes",
    ]
    return result[cols]

def validate_polarity_gold(df: pd.DataFrame, *, require_reviewed: bool = True) -> dict:
    required = {"pair_id", "gold_polarity", "annotation_status"}
    missing = required - set(df.columns)
    if missing:
        raise GoldValidationError(f"gold_polarity.csv kurang kolom: {sorted(missing)}")
    if df["pair_id"].duplicated().any():
        raise GoldValidationError("Duplicate pair_id pada gold_polarity.csv")

    polarities = set(df["gold_polarity"].fillna("").astype(str).str.strip().str.upper())
    bad = polarities - VALID_POLARITIES
    if bad:
        raise GoldValidationError(
            f"gold_polarity belum lengkap/invalid. Nilai yang tidak valid: {sorted(bad)}"
        )
    if require_reviewed:
        statuses = set(df["annotation_status"].fillna("").astype(str).str.strip())
        if statuses != {"REVIEWED"}:
            raise GoldValidationError(
                f"annotation_status polarity harus REVIEWED semua. Ditemukan: {sorted(statuses)}"
            )
    return {"polarity_rows": int(len(df)), "status": "PASS"}


def build_implicit_template(
    terma_df: pd.DataFrame,
    *,
    dev_sample_size: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create an implicit-aspect annotation subset.

    All validation orphan opinions are retained for final evaluation. A fixed
    train sample is retained for development. One row corresponds to one opinion
    span, because a sentence can contain multiple orphan opinions.
    """
    required = {
        "sentence_id",
        "source_split",
        "sentence",
        "aspects",
        "opinions",
        "labels_available",
    }
    missing = required - set(terma_df.columns)
    if missing:
        raise GoldValidationError(f"terma_processed.csv kurang kolom: {sorted(missing)}")

    rows: list[dict] = []
    for source in terma_df.itertuples(index=False):
        if not bool(source.labels_available):
            continue
        aspects = parse_json_list(source.aspects)
        opinions = parse_json_list(source.opinions)
        if aspects or not opinions:
            continue

        for idx, opinion in enumerate(opinions, start=1):
            rows.append(
                {
                    "implicit_id": f"{source.sentence_id}__O{idx}",
                    "sentence_id": source.sentence_id,
                    "source_split": source.source_split,
                    "sentence": source.sentence,
                    "opinion_id": f"O{idx}",
                    "opinion_text": opinion["text"],
                    "opinion_start": int(opinion["start"]),
                    "opinion_end": int(opinion["end"]),
                }
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    final_rows = candidates[candidates["source_split"] == "validation"].copy()
    final_rows["implicit_scope"] = "FINAL_IMPLICIT"

    dev_pool = candidates[candidates["source_split"] == "train"].copy()
    n_dev = min(dev_sample_size, len(dev_pool))
    dev_rows = (
        dev_pool.sample(n=n_dev, random_state=seed).copy()
        if n_dev
        else dev_pool.head(0).copy()
    )
    dev_rows["implicit_scope"] = "DEV_IMPLICIT"

    result = pd.concat([final_rows, dev_rows], ignore_index=True)
    result["decision"] = ""
    result["gold_aspect_category"] = ""
    result["gold_polarity"] = ""
    result["annotation_status"] = ""
    result["notes"] = ""
    return result[
        [
            "implicit_id",
            "sentence_id",
            "source_split",
            "implicit_scope",
            "sentence",
            "opinion_id",
            "opinion_text",
            "opinion_start",
            "opinion_end",
            "decision",
            "gold_aspect_category",
            "gold_polarity",
            "annotation_status",
            "notes",
        ]
    ]

def validate_implicit_gold(df: pd.DataFrame, *, require_reviewed: bool = True) -> dict:
    required = {
        "implicit_id",
        "decision",
        "gold_aspect_category",
        "annotation_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise GoldValidationError(f"gold_implicit.csv kurang kolom: {sorted(missing)}")
    if df["implicit_id"].duplicated().any():
        raise GoldValidationError("Duplicate implicit_id pada gold_implicit.csv")

    decisions = df["decision"].fillna("").astype(str).str.strip().str.upper()
    bad_decisions = set(decisions) - VALID_IMPLICIT_DECISIONS
    if bad_decisions:
        raise GoldValidationError(
            f"decision implicit belum lengkap/invalid: {sorted(bad_decisions)}"
        )

    implicit_rows = df[decisions == "IMPLICIT"].copy()
    categories = implicit_rows["gold_aspect_category"].fillna("").astype(str).str.strip()
    bad_categories = sorted(set(categories) - set(IMPLICIT_CATEGORIES))
    if bad_categories:
        raise GoldValidationError(
            f"Kategori implicit di luar ontology: {bad_categories}"
        )
    if (categories == "").any():
        raise GoldValidationError("Row IMPLICIT harus memiliki gold_aspect_category")

    if require_reviewed:
        statuses = set(df["annotation_status"].fillna("").astype(str).str.strip())
        if statuses != {"REVIEWED"}:
            raise GoldValidationError(
                f"annotation_status implicit harus REVIEWED semua. Ditemukan: {sorted(statuses)}"
            )

    return {
        "implicit_candidates": int(len(df)),
        "implicit_gold_rows": int((decisions == "IMPLICIT").sum()),
        "status": "PASS",
    }


def h2_status(project_root: str | Path) -> dict:
    """Return a lightweight file-level progress summary for H2."""
    root = Path(project_root)
    files = {
        "terma_processed": root / "data" / "processed" / "terma_processed.csv",
        "hoasa_processed": root / "data" / "processed" / "hoasa_processed.csv",
        "sentence_metadata": root / "data" / "processed" / "sentence_metadata.csv",
        "gold_review_log": root / "data" / "gold" / "gold_review_log.csv",
        "gold_pairs": root / "data" / "gold" / "gold_pairs.csv",
        "gold_polarity": root / "data" / "gold" / "gold_polarity.csv",
        "gold_implicit": root / "data" / "gold" / "gold_implicit.csv",
        "development": root / "data" / "splits" / "development.csv",
        "final_test": root / "data" / "splits" / "final_test.csv",
        "split_metadata": root / "data" / "splits" / "split_metadata.json",
    }
    return {name: path.exists() for name, path in files.items()}


# ============================================================
# AUTO 1A1O (development only)
# ============================================================

AUTO_STATUS = "AUTO"
VALID_AUTO_STATUSES = {AUTO_STATUS, "REVIEWED"}


def build_auto_1a1o_sheet(
    terma_df: pd.DataFrame,
    review_df: pd.DataFrame,
    *,
    split: str = "train",
) -> pd.DataFrame:
    """
    Development sentences with exactly one aspect and one opinion that are NOT
    in gold_review_log.csv. Their only possible pairing (A1-O1) is pre-filled
    with annotation_status=AUTO; a human verifies each row and sets REVIEWED,
    or changes accepted_pairs to NONE/AMBIGUOUS.

    Only the development split is allowed: final-test gold must be 100% reviewed.
    """
    if split != "train":
        raise GoldValidationError("AUTO 1A1O hanya untuk development (TermA train)")

    reviewed_ids = set(review_df["sentence_id"])
    pool = terma_df[
        (terma_df["source_split"] == split)
        & (terma_df["n_aspects"] == 1)
        & (terma_df["n_opinions"] == 1)
        & ~terma_df["sentence_id"].isin(reviewed_ids)
    ]
    rows = []
    for row in pool.itertuples(index=False):
        aspect = parse_json_list(row.aspects)[0]
        opinion = parse_json_list(row.opinions)[0]
        rows.append({
            "sentence_id": row.sentence_id,
            "source_split": row.source_split,
            "sentence": row.sentence,
            "aspects": f"A1={aspect['text']}",
            "opinions": f"O1={opinion['text']}",
            "accepted_pairs": "A1-O1",
            "review_scope": "DEV_AUTO_1A1O",
            "annotation_status": AUTO_STATUS,
            "notes": "",
        })
    return pd.DataFrame(rows, columns=[
        "sentence_id", "source_split", "sentence", "aspects", "opinions",
        "accepted_pairs", "review_scope", "annotation_status", "notes",
    ])


def validate_auto_1a1o(auto_df: pd.DataFrame, terma_df: pd.DataFrame, review_df: pd.DataFrame) -> dict:
    """Validate the AUTO 1A1O sheet; it must not overlap the review log or touch the final split."""
    statuses = set(auto_df["annotation_status"].fillna("").astype(str).str.strip())
    bad = statuses - VALID_AUTO_STATUSES
    if bad:
        raise GoldValidationError(f"annotation_status AUTO 1A1O invalid: {sorted(bad)}")
    if set(auto_df["source_split"]) - {"train"}:
        raise GoldValidationError("AUTO 1A1O memuat kalimat di luar development split")
    overlap = set(auto_df["sentence_id"]) & set(review_df["sentence_id"])
    if overlap:
        raise GoldValidationError(f"AUTO 1A1O overlap dengan gold_review_log: {sorted(overlap)[:10]}")

    # Reuse the review-log checks (format A1-O1 / NONE / AMBIGUOUS, index range, split match).
    summary = validate_review_log(auto_df, terma_df, require_reviewed=False)
    for value in auto_df["accepted_pairs"].astype(str).str.strip().str.upper():
        if value not in {"A1-O1", "NONE", "AMBIGUOUS"}:
            raise GoldValidationError(f"accepted_pairs 1A1O hanya boleh A1-O1/NONE/AMBIGUOUS, ditemukan {value!r}")

    counts = auto_df["annotation_status"].str.strip().value_counts().to_dict()
    return {
        "rows": int(len(auto_df)),
        "auto": int(counts.get(AUTO_STATUS, 0)),
        "reviewed": int(counts.get("REVIEWED", 0)),
        "pairs": summary["expected_pair_count"],
        "status": "PASS",
    }
