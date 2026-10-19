"""
Memuat gold evaluasi per split (H3.3).

Universe kalimat yang dievaluasi:
- final (TermA validation): SEMUA kalimat kecuali yang direview AMBIGUOUS.
  Setiap kalimat yang memiliki aspek & opini wajib sudah direview.
- development (TermA train): kalimat yang direview + AUTO 1A1O + kalimat tanpa
  aspek/opini (gold kosong). Kalimat kompleks yang belum direview DIKELUARKAN,
  karena prediksi di sana tidak bisa dinilai benar/salah.

Setiap kalimat diberi strata (EMPTY / SIMPLE / COMPLEX) dan sumber gold
(REVIEWED / AUTO_1A1O / NO_SPANS) agar metrik bisa dilaporkan terpisah.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..gold import GoldValidationError, parse_json_list

SPLITS = {"dev": "train", "final": "validation"}

SOURCE_REVIEWED = "REVIEWED"
SOURCE_AUTO = "AUTO_1A1O"
SOURCE_NO_SPANS = "NO_SPANS"

STRATUM_EMPTY = "EMPTY"      # gold tanpa pasangan (tidak ada aspek/opini, atau NONE)
STRATUM_SIMPLE = "SIMPLE"    # tepat 1 aspek + 1 opini
STRATUM_COMPLEX = "COMPLEX"  # lainnya


@dataclass(frozen=True)
class GoldPairSpan:
    pair_id: str
    aspect_span: Tuple[int, int]
    opinion_span: Tuple[int, int]


@dataclass
class EvalGold:
    split: str
    sentence_ids: List[str]
    tokens: Dict[str, List[str]]
    pairs: Dict[str, List[GoldPairSpan]]
    source: Dict[str, str]
    stratum: Dict[str, str]
    version: str = ""
    excluded: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "split": self.split,
            "version": self.version,
            "n_sentences": len(self.sentence_ids),
            "n_pairs": sum(len(self.pairs[s]) for s in self.sentence_ids),
            "by_source": pd.Series(self.source).value_counts().to_dict(),
            "by_stratum": pd.Series(self.stratum).value_counts().to_dict(),
            "excluded": self.excluded,
        }


def _version(gold: EvalGold) -> str:
    payload = [
        [sid, gold.source[sid], gold.stratum[sid], [asdict(p) for p in gold.pairs[sid]]]
        for sid in gold.sentence_ids
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _stratum(n_aspects: int, n_opinions: int, n_pairs: int) -> str:
    if n_pairs == 0:
        return STRATUM_EMPTY
    if n_aspects == 1 and n_opinions == 1:
        return STRATUM_SIMPLE
    return STRATUM_COMPLEX


def build_eval_gold(
    split: str,
    terma_df: pd.DataFrame,
    review_df: pd.DataFrame,
    gold_pairs_df: pd.DataFrame,
    auto_df: pd.DataFrame | None = None,
) -> EvalGold:
    if split not in SPLITS:
        raise ValueError(f"split harus salah satu dari {sorted(SPLITS)}")
    source_split = SPLITS[split]
    terma = terma_df[terma_df["source_split"] == source_split]

    decisions: Dict[str, Tuple[str, str]] = {}  # sid -> (accepted_upper, source)
    for row in review_df[review_df["source_split"] == source_split].itertuples(index=False):
        decisions[row.sentence_id] = (str(row.accepted_pairs).strip().upper(), SOURCE_REVIEWED)
    if split == "dev" and auto_df is not None:
        for row in auto_df.itertuples(index=False):
            source = SOURCE_REVIEWED if str(row.annotation_status).strip() == "REVIEWED" else SOURCE_AUTO
            decisions[row.sentence_id] = (str(row.accepted_pairs).strip().upper(), source)

    reviewed_pairs: Dict[str, List[GoldPairSpan]] = {}
    for g in gold_pairs_df[gold_pairs_df["source_split"] == source_split].itertuples(index=False):
        reviewed_pairs.setdefault(g.sentence_id, []).append(GoldPairSpan(
            pair_id=g.pair_id,
            aspect_span=(int(g.aspect_start), int(g.aspect_end)),
            opinion_span=(int(g.opinion_start), int(g.opinion_end)),
        ))

    gold = EvalGold(split=split, sentence_ids=[], tokens={}, pairs={}, source={}, stratum={})
    excluded = {"ambiguous": 0, "unreviewed_complex": 0}

    for row in terma.itertuples(index=False):
        sid = row.sentence_id
        aspects = parse_json_list(row.aspects)
        opinions = parse_json_list(row.opinions)
        has_spans = bool(aspects) and bool(opinions)

        if not has_spans:
            pairs, source = [], SOURCE_NO_SPANS
        elif sid in decisions:
            accepted, source = decisions[sid]
            if accepted == "AMBIGUOUS":
                excluded["ambiguous"] += 1
                continue
            if accepted == "NONE":
                pairs = []
            elif sid in reviewed_pairs:
                pairs = reviewed_pairs[sid]
            elif accepted == "A1-O1" and len(aspects) == 1 and len(opinions) == 1:
                a, o = aspects[0], opinions[0]
                pairs = [GoldPairSpan(
                    pair_id=f"{sid}__A1__O1",
                    aspect_span=(int(a["start"]), int(a["end"])),
                    opinion_span=(int(o["start"]), int(o["end"])),
                )]
            else:
                raise GoldValidationError(
                    f"{sid}: accepted_pairs={accepted} tetapi pasangan tidak ada di gold_pairs.csv"
                )
        elif split == "final":
            raise GoldValidationError(
                f"{sid}: kalimat final dengan aspek & opini belum direview -- gold final harus lengkap"
            )
        else:
            excluded["unreviewed_complex"] += 1
            continue

        gold.sentence_ids.append(sid)
        gold.tokens[sid] = json.loads(row.tokens)
        gold.pairs[sid] = pairs
        gold.source[sid] = source
        gold.stratum[sid] = _stratum(len(aspects), len(opinions), len(pairs))

    gold.excluded = excluded
    gold.version = _version(gold)
    return gold


def load_eval_gold(split: str, project_root: str | Path) -> EvalGold:
    root = Path(project_root)
    terma = pd.read_csv(root / "data" / "processed" / "terma_processed.csv")
    review = pd.read_csv(root / "data" / "gold" / "gold_review_log.csv")
    gold_pairs = pd.read_csv(root / "data" / "gold" / "gold_pairs.csv")
    auto_path = root / "data" / "gold" / "gold_dev_auto_1a1o.csv"
    auto = pd.read_csv(auto_path) if auto_path.exists() else None
    return build_eval_gold(split, terma, review, gold_pairs, auto)
