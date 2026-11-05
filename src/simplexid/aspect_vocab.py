"""
Kosakata head aspek, dibangun HANYA dari span aspek TermA train (H4.3).

Karena kosakata berasal dari split yang juga dipakai mengukur di dev, pengukuran
dev memakai cross-fit: kalimat di lipatan k dinilai dengan kosakata dari lipatan
lain. Sistem final (dan final test) memakai kosakata dari seluruh train.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Set

import pandas as pd

from .oracle import span_head
from .parser import ParsedSentence

N_FOLDS = 5
MIN_COUNT = 2
VOCAB_PATH = Path(__file__).resolve().parents[2] / "resources" / "aspects" / "aspect_vocab_v1.tsv"


def fold_of(sentence_id: str, n_folds: int = N_FOLDS) -> int:
    return zlib.crc32(sentence_id.encode("utf-8")) % n_folds


def count_aspect_heads(terma_df: pd.DataFrame, parses: Dict[str, ParsedSentence],
                       exclude: Optional[Iterable[str]] = None) -> Counter:
    """Hitung bentuk permukaan head aspek; final/test split tidak pernah boleh masuk."""
    train = terma_df[terma_df["source_split"] == "train"]
    excluded: Set[str] = set(exclude or ())
    counts: Counter = Counter()
    for row in train.itertuples(index=False):
        if row.sentence_id in excluded:
            continue
        sent = parses[row.sentence_id]
        for aspect in json.loads(row.aspects):
            head = sent.tokens[span_head(sent, int(aspect["start"]), int(aspect["end"]))]
            counts[head.text.lower()] += 1
    return counts


def vocabulary(counts: Counter, min_count: int = MIN_COUNT) -> FrozenSet[str]:
    return frozenset(word for word, n in counts.items() if n >= min_count)


def crossfit_vocabularies(terma_df: pd.DataFrame, parses: Dict[str, ParsedSentence],
                          n_folds: int = N_FOLDS) -> Dict[int, FrozenSet[str]]:
    train_ids = terma_df.loc[terma_df["source_split"] == "train", "sentence_id"]
    folds: Dict[int, FrozenSet[str]] = {}
    for k in range(n_folds):
        held_out = [s for s in train_ids if fold_of(s, n_folds) == k]
        folds[k] = vocabulary(count_aspect_heads(terma_df, parses, exclude=held_out))
    return folds


def write_vocabulary(counts: Counter, path: Path = VOCAB_PATH, min_count: int = MIN_COUNT) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(((w, n) for w, n in counts.items() if n >= min_count), key=lambda x: (-x[1], x[0]))
    path.write_text("word\tcount\n" + "".join(f"{w}\t{n}\n" for w, n in rows), encoding="utf-8")
    return len(rows)


def load_vocabulary(path: Path = VOCAB_PATH) -> FrozenSet[str]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return frozenset(line.split("\t")[0] for line in lines if line.strip())
