"""
Analisis oracle untuk pasangan gold di atas pohon dependensi (H3.2).

Menjawab: seberapa jauh head aspek dan head opini gold terpisah pada pohon
UDPipe, lewat pola jalur apa, dan berapa recall MAKSIMUM yang mungkin dicapai
rules yang hanya menjangkau jarak <= k. Hasilnya dipakai untuk memutuskan
rule mana yang perlu diperbaiki di H4 -- HANYA pada split development.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .parser import ParsedSentence

SOURCE_REVIEWED = "REVIEWED"
SOURCE_AUTO_1A1O = "AUTO_1A1O"


@dataclass
class OracleRecord:
    sentence_id: str
    source: str
    aspect_text: str
    opinion_text: str
    aspect_head: int
    opinion_head: int
    aspect_head_pos: str
    opinion_head_pos: str
    distance: Optional[int]
    pattern: str


def _depth(sent: ParsedSentence, idx: int) -> int:
    depth = 0
    seen = set()
    while sent.tokens[idx].head_idx != idx and idx not in seen:
        seen.add(idx)
        idx = sent.tokens[idx].head_idx
        depth += 1
    return depth


def span_head(sent: ParsedSentence, start: int, end: int) -> int:
    """
    Head sintaksis span [start, end]: token di span yang head-nya berada di
    luar span (atau root). Jika lebih dari satu, ambil yang paling dekat ke root,
    lalu yang paling kanan (frasa nomina Indonesia umumnya head-initial,
    tetapi frasa opini seperti "tidak stabil" ber-head di kanan).
    """
    candidates = [
        i for i in range(start, end + 1)
        if not (start <= sent.tokens[i].head_idx <= end) or sent.tokens[i].head_idx == i
    ]
    if not candidates:  # siklus tidak mungkin pada pohon valid; jaga-jaga
        candidates = list(range(start, end + 1))
    return min(candidates, key=lambda i: (_depth(sent, i), -i))


def _ancestors(sent: ParsedSentence, idx: int) -> List[int]:
    chain = [idx]
    while sent.tokens[chain[-1]].head_idx != chain[-1]:
        chain.append(sent.tokens[chain[-1]].head_idx)
    return chain


def dependency_path(sent: ParsedSentence, src: int, dst: int) -> Optional[List[str]]:
    """
    Jalur src -> dst sebagai daftar langkah: "^dep" = naik dari anak ke head
    (dep milik anak), "v dep" = turun dari head ke anak. None jika beda pohon.
    """
    up = _ancestors(sent, src)
    down = _ancestors(sent, dst)
    common = next((a for a in up if a in set(down)), None)
    if common is None:
        return None
    steps = [f"^{sent.tokens[i].dep}" for i in up[: up.index(common)]]
    steps += [f"v{sent.tokens[i].dep}" for i in reversed(down[: down.index(common)])]
    return steps


def pair_pattern(sent: ParsedSentence, aspect_head: int, opinion_head: int) -> str:
    if aspect_head == opinion_head:
        return "SAME_TOKEN"
    path = dependency_path(sent, aspect_head, opinion_head)
    if path is None:
        return "DISCONNECTED"
    a_pos = sent.tokens[aspect_head].pos
    o_pos = sent.tokens[opinion_head].pos
    return f"{a_pos} {' '.join(path)} {o_pos}"


def build_dev_gold(terma_df: pd.DataFrame, gold_pairs_df: pd.DataFrame,
                   review_df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """
    Pasangan gold untuk analisis: pasangan hasil review (gold_pairs.csv) +
    pasangan trivial dari kalimat 1-aspek-1-opini yang tidak direview.
    Kolom `source` membedakan keduanya agar bisa dilaporkan terpisah.
    """
    cols = ["sentence_id", "aspect_text", "aspect_start", "aspect_end",
            "opinion_text", "opinion_start", "opinion_end"]
    reviewed = gold_pairs_df.loc[gold_pairs_df["source_split"] == split, cols].copy()
    reviewed["source"] = SOURCE_REVIEWED

    reviewed_ids = set(review_df["sentence_id"])
    pool = terma_df[(terma_df["source_split"] == split)
                    & (terma_df["n_aspects"] == 1) & (terma_df["n_opinions"] == 1)
                    & ~terma_df["sentence_id"].isin(reviewed_ids)]
    auto_rows = []
    for row in pool.itertuples(index=False):
        a = json.loads(row.aspects)[0]
        o = json.loads(row.opinions)[0]
        auto_rows.append({
            "sentence_id": row.sentence_id,
            "aspect_text": a["text"], "aspect_start": a["start"], "aspect_end": a["end"],
            "opinion_text": o["text"], "opinion_start": o["start"], "opinion_end": o["end"],
            "source": SOURCE_AUTO_1A1O,
        })
    auto = pd.DataFrame(auto_rows, columns=cols + ["source"])
    return pd.concat([reviewed, auto], ignore_index=True)


def oracle_records(gold_df: pd.DataFrame, parses: Dict[str, ParsedSentence]) -> List[OracleRecord]:
    from .parser import dependency_distance

    records = []
    for g in gold_df.itertuples(index=False):
        sent = parses[g.sentence_id]
        a_head = span_head(sent, int(g.aspect_start), int(g.aspect_end))
        o_head = span_head(sent, int(g.opinion_start), int(g.opinion_end))
        records.append(OracleRecord(
            sentence_id=g.sentence_id,
            source=g.source,
            aspect_text=g.aspect_text,
            opinion_text=g.opinion_text,
            aspect_head=a_head,
            opinion_head=o_head,
            aspect_head_pos=sent.tokens[a_head].pos,
            opinion_head_pos=sent.tokens[o_head].pos,
            distance=dependency_distance(sent, a_head, o_head),
            pattern=pair_pattern(sent, a_head, o_head),
        ))
    return records


def rule_reach(gold_df: pd.DataFrame, parses: Dict[str, ParsedSentence], extractor) -> List[bool]:
    """
    Untuk tiap pasangan gold: apakah `extractor(sent)` menghasilkan pasangan
    yang head aspek DAN head opininya jatuh di dalam span gold (head match).
    Ini pratinjau kasar, bukan metrik resmi (metrik resmi di H3.3).
    """
    cache: Dict[str, list] = {}
    hits = []
    for g in gold_df.itertuples(index=False):
        if g.sentence_id not in cache:
            cache[g.sentence_id] = extractor(parses[g.sentence_id])
        hits.append(any(
            int(g.aspect_start) <= p.aspect_idx <= int(g.aspect_end)
            and int(g.opinion_start) <= p.opinion_idx <= int(g.opinion_end)
            for p in cache[g.sentence_id]
        ))
    return hits


def summarize(records: Sequence[OracleRecord], max_k: int = 4, top_patterns: int = 25) -> dict:
    n = len(records)
    dist = Counter("None" if r.distance is None else str(r.distance) for r in records)
    cumulative = {
        f"<={k}": round(sum(1 for r in records if r.distance is not None and r.distance <= k) / n, 4)
        for k in range(1, max_k + 1)
    } if n else {}
    patterns = Counter(r.pattern for r in records).most_common(top_patterns)
    running = 0
    pattern_rows = []
    for pattern, count in patterns:
        running += count
        pattern_rows.append({"pattern": pattern, "count": count,
                             "share": round(count / n, 4), "cumulative": round(running / n, 4)})
    return {
        "n_pairs": n,
        "aspect_head_pos": dict(Counter(r.aspect_head_pos for r in records).most_common()),
        "opinion_head_pos": dict(Counter(r.opinion_head_pos for r in records).most_common()),
        "distance_distribution": dict(sorted(
            dist.items(), key=lambda kv: (kv[0] == "None", int(kv[0]) if kv[0] != "None" else 0)
        )),
        "recall_upper_bound_by_distance": cumulative,
        "top_patterns": pattern_rows,
    }
