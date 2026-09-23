"""
Loader untuk dataset TermA dan HoASA (benchmark IndoNLU) dari HuggingFace Datasets.

TermA: span-extraction (IOB) untuk aspek (ASP) dan sentimen/opini (SEN) --
       dipakai sebagai ground truth pasangan aspek-opini.
HoASA: 10 kategori aspek hotel eksplisit -- dipakai sebagai basis kamus DAM.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GoldPair:
    """Satu pasangan aspek-opini ground truth dari satu kalimat TermA."""
    sentence_id: str
    tokens: List[str]
    aspect_span: Tuple[int, int]     # (start_idx, end_idx) inclusive, index token
    opinion_span: Tuple[int, int]
    aspect_text: str
    opinion_text: str


def load_terma():
    """
    Mengunduh TermA via HuggingFace `datasets` dan mengonversi label IOB
    (format: B-ASPECT, I-ASPECT, B-SENTIMENT, I-SENTIMENT, O) menjadi
    daftar GoldPair per kalimat.

    Catatan: TermA memberi label span aspek dan span opini per kalimat, TAPI
    tidak secara eksplisit menyatakan pasangan mana-dengan-mana ketika ada
    lebih dari satu span aspek dan satu span opini dalam satu kalimat.
    Untuk kalimat dengan tepat 1 span aspek + 1 span opini, pemasangannya
    trivial. Untuk kalimat dengan >1 dari salah satu, perlu heuristik
    tambahan (mis. jarak token terdekat) atau verifikasi manual saat membangun
    tag sekunder (lihat secondary_tagging.py) -- ini SENGAJA diserahkan ke
    Anda karena keputusan pemasangan multi-span mempengaruhi validitas
    evaluasi Eksperimen A/B.
    """
    from datasets import load_dataset

    raw = load_dataset("indonlp/indonlu", "terma")
    splits = {}
    for split_name in ("train", "validation", "test"):
        pairs = []
        for i, ex in enumerate(raw[split_name]):
            tokens = ex["tokens"]
            tags = ex["seq_label"]  # sesuaikan nama field dgn skema aktual dataset
            aspect_spans = _extract_spans(tags, prefix="ASPECT")
            opinion_spans = _extract_spans(tags, prefix="SENTIMENT")
            for a_span, o_span in _pair_spans(aspect_spans, opinion_spans):
                pairs.append(GoldPair(
                    sentence_id=f"{split_name}-{i}",
                    tokens=tokens,
                    aspect_span=a_span,
                    opinion_span=o_span,
                    aspect_text=" ".join(tokens[a_span[0]:a_span[1]+1]),
                    opinion_text=" ".join(tokens[o_span[0]:o_span[1]+1]),
                ))
        splits[split_name] = pairs
    return splits


def _extract_spans(tags: List[str], prefix: str) -> List[Tuple[int, int]]:
    """Ekstrak span (start,end) dari label IOB untuk satu tipe entitas."""
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == f"B-{prefix}":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == f"I-{prefix}" and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def _pair_spans(aspect_spans, opinion_spans):
    """
    Heuristik pemasangan default: jika jumlah span aspek == jumlah span opini,
    pasangkan berdasarkan urutan kemunculan (index ke-i dengan index ke-i).
    Jika tidak sama, pasangkan tiap opini dengan aspek TERDEKAT (jarak token).

    PERINGATAN: ini adalah heuristik sederhana, bukan ground truth resmi.
    Untuk kalimat dgn struktur kompleks (kandidat R3/R4/R8), verifikasi manual
    tetap diperlukan -- lihat catatan di load_terma().
    """
    if not aspect_spans or not opinion_spans:
        return []
    if len(aspect_spans) == len(opinion_spans):
        return list(zip(aspect_spans, opinion_spans))
    pairs = []
    for o_span in opinion_spans:
        o_mid = (o_span[0] + o_span[1]) / 2
        nearest = min(aspect_spans, key=lambda a: abs((a[0]+a[1])/2 - o_mid))
        pairs.append((nearest, o_span))
    return pairs


def load_hoasa():
    """
    Mengunduh HoASA -- dipakai untuk membangun taksonomi 10 kategori aspek
    hotel sebagai basis kamus DAM (lihat dam.py).
    """
    from datasets import load_dataset
    return load_dataset("indonlp/indonlu", "hoasa")


HOASA_CATEGORIES = [
    "ac", "air_panas", "bau", "general", "kebersihan",
    "linen", "service", "sunrise_meal", "tv", "wifi",
]


if __name__ == "__main__":
    print("Mengunduh TermA...")
    terma = load_terma()
    for split, pairs in terma.items():
        print(f"  {split}: {len(pairs)} pasangan aspek-opini")

    print("Mengunduh HoASA...")
    hoasa = load_hoasa()
    for split in hoasa:
        print(f"  {split}: {len(hoasa[split])} ulasan")
