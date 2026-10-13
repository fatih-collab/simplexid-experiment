"""
Domain-Aspect Mapping (DAM) dengan disambiguasi kontekstual (R8 revisi).

Starter dictionary di bawah HANYA CONTOH AWAL untuk domain hotel berbasis
10 kategori HoASA -- WAJIB diperluas dari eksplorasi data TermA/HoASA
sesuai Bagian 4 rancangan eksperimen sebelum dipakai untuk hasil resmi.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


# kata_opini -> {domain -> [daftar kandidat kategori aspek implisit]}
# Starter kecil; perluas berdasarkan analisis frekuensi pada TermA.
DAM_DICT: Dict[str, Dict[str, List[str]]] = {
    "pengap":   {"hotel": ["ac"]},
    "dingin":   {"hotel": ["ac"]},
    "panas":    {"hotel": ["ac", "air_panas"]},
    "apek":     {"hotel": ["kebersihan", "bau"]},
    "bau":      {"hotel": ["bau", "kebersihan"]},
    "lemot":    {"hotel": ["wifi"]},
    "lambat":   {"hotel": ["wifi", "service"]},
    "kotor":    {"hotel": ["kebersihan", "linen"]},
    "mahal":    {"hotel": ["general", "sunrise_meal"]},   # contoh Gambar 3: HARGA_KAMAR vs HARGA_LAYANAN_TAMBAHAN
    "ramah":    {"hotel": ["service"]},
    "lambat_checkin": {"hotel": ["service"]},
    "buram":    {"hotel": ["tv"]},
    "kasar":    {"hotel": ["linen"]},
}


@dataclass
class DAMResult:
    opinion_word: str
    candidates: List[str]
    chosen: Optional[str]
    disambiguated: bool          # True jika dipilih via disambiguasi (>1 kandidat)


def lookup(opinion_lemma: str, domain: str) -> List[str]:
    """Kembalikan daftar kandidat kategori aspek implisit untuk kata opini pada domain tsb."""
    entry = DAM_DICT.get(opinion_lemma.lower())
    if not entry:
        return []
    return entry.get(domain, [])


def disambiguate(opinion_lemma: str, domain: str,
                  documented_aspects: List[str],
                  similarity_fn=None) -> DAMResult:
    """
    Terapkan R8 (revisi): jika >1 kandidat, pilih kandidat dengan kesesuaian
    tertinggi terhadap aspek-aspek yang SUDAH diekstraksi (documented_aspects)
    pada dokumen yang sama.

    similarity_fn: callable(kategori_a: str, kategori_b: str) -> float
        Default memakai co-occurrence sederhana (lihat _default_similarity);
        untuk hasil lebih baik, ganti dengan cosine similarity embedding
        (mis. FastText Bahasa Indonesia) -- lihat catatan di bawah.
    """
    candidates = lookup(opinion_lemma, domain)
    if not candidates:
        return DAMResult(opinion_lemma, [], None, False)
    if len(candidates) == 1:
        return DAMResult(opinion_lemma, candidates, candidates[0], False)

    sim = similarity_fn or _default_similarity
    if not documented_aspects:
        # tidak ada konteks dokumen utk membandingkan -> fallback kandidat pertama
        # (JANGAN gunakan ini sebagai keputusan final di eksperimen -- catat
        # sebagai kasus "tidak dapat didisambiguasi", lihat evaluate.py)
        return DAMResult(opinion_lemma, candidates, candidates[0], False)

    scored = []
    for cand in candidates:
        best = max(sim(cand, doc_asp) for doc_asp in documented_aspects)
        scored.append((cand, best))
    scored.sort(key=lambda x: -x[1])
    return DAMResult(opinion_lemma, candidates, scored[0][0], True)


# --- similarity co-occurrence sederhana sebagai fallback tanpa dependency embedding ---
_COOC_HINTS = {
    ("sunrise_meal", "general"): 0.3,
    ("kebersihan", "linen"): 0.6,
    ("ac", "air_panas"): 0.2,
}

# Kata kunci yang berasosiasi dengan tiap kategori HoASA -- dipakai fallback
# similarity ketika membandingkan kandidat kategori dengan TEKS aspek
# terdokumen (bukan kategori lain). WAJIB diperluas -- ini starter minimal.
CATEGORY_KEYWORDS = {
    "sunrise_meal": {"sarapan", "breakfast", "makan", "menu"},
    "ac": {"ac", "pendingin", "suhu", "kamar"},
    "air_panas": {"air", "shower", "mandi", "kamar_mandi"},
    "wifi": {"wifi", "internet", "koneksi", "sinyal"},
    "kebersihan": {"bersih", "kamar", "toilet"},
    "linen": {"seprai", "handuk", "bantal", "selimut"},
    "service": {"pelayanan", "staf", "resepsionis", "checkin", "checkout"},
    "tv": {"tv", "televisi", "channel"},
    "general": {"hotel", "penginapan", "kamar"},
    "bau": {"aroma", "wangi"},
}

def _default_similarity(cat_a: str, cat_b: str) -> float:
    if cat_a == cat_b:
        return 1.0
    direct = _COOC_HINTS.get((cat_a, cat_b)) or _COOC_HINTS.get((cat_b, cat_a))
    if direct:
        return direct
    # fallback: cocokkan kata kunci kategori terhadap teks aspek (cat_b di sini
    # sering berupa teks aspek terdokumen seperti "sarapan", bukan nama kategori)
    kw = CATEGORY_KEYWORDS.get(cat_a, set())
    b_tokens = set(cat_b.lower().replace("_", " ").split())
    overlap = kw & b_tokens
    return 0.5 if overlap else 0.0


def make_embedding_similarity(keyed_vectors):
    """
    Factory untuk similarity_fn berbasis embedding (mis. gensim KeyedVectors
    dari FastText Bahasa Indonesia). Pakai seperti:

        from gensim.models import KeyedVectors
        kv = KeyedVectors.load("path/ke/fasttext_id.kv")
        sim_fn = make_embedding_similarity(kv)
        disambiguate(..., similarity_fn=sim_fn)
    """
    def sim(cat_a: str, cat_b: str) -> float:
        try:
            return float(keyed_vectors.similarity(cat_a, cat_b))
        except KeyError:
            return _default_similarity(cat_a, cat_b)
    return sim
