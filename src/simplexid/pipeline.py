"""
Orkestrasi pipeline SimplEx-ID lengkap, dikontrol oleh VariantConfig (config.py).

Ini adalah satu-satunya tempat yang perlu diubah untuk menjalankan varian
V0-V9 -- logika tiap mekanisme sendiri (rules.py, dam.py, scoring.py)
TIDAK perlu diduplikasi/diedit per varian.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

from .parser import ParsedSentence, parse_tokens
from .rules import extract_explicit_pairs, expand_r3, expand_r4, find_orphan_opinions, Pair
from .dam import disambiguate, lookup
from .scoring import compute_factors, apply_scope_consistency, SentimentFactors
from .config import VariantConfig


@dataclass
class ExtractionResult:
    pair: Pair
    factors: SentimentFactors
    domain_aspect: bool = False   # True jika aspect berasal dari R8 (implisit)


def run_pipeline(tokens: List[str], variant: VariantConfig, domain: str,
                  swn_lexicon: dict,
                  document_context_aspects: List[str] = None,
                  parsed_sentence: ParsedSentence = None) -> List[ExtractionResult]:
    """
    Jalankan pipeline lengkap pada satu kalimat (sudah ditokenisasi, selaras
    dengan tokenisasi ground truth TermA) sesuai flag pada `variant`.

    parsed_sentence: jika diberikan, lewati parse_tokens(tokens) dan pakai
        ParsedSentence ini langsung -- dipakai untuk pengujian offline
        (lihat test_offline.py::test_pipeline_no_dict_collision) tanpa
        memerlukan UDPipe/koneksi internet.

    document_context_aspects: daftar teks aspek eksplisit yang SUDAH
        diekstraksi dari kalimat-kalimat LAIN dalam dokumen ulasan yang sama
        (lihat contoh "sarapan" di Gambar 3 draf paten -- kasus tsb
        melibatkan dua kalimat berbeda dalam satu ulasan). Kosongkan/None
        jika memproses kalimat secara berdiri sendiri tanpa konteks dokumen
        (disambiguasi R8 hanya akan memakai aspek dari kalimat itu sendiri).
        Lihat run_experiment.py::process_document() untuk contoh agregasi
        lintas-kalimat dalam satu dokumen ulasan.
    """
    sent = parsed_sentence if parsed_sentence is not None else parse_tokens(tokens)

    # --- Kelompok Aturan Pertama (selalu aktif di semua varian) ---
    pairs = extract_explicit_pairs(sent)

    # --- R3/R4: ekspansi konjungtif ---
    if variant.use_r3:
        pairs += expand_r3(sent, pairs)
    if variant.use_r4:
        pairs += expand_r4(sent, pairs)

    results: Dict[Tuple[int, int], ExtractionResult] = {}
    for p in pairs:
        factors = compute_factors(sent, p.opinion_idx, swn_lexicon,
                                   distance_weighted=variant.distance_weighted_formula)
        results[(p.aspect_idx, p.opinion_idx)] = ExtractionResult(pair=p, factors=factors)

    # --- R8: resolusi aspek implisit (dengan/tanpa disambiguasi kontekstual) ---
    if variant.use_r8:
        local_aspects = _current_documented_aspects(sent, results, domain)
        documented_aspects = local_aspects + (document_context_aspects or [])
        orphans = find_orphan_opinions(sent, [r.pair for r in results.values()])
        for o in orphans:
            candidates = lookup(o.lemma, domain)
            if not candidates:
                continue
            if variant.r8_contextual_disambiguation:
                dam_result = disambiguate(o.lemma, domain, documented_aspects)
                chosen = dam_result.chosen
            else:
                chosen = candidates[0]   # pemetaan statis: selalu kandidat pertama
            if chosen is None:
                continue
            factors = compute_factors(sent, o.idx, swn_lexicon,
                                       distance_weighted=variant.distance_weighted_formula)
            pseudo_pair = Pair(aspect_idx=-1, opinion_idx=o.idx, rule="R8",
                                aspect_text=chosen, opinion_text=o.text)
            results[(-1, o.idx)] = ExtractionResult(pair=pseudo_pair, factors=factors,
                                                     domain_aspect=True)

    # --- Scope consistency untuk pasangan hasil R3/R4 ---
    entries = [{"pair": r.pair, "factors": r.factors} for r in results.values()]
    entries = apply_scope_consistency(
        sent, entries, swn_lexicon,
        distance_weighted=variant.distance_weighted_formula,
        enabled=variant.scope_consistency,
    )
    for e in entries:
        key = (e["pair"].aspect_idx, e["pair"].opinion_idx)
        results[key].factors = e["factors"]

    return list(results.values())


def _current_documented_aspects(sent: ParsedSentence, results: Dict[Tuple[int, int], ExtractionResult],
                                 domain: str) -> List[str]:
    """Daftar teks aspek eksplisit yang sudah diekstraksi -- konteks untuk disambiguasi R8."""
    return [r.pair.aspect_text for r in results.values() if not r.domain_aspect]
