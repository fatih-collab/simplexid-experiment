"""
Uji logika inti (rules.py, dam.py, scoring.py) memakai ParsedSentence buatan
tangan (mock), TANPA memerlukan UDPipe/koneksi internet -- untuk memverifikasi
kebenaran logika sebelum dijalankan dengan parser & data sungguhan.

Jalankan: python src/test_offline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from parser import ParsedSentence, Token, dependency_distance
from rules import extract_explicit_pairs, expand_r3, expand_r4, find_orphan_opinions
from dam import disambiguate, lookup
from scoring import compute_factors


def mock_sentence(spec):
    """
    spec: list of (text, lemma, pos, dep, head_idx)
    index token = urutan dalam list (0-based)
    """
    tokens = [Token(idx=i, text=t, lemma=l, pos=p, dep=d, head_idx=h)
              for i, (t, l, p, d, h) in enumerate(spec)]
    return ParsedSentence(tokens=tokens)


def test_r0_amod():
    # "kecepatan internetnya sangat cepat" -> (kecepatan, cepat) via amod
    # idx: 0=kecepatan(NOUN,root) 1=internetnya(NOUN,compound->0) 2=sangat(ADV,advmod->3) 3=cepat(ADJ,amod->0)
    sent = mock_sentence([
        ("Kecepatan", "kecepatan", "NOUN", "root", 0),
        ("internetnya", "internet", "NOUN", "compound", 0),
        ("sangat", "sangat", "ADV", "advmod", 3),
        ("cepat", "cepat", "ADJ", "amod", 0),
    ])
    pairs = extract_explicit_pairs(sent)
    assert any(p.rule == "R0" and p.aspect_idx == 0 and p.opinion_idx == 3 for p in pairs), pairs
    print("test_r0_amod: OK ->", [(p.rule, p.aspect_text, p.opinion_text) for p in pairs])


def test_r3_conj_expansion():
    # "Kecepatan dan stabilitas koneksi sangat baik"
    # 0=Kecepatan(NOUN,root) 1=dan(CCONJ,cc->2) 2=stabilitas(NOUN,conj->0) 3=koneksi(NOUN,compound->2)
    # 4=sangat(ADV,advmod->5) 5=baik(ADJ,amod->0)
    sent = mock_sentence([
        ("Kecepatan", "kecepatan", "NOUN", "root", 0),
        ("dan", "dan", "CCONJ", "cc", 2),
        ("stabilitas", "stabilitas", "NOUN", "conj", 0),
        ("koneksi", "koneksi", "NOUN", "compound", 2),
        ("sangat", "sangat", "ADV", "advmod", 5),
        ("baik", "baik", "ADJ", "amod", 0),
    ])
    pairs = extract_explicit_pairs(sent)
    assert any(p.rule == "R0" and p.aspect_idx == 0 and p.opinion_idx == 5 for p in pairs), pairs
    expanded = expand_r3(sent, pairs)
    assert any(p.aspect_idx == 2 and p.opinion_idx == 5 for p in expanded), expanded
    print("test_r3_conj_expansion: OK -> R3 menghasilkan", [(p.aspect_text, p.opinion_text) for p in expanded])


def test_r4_conj_expansion_and_asym_scope():
    # "Sinyalnya sangat lemah dan tidak stabil"
    # 0=Sinyal(NOUN,root) 1=sangat(ADV,advmod->2) 2=lemah(ADJ,amod->0) 3=dan(CCONJ,cc->5)
    # 4=tidak(ADV,advmod->5) 5=stabil(ADJ,conj->2)
    sent = mock_sentence([
        ("Sinyal", "sinyal", "NOUN", "root", 0),
        ("sangat", "sangat", "ADV", "advmod", 2),
        ("lemah", "lemah", "ADJ", "amod", 0),
        ("dan", "dan", "CCONJ", "cc", 5),
        ("tidak", "tidak", "ADV", "advmod", 5),
        ("stabil", "stabil", "ADJ", "conj", 2),
    ])
    pairs = extract_explicit_pairs(sent)
    assert any(p.rule == "R0" and p.aspect_idx == 0 and p.opinion_idx == 2 for p in pairs), pairs
    expanded = expand_r4(sent, pairs)
    assert any(p.aspect_idx == 0 and p.opinion_idx == 5 for p in expanded), expanded
    print("test_r4_conj_expansion_and_asym_scope: OK -> R4 menghasilkan",
          [(p.aspect_text, p.opinion_text) for p in expanded])

    swn = {"lemah": -0.4, "stabil": 0.4}
    f_lemah = compute_factors(sent, 2, swn, distance_weighted=True)
    f_stabil = compute_factors(sent, 5, swn, distance_weighted=True)
    assert f_lemah.neg == 1.0 and f_lemah.intensifier > 1.0, f_lemah
    assert f_stabil.neg == -1.0 and f_stabil.intensifier == 1.0, f_stabil
    print(f"  Neg/Int (lemah) = {f_lemah.neg}, {f_lemah.intensifier:.2f} (benar: tidak ada negasi, ada intensifier)")
    print(f"  Neg/Int (stabil) = {f_stabil.neg}, {f_stabil.intensifier:.2f} (benar: ada negasi, TIDAK ada intensifier -- scope consistency bekerja)")


def test_dam_disambiguation():
    # kasus "mahal" dgn konteks aspek "sarapan" -> harus pilih sunrise_meal-terkait
    result = disambiguate("mahal", "hotel", documented_aspects=["sarapan"])
    print("test_dam_disambiguation: kandidat =", result.candidates, "-> dipilih =", result.chosen,
          "(disambiguated=%s)" % result.disambiguated)
    assert result.disambiguated is True
    assert len(result.candidates) > 1


def test_orphan_detection():
    sent = mock_sentence([
        ("mahal", "mahal", "ADJ", "root", 0),
        ("sekali", "sekali", "ADV", "advmod", 0),
    ])
    pairs = extract_explicit_pairs(sent)
    orphans = find_orphan_opinions(sent, pairs)
    assert len(orphans) == 1 and orphans[0].lemma == "mahal"
    print("test_orphan_detection: OK -> orphan =", orphans[0].text)


def test_pipeline_no_dict_collision():
    """
    Regresi untuk bug yang ditemukan di hasil eksperimen pertama pengguna:
    pipeline.run_pipeline() sebelumnya mengunci dict `results` hanya dengan
    opinion_idx, sehingga pasangan hasil R3 (yang berbagi opinion_idx yang
    SAMA dengan pasangan asal, mis. (kecepatan,baik) dan (stabilitas,baik)
    sama-sama beropini "baik") saling menimpa -- hanya pasangan TERAKHIR
    yang tersisa. Ini menjelaskan mengapa mengaktifkan R3 pada eksperimen
    pertama justru MENURUNKAN recall, padahal R3 seharusnya murni menambah.
    """
    import config
    from pipeline import run_pipeline

    sent = mock_sentence([
        ("Kecepatan", "kecepatan", "NOUN", "root", 0),
        ("dan", "dan", "CCONJ", "cc", 2),
        ("stabilitas", "stabilitas", "NOUN", "conj", 0),
        ("koneksi", "koneksi", "NOUN", "compound", 2),
        ("sangat", "sangat", "ADV", "advmod", 5),
        ("baik", "baik", "ADJ", "amod", 0),
    ])
    swn = {"baik": 0.6}
    variant = config.VARIANTS["V3"]  # R3+R4 aktif
    results = run_pipeline(tokens=None, variant=variant, domain="hotel",
                            swn_lexicon=swn, parsed_sentence=sent)
    pairs_found = {(r.pair.aspect_text, r.pair.opinion_text) for r in results}
    assert ("Kecepatan", "baik") in pairs_found, \
        f"BUG: pasangan asal (Kecepatan,baik) hilang/tertimpa! Hasil: {pairs_found}"
    assert ("stabilitas", "baik") in pairs_found, \
        f"BUG: pasangan hasil R3 (stabilitas,baik) tidak ditemukan! Hasil: {pairs_found}"
    print("test_pipeline_no_dict_collision: OK -> kedua pasangan tetap ada:", pairs_found)


if __name__ == "__main__":
    test_r0_amod()
    test_r3_conj_expansion()
    test_r4_conj_expansion_and_asym_scope()
    test_dam_disambiguation()
    test_orphan_detection()
    test_pipeline_no_dict_collision()
    print("\nSEMUA TES OFFLINE LOLOS.")
