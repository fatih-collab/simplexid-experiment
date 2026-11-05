"""
Definisi varian ablation (kumulatif A0-A6 dan leave-one-out L_*) sebagai kombinasi flag.

Setiap varian adalah dict flag yang dikonsumsi oleh pipeline.py.
Menambah/mengubah varian cukup dengan mengedit dict VARIANTS di bawah --
tidak perlu duplikasi logika di modul lain.
"""

from dataclasses import dataclass, field


@dataclass
class VariantConfig:
    name: str
    use_r3: bool = False              # ekspansi konjungtif sisi ASPEK
    use_r4: bool = False              # ekspansi konjungtif sisi OPINI
    use_r8: bool = False              # resolusi aspek implisit (DAM)
    r8_contextual_disambiguation: bool = False   # disambiguasi kontekstual pada R8
    distance_weighted_formula: bool = False       # Neg/Int sebagai fungsi jarak dependensi
    scope_consistency: bool = False               # hitung ulang Neg/Int utk pasangan R3/R4
    description: str = ""


def _variant(name, description, **flags) -> VariantConfig:
    return VariantConfig(name=name, description=description, **flags)


_R34 = dict(use_r3=True, use_r4=True)
_R8S = dict(use_r8=True)
_R8C = dict(use_r8=True, r8_contextual_disambiguation=True)
_DW = dict(distance_weighted_formula=True)
_SC = dict(scope_consistency=True)

# Kumulatif A0..A6 (A6 = sistem lengkap) + leave-one-out L_* dari A6 + S_R4 untuk cek interaksi R3/R4.
# Leave-one-out SC sama dengan A5 (A5 vs A6), jadi tidak punya varian tersendiri.
# Setiap varian wajib punya kombinasi flag yang unik (diuji di tests/test_h4_ablation.py).
VARIANTS = {
    "A0": _variant("A0", "Base: R0,R1,R2,R5,R6,R7"),
    "A1": _variant("A1", "A0 + R3", use_r3=True),
    "A2": _variant("A2", "A1 + R4", **_R34),
    "A3": _variant("A3", "A2 + R8 statis", **_R34, **_R8S),
    "A4": _variant("A4", "A2 + R8 kontekstual", **_R34, **_R8C),
    "A5": _variant("A5", "A4 + DW scoring", **_R34, **_R8C, **_DW),
    "A6": _variant("A6", "A5 + scope consistency (LENGKAP)", **_R34, **_R8C, **_DW, **_SC),
    "S_R4": _variant("S_R4", "A0 + R4 saja", use_r4=True),
    "L_R3": _variant("L_R3", "A6 tanpa R3", use_r4=True, **_R8C, **_DW, **_SC),
    "L_R4": _variant("L_R4", "A6 tanpa R4", use_r3=True, **_R8C, **_DW, **_SC),
    "L_R8": _variant("L_R8", "A6 tanpa R8", **_R34, **_DW, **_SC),
    "L_CTX": _variant("L_CTX", "A6 dengan R8 statis", **_R34, **_R8S, **_DW, **_SC),
    "L_DW": _variant("L_DW", "A6 dengan skor statis", **_R34, **_R8C, **_SC),
}

# (A, B, deskripsi, target). Tiap perbandingan hanya diuji pada target yang memang
# bisa dipengaruhi mekanismenya, agar koreksi Holm tidak terbebani uji yang pasti nol.
KEY_COMPARISONS = [
    ("A0", "A6", "Perbandingan utama", ("pair_f1_head", "polarity_end_to_end", "polarity_oracle_pair")),
    ("A0", "A1", "Kontribusi R3", ("pair_f1_head", "pair_f1_head_coord_aspect")),
    ("A1", "A2", "Kontribusi R4 di atas R3", ("pair_f1_head", "pair_f1_head_coord_opinion")),
    ("S_R4", "A2", "Kontribusi R3 di atas R4", ("pair_f1_head",)),
    ("A2", "A3", "Kontribusi R8 statis", ("implicit_category",)),
    ("A3", "A4", "Kontribusi disambiguasi kontekstual R8", ("implicit_category",)),
    ("A4", "A5", "Kontribusi DW scoring", ("polarity_oracle_pair", "polarity_end_to_end")),
    ("A5", "A6", "Kontribusi scope consistency",
     ("polarity_oracle_pair", "polarity_oracle_pair_asym_mod", "polarity_end_to_end")),
    ("L_R3", "A6", "Leave-one-out R3", ("pair_f1_head",)),
    ("L_R4", "A6", "Leave-one-out R4", ("pair_f1_head",)),
    ("L_R8", "A6", "Leave-one-out R8", ("implicit_category",)),
    ("L_CTX", "A6", "Leave-one-out disambiguasi", ("implicit_category",)),
    ("L_DW", "A6", "Leave-one-out DW", ("polarity_oracle_pair", "polarity_end_to_end")),
]

# Ambang batas jarak dependensi untuk faktor negasi (lihat scoring.py)
NEG_DISTANCE_THRESHOLD = 3   # edge; perlu di-tuning empiris (lihat catatan ablasi di draf paten)
INT_DECAY_LAMBDA = 0.4       # konstanta peluruhan intensifier; perlu di-tuning empiris

# Skor |predikat dinegasi yang tidak ada di leksikon| (dikalikan Neg menjadi negatif lemah).
NEGATED_UNKNOWN_SCORE = 0.3

RANDOM_SEED = 42
K_FOLDS = 5
