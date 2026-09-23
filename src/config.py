"""
Definisi varian ablation V0-V9 sebagai kombinasi flag.

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


VARIANTS = {
    "V0": VariantConfig(
        name="V0",
        description="Baseline murni: R0,R1,R2,R5,R6,R7 saja (setara SimplEx lama diadaptasi)",
    ),
    "V1": VariantConfig(
        name="V1", use_r3=True,
        description="V0 + R3 saja (ekspansi sisi aspek)",
    ),
    "V2": VariantConfig(
        name="V2", use_r4=True,
        description="V0 + R4 saja (ekspansi sisi opini)",
    ),
    "V3": VariantConfig(
        name="V3", use_r3=True, use_r4=True,
        description="V0 + R3 + R4 (dual conjunctive expansion)",
    ),
    "V4": VariantConfig(
        name="V4", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=False,
        description="V3 + R8 TANPA disambiguasi kontekstual (setara Liu 2005/Fei 2012)",
    ),
    "V5": VariantConfig(
        name="V5", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=True,
        description="V3 + R8 DENGAN disambiguasi kontekstual (revisi)",
    ),
    "V6": VariantConfig(
        name="V6", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=True, distance_weighted_formula=False,
        description="V5 + formula skor TANPA bobot jarak (setara SO-CAL/Polanyi-Zaenen)",
    ),
    "V7": VariantConfig(
        name="V7", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=True, distance_weighted_formula=True,
        description="V5 + formula skor DENGAN bobot jarak struktural (revisi)",
    ),
    "V8": VariantConfig(
        name="V8", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=True, distance_weighted_formula=True,
        scope_consistency=False,
        description="V7 TANPA scope consistency (pasangan R3/R4 mewarisi Neg/Int pasangan asal)",
    ),
    "V9": VariantConfig(
        name="V9", use_r3=True, use_r4=True, use_r8=True,
        r8_contextual_disambiguation=True, distance_weighted_formula=True,
        scope_consistency=True,
        description="SimplEx-ID LENGKAP (versi final yang diklaim di paten)",
    ),
}

# Perbandingan kunci yang wajib dilaporkan (lihat Bagian 6 rancangan eksperimen)
KEY_COMPARISONS = [
    ("V0", "V9", "Perbandingan utama (Eksperimen A)"),
    ("V1", "V3", "Kontribusi R4 di atas R3 (cek superadditive R3+R4)"),
    ("V2", "V3", "Kontribusi R3 di atas R4 (cek superadditive R3+R4)"),
    ("V4", "V5", "Kontribusi disambiguasi kontekstual R8 (uji khusus subset IMPLICIT_AMBIG)"),
    ("V6", "V7", "Kontribusi structure-weighted scoring"),
    ("V8", "V9", "Kontribusi scope consistency (uji khusus subset ASYM_MOD)"),
]

# Ambang batas jarak dependensi untuk faktor negasi (lihat scoring.py)
NEG_DISTANCE_THRESHOLD = 3   # edge; perlu di-tuning empiris (lihat catatan ablasi di draf paten)
INT_DECAY_LAMBDA = 0.4       # konstanta peluruhan intensifier; perlu di-tuning empiris

RANDOM_SEED = 42
K_FOLDS = 5
