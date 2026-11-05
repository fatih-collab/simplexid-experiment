"""
Evaluasi tertarget scope consistency (SC).

SC hanya mengubah label pada sebagian kecil pasangan R4, dan hampir tak satu pun
masuk sampel gold_polarity. Modul ini mengambil pasangan yang labelnya BERBEDA antara
sistem tanpa SC dan dengan SC, tanpa memakai gold, untuk dianotasi secara buta.

Hanya pasangan berbeda yang perlu dilabeli: pada uji McNemar, pasangan yang labelnya
sama tidak ikut menentukan hasil. Karena itu sampel ini mengukur SC vs pewarisan,
BUKAN akurasi polaritas sistem secara umum.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

from ..config import VariantConfig
from ..parser import ParsedSentence
from ..pipeline import run_pipeline
from .metrics import polarity_of
from .significance import mcnemar_exact

VALID_PAIR_DECISIONS = {"YES", "NO", "AMBIGUOUS"}
VALID_POLARITIES = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
SEED = 42

SHEET_COLUMNS = ["item_id", "sentence_id", "sentence", "aspect_text", "opinion_text",
                 "is_valid_pair", "gold_polarity", "annotation_status", "notes"]
SYSTEM_COLUMNS = ["item_id", "sentence_id", "aspect_idx", "opinion_idx", "rule",
                  "pred_without_sc", "pred_with_sc"]


def discordant_pairs(
    sentence_ids: Iterable[str],
    parses: Dict[str, ParsedSentence],
    sentences: Dict[str, str],
    without_sc: VariantConfig,
    with_sc: VariantConfig,
    lexicon: dict,
    vocabulary_for: Callable[[str], object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(lembar anotasi tanpa prediksi, prediksi sistem), urutan diacak deterministik."""
    sheet: List[dict] = []
    system: List[dict] = []
    for sid in sentence_ids:
        def predict(variant):
            results = run_pipeline(None, variant, "hotel", lexicon, parsed_sentence=parses[sid],
                                   aspect_vocabulary=vocabulary_for(sid))
            return {(r.pair.aspect_idx, r.pair.opinion_idx): r for r in results}

        base, full = predict(without_sc), predict(with_sc)
        for key, result in full.items():
            if key[0] < 0 or key not in base:
                continue
            before = polarity_of(base[key].factors.score_final)
            after = polarity_of(result.factors.score_final)
            if before == after:
                continue
            item_id = f"{sid}__a{key[0]}__o{key[1]}"
            sheet.append({"item_id": item_id, "sentence_id": sid, "sentence": sentences[sid],
                          "aspect_text": result.pair.aspect_text, "opinion_text": result.pair.opinion_text,
                          "is_valid_pair": "", "gold_polarity": "", "annotation_status": "", "notes": ""})
            system.append({"item_id": item_id, "sentence_id": sid, "aspect_idx": key[0], "opinion_idx": key[1],
                           "rule": result.pair.rule, "pred_without_sc": before, "pred_with_sc": after})

    sheet_df = pd.DataFrame(sheet, columns=SHEET_COLUMNS)
    if not sheet_df.empty:
        sheet_df = sheet_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sheet_df, pd.DataFrame(system, columns=SYSTEM_COLUMNS)


def read_sheet(path) -> pd.DataFrame:
    """Baca lembar; pemisah dideteksi otomatis karena Excel locale Indonesia menyimpan ';'."""
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def evaluate_annotations(sheet: pd.DataFrame, system: pd.DataFrame) -> dict:
    """Bandingkan pewarisan vs SC pada baris REVIEWED berpasangan valid."""
    if set(sheet["item_id"]) - set(system["item_id"]):
        raise ValueError("Ada item_id di lembar yang tidak ada di file prediksi")

    status = sheet["annotation_status"].str.strip().str.upper()
    reviewed = sheet[status == "REVIEWED"].copy()
    decision = reviewed["is_valid_pair"].str.strip().str.upper()
    polarity = reviewed["gold_polarity"].str.strip().str.upper()

    bad = set(decision) - VALID_PAIR_DECISIONS
    if bad:
        raise ValueError(f"is_valid_pair harus YES/NO/AMBIGUOUS, ditemukan {sorted(bad)}")
    valid = decision == "YES"
    bad = set(polarity[valid]) - VALID_POLARITIES
    if bad:
        raise ValueError(f"gold_polarity baris valid harus POSITIVE/NEGATIVE/NEUTRAL, ditemukan {sorted(bad)}")

    merged = reviewed[valid].assign(gold=polarity[valid]).merge(system, on="item_id")
    result = mcnemar_exact((merged["pred_without_sc"] == merged["gold"]).tolist(),
                           (merged["pred_with_sc"] == merged["gold"]).tolist())
    neither = int(((merged["pred_without_sc"] != merged["gold"]) & (merged["pred_with_sc"] != merged["gold"])).sum())
    return {
        "items_total": int(len(sheet)),
        "reviewed": int(len(reviewed)),
        "invalid_pair": int((decision == "NO").sum()),
        "ambiguous": int((decision == "AMBIGUOUS").sum()),
        "evaluated": int(len(merged)),
        "only_inheritance_correct": result["a_only_correct"],
        "only_sc_correct": result["b_only_correct"],
        "both_wrong": neither,
        "accuracy_inheritance": round(float((merged["pred_without_sc"] == merged["gold"]).mean()), 4) if len(merged) else None,
        "accuracy_sc": round(float((merged["pred_with_sc"] == merged["gold"]).mean()), 4) if len(merged) else None,
        "p_value": result["p_value"],
    }
