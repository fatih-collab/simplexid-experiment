"""
Runner utama: jalankan varian V0-V9 pada TermA test set, k-fold cross
validation, dan paired t-test antar-varian kunci (Bagian 9 rancangan
eksperimen). Hasil disimpan ke results/metrics.csv dan results/significance.csv.
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from config import VARIANTS, KEY_COMPARISONS, K_FOLDS, RANDOM_SEED
from data_loader import load_terma
from pipeline import run_pipeline
from evaluate import evaluate_sentence, aggregate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DOMAIN = "hotel"


def load_swn_lexicon() -> dict:
    """
    TODO: ganti dengan leksikon sentimen Bahasa Indonesia riil Anda
    (mis. hasil terjemahan SentiWordNet atau leksikon InSet/SentiStrengthID).
    Placeholder di bawah HANYA untuk menjalankan kode tanpa error.
    """
    return {
        "baik": 0.6, "bagus": 0.7, "buruk": -0.6, "jelek": -0.7,
        "bersih": 0.5, "kotor": -0.5, "ramah": 0.6, "kasar": -0.5,
        "cepat": 0.4, "lambat": -0.4, "lemot": -0.5, "lemah": -0.4,
        "stabil": 0.4, "mahal": -0.3, "murah": 0.3, "enak": 0.6, "variatif": 0.3,
    }


def group_pairs_by_sentence(pairs):
    grouped = defaultdict(list)
    for p in pairs:
        grouped[p.sentence_id].append(p)
    return grouped


def build_gold_polarity_lookup(gold_pairs) -> dict:
    """
    TODO WAJIB DIISI: bangun dict (aspect_text_lower, opinion_text_lower)
    -> label polaritas ("positive"/"negative"/"neutral") dari ground truth.

    TermA murni (span aspek+opini) TIDAK menyertakan label polaritas
    langsung. Opsi untuk mengisi ini:
      1. Sambungkan dengan HoASA (ulasan sama dari AiryRooms) yang punya
         label polaritas per kategori aspek -- perlu pemetaan aspect_text
         eksplisit TermA ke salah satu dari 10 kategori HoASA.
      2. Hitung polaritas otomatis dari leksikon sentimen (SWN_score kata
         opini ground truth) sebagai proxy -- lebih cepat tapi kurang
         independen sebagai "ground truth" karena memakai leksikon yang
         sama dengan yang dievaluasi.
      3. Anotasi manual polaritas untuk subset yang dipakai (paling valid,
         paling mahal).

    Placeholder saat ini mengembalikan dict KOSONG secara sengaja, supaya
    Anda tidak lupa mengisi ini -- lihat peringatan runtime di main().
    """
    return {}


def run_variant_on_fold(sentence_groups, sentence_ids, variant, swn_lexicon,
                         gold_polarity_lookup=None):
    """Jalankan satu varian pada subset kalimat (satu fold), kembalikan aggregate counts."""
    per_sentence_counts = []
    for sid in sentence_ids:
        gold_pairs = sentence_groups[sid]
        tokens = gold_pairs[0].tokens
        predictions = run_pipeline(tokens, variant, DOMAIN, swn_lexicon)
        counts = evaluate_sentence(gold_pairs, predictions, tokens, gold_polarity_lookup)
        per_sentence_counts.append(counts)
    return per_sentence_counts


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    swn_lexicon = load_swn_lexicon()

    print("Memuat TermA...")
    terma = load_terma()
    test_pairs = terma["test"]
    sentence_groups = group_pairs_by_sentence(test_pairs)
    sentence_ids = list(sentence_groups.keys())

    # PENTING: accuracy (ketepatan polaritas) HANYA terhitung jika Anda
    # menyediakan gold_polarity_lookup: dict (aspect_text_lower, opinion_text_lower)
    # -> "positive"/"negative"/"neutral". TANPA ini, kolom accuracy akan
    # tetap NaN/0 untuk SEMUA varian -- dan kontribusi formula berbobot-jarak
    # (V6 vs V7) serta scope consistency (V8 vs V9) TIDAK AKAN TERLIHAT SAMA
    # SEKALI, karena kedua mekanisme itu hanya memengaruhi skor/polaritas,
    # bukan pasangan mana yang terekstraksi. Ini persis penyebab V6-V9
    # tampak identik pada eksperimen pertama Anda.
    gold_polarity_lookup = build_gold_polarity_lookup(test_pairs)
    if not gold_polarity_lookup:
        print("\n" + "="*70)
        print("PERINGATAN: gold_polarity_lookup KOSONG.")
        print("Accuracy TIDAK akan terukur untuk varian manapun, dan kontribusi")
        print("V6-vs-V7 (formula berbobot-jarak) serta V8-vs-V9 (scope consistency)")
        print("TIDAK AKAN TERDETEKSI. Isi build_gold_polarity_lookup() di bawah")
        print("sebelum hasil dipakai untuk mengisi tabel Efek Teknis draf paten.")
        print("="*70 + "\n")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sentence_ids_arr = np.array(sentence_ids)

    # fold_scores[variant_name][metric] = list nilai per fold (utk paired t-test)
    fold_scores = {v: defaultdict(list) for v in VARIANTS}

    for fold_i, (_, test_idx) in enumerate(kf.split(sentence_ids_arr)):
        fold_sentence_ids = sentence_ids_arr[test_idx].tolist()
        print(f"Fold {fold_i+1}/{K_FOLDS} ({len(fold_sentence_ids)} kalimat)...")

        for vname, variant in VARIANTS.items():
            per_sentence = run_variant_on_fold(sentence_groups, fold_sentence_ids,
                                                variant, swn_lexicon, gold_polarity_lookup)
            total = aggregate(per_sentence)
            fold_scores[vname]["precision"].append(total.precision)
            fold_scores[vname]["recall"].append(total.recall)
            fold_scores[vname]["f1"].append(total.f1)
            fold_scores[vname]["accuracy"].append(total.accuracy)

    # --- simpan metrics.csv (rata-rata +- std per varian) ---
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "description", "precision_mean", "precision_std",
                          "recall_mean", "recall_std", "f1_mean", "f1_std",
                          "accuracy_mean", "accuracy_std"])
        for vname, variant in VARIANTS.items():
            row = [vname, variant.description]
            for metric in ("precision", "recall", "f1", "accuracy"):
                vals = fold_scores[vname][metric]
                row += [np.mean(vals), np.std(vals)]
            writer.writerow(row)
    print(f"Tersimpan: {metrics_path}")

    # --- paired t-test untuk perbandingan kunci ---
    sig_path = os.path.join(RESULTS_DIR, "significance.csv")
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "description", "metric", "mean_diff",
                          "ci_lower_95", "ci_upper_95", "p_value"])
        for v_a, v_b, desc in KEY_COMPARISONS:
            for metric in ("precision", "recall", "f1", "accuracy"):
                a = np.array(fold_scores[v_a][metric])
                b = np.array(fold_scores[v_b][metric])
                diff = b - a
                mean_diff = diff.mean()
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                ci_lower = mean_diff - 1.96 * se
                ci_upper = mean_diff + 1.96 * se
                t_stat, p_val = stats.ttest_rel(b, a)
                writer.writerow([f"{v_a}_vs_{v_b}", desc, metric,
                                  mean_diff, ci_lower, ci_upper, p_val])
    print(f"Tersimpan: {sig_path}")
    print("\nSelesai. Pindahkan angka dari metrics.csv/significance.csv ke tabel")
    print("Efek Teknis pada Draft_Paten_SimplEx-ID.docx (Bagian 11 rancangan eksperimen).")


if __name__ == "__main__":
    main()
