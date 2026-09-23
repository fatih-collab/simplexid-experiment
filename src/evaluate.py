"""
Evaluasi hasil ekstraksi terhadap ground truth TermA: Precision, Recall,
F1, dan Accuracy (polaritas), sesuai definisi operasional di Bagian 8
rancangan eksperimen.
"""

from dataclasses import dataclass
from typing import List
from data_loader import GoldPair
from pipeline import ExtractionResult


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    polarity_correct: int = 0
    polarity_total: int = 0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self):
        return self.polarity_correct / self.polarity_total if self.polarity_total else 0.0


def polarity_of(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def evaluate_sentence(gold_pairs: List[GoldPair], predictions: List[ExtractionResult],
                       tokens: List[str], gold_polarity_lookup=None) -> ConfusionCounts:
    """
    Bandingkan prediksi (ExtractionResult) dengan ground truth (GoldPair) untuk
    SATU kalimat. Pencocokan aspek dilakukan berbasis teks (bukan index span
    murni) karena R8 menghasilkan aspek implisit tanpa span asli di kalimat --
    lihat catatan di bawah untuk penanganan kasus tsb.

    gold_polarity_lookup: optional dict (aspect_text, opinion_text) -> label
        polaritas ground truth, untuk menghitung accuracy. Jika None, accuracy
        tidak dihitung (polarity_total tetap 0).
    """
    counts = ConfusionCounts()

    gold_set = {(g.aspect_text.lower(), g.opinion_text.lower()) for g in gold_pairs}
    pred_set = set()
    for r in predictions:
        a_text = r.pair.aspect_text.lower()
        o_text = r.pair.opinion_text.lower()
        pred_set.add((a_text, o_text))

        if gold_polarity_lookup is not None:
            key = (a_text, o_text)
            if key in gold_polarity_lookup:
                counts.polarity_total += 1
                pred_pol = polarity_of(r.factors.score_final)
                if pred_pol == gold_polarity_lookup[key]:
                    counts.polarity_correct += 1

    counts.tp = len(gold_set & pred_set)
    counts.fp = len(pred_set - gold_set)
    counts.fn = len(gold_set - pred_set)
    return counts


def aggregate(counts_list: List[ConfusionCounts]) -> ConfusionCounts:
    """Jumlahkan TP/FP/FN/polarity di seluruh kalimat -> hitung P/R/F1/Acc micro-average."""
    total = ConfusionCounts()
    for c in counts_list:
        total.tp += c.tp
        total.fp += c.fp
        total.fn += c.fn
        total.polarity_correct += c.polarity_correct
        total.polarity_total += c.polarity_total
    return total


# --- catatan penanganan R8 (aspek implisit) ---
#
# Ground truth TermA HANYA memuat aspek EKSPLISIT (yang muncul sebagai span
# di kalimat). Pasangan hasil R8 (aspek implisit, mis. "HARGA_LAYANAN_TAMBAHAN")
# TIDAK akan pernah cocok dengan span TermA apa pun secara definisi, sehingga:
#   - Pada Eksperimen A/B (perbandingan P/R/F1 terhadap TermA apa adanya),
#     kontribusi R8 TIDAK akan tampak / bahkan bisa terlihat sebagai FP palsu.
#   - Untuk mengevaluasi R8 secara adil, WAJIB dibuat ground truth TAMBAHAN
#     khusus untuk kasus aspek implisit (subset IMPLICIT_AMBIG dari anotasi
#     sekunder, lihat secondary_tagging.py) -- annotator memberi label
#     kategori aspek implisit yang benar secara manual, lalu bandingkan
#     `chosen` dari dam.disambiguate() terhadap label tsb secara terpisah
#     dari evaluasi P/R/F1 utama. Ini PENTING dan mudah terlewat.
