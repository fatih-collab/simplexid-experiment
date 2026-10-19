"""
Metrik evaluasi SimplEx-ID (H3.3).

1. Ekstraksi pasangan eksplisit  -> P/R/F1 micro, per strata, per sumber, per rule.
2. Polaritas pada gold_polarity  -> tiga mode:
     end_to_end  : pasangan gold yang tidak terekstraksi dihitung SALAH,
     conditional : hanya pasangan gold yang berhasil diekstraksi,
     oracle_pair : pasangan gold diberikan langsung ke modul scoring
                   (memisahkan efek formula skor dari error ekstraksi).
3. Aspek implisit pada gold_implicit -> deteksi P/R/F1 + akurasi/macro-F1 kategori,
   dibandingkan majority baseline.

Semua fungsi mengembalikan juga data per-kalimat / per-item agar uji
signifikansi berpasangan (significance.py) bisa dihitung.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .gold_loader import EvalGold
from .matching import PredPair, match_pairs

POLARITY_EPS = 0.05
POLARITY_LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def prf(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "tp": self.tp, "fp": self.fp, "fn": self.fn}


def polarity_of(score: float, eps: float = POLARITY_EPS) -> str:
    if score > eps:
        return "POSITIVE"
    if score < -eps:
        return "NEGATIVE"
    return "NEUTRAL"


def _explicit(preds: Sequence[PredPair]) -> List[PredPair]:
    return [p for p in preds if not p.is_implicit]


# ------------------------------------------------------------------ extraction

def extraction_counts(gold: EvalGold, preds: Dict[str, List[PredPair]], level: str) -> Dict[str, Counts]:
    out = {}
    for sid in gold.sentence_ids:
        p = _explicit(preds.get(sid, []))
        g = gold.pairs[sid]
        tp = len(match_pairs(level, p, g))
        out[sid] = Counts(tp=tp, fp=len(p) - tp, fn=len(g) - tp)
    return out


def summarize_extraction(gold: EvalGold, per_sentence: Dict[str, Counts]) -> dict:
    def total(ids) -> dict:
        c = Counts()
        for sid in ids:
            c = c + per_sentence[sid]
        return {**c.prf(), "n_sentences": len(ids)}

    result = {"overall": total(gold.sentence_ids)}
    for attr in ("stratum", "source"):
        groups: Dict[str, list] = {}
        for sid in gold.sentence_ids:
            groups.setdefault(getattr(gold, attr)[sid], []).append(sid)
        result[f"by_{attr}"] = {k: total(v) for k, v in sorted(groups.items())}
    return result


def per_rule_precision(gold: EvalGold, preds: Dict[str, List[PredPair]], level: str) -> dict:
    hit, total = Counter(), Counter()
    for sid in gold.sentence_ids:
        p = _explicit(preds.get(sid, []))
        matched = match_pairs(level, p, gold.pairs[sid])
        for i, pair in enumerate(p):
            total[pair.rule] += 1
            hit[pair.rule] += i in matched
    return {rule: {"predicted": total[rule], "correct": hit[rule],
                   "precision": round(hit[rule] / total[rule], 4)} for rule in sorted(total)}


# ------------------------------------------------------------------ polarity

def polarity_items(
    gold: EvalGold,
    preds: Dict[str, List[PredPair]],
    polarity_df: pd.DataFrame,
    oracle_score_fn: Optional[Callable[[str, tuple, tuple], float]] = None,
) -> pd.DataFrame:
    """
    Satu baris per pasangan gold_polarity pada split ini.
    oracle_score_fn(sentence_id, aspect_span, opinion_span) -> skor (mode oracle_pair).
    """
    scope = "DEV_POLARITY" if gold.split == "dev" else "FINAL_POLARITY"
    rows_df = polarity_df[polarity_df["polarity_scope"] == scope]

    span_of = {p.pair_id: (sid, i, p) for sid in gold.sentence_ids for i, p in enumerate(gold.pairs[sid])}
    matched_cache: Dict[str, Dict[int, int]] = {}

    rows = []
    for row in rows_df.itertuples(index=False):
        if row.pair_id not in span_of:
            continue  # kalimat dikeluarkan dari universe (mis. AMBIGUOUS)
        sid, gold_idx, gp = span_of[row.pair_id]
        explicit = _explicit(preds.get(sid, []))
        if sid not in matched_cache:
            m = match_pairs("head", explicit, gold.pairs[sid])
            matched_cache[sid] = {g: p for p, g in m.items()}
        pred_idx = matched_cache[sid].get(gold_idx)
        e2e = polarity_of(explicit[pred_idx].score) if pred_idx is not None else None
        oracle = (polarity_of(oracle_score_fn(sid, gp.aspect_span, gp.opinion_span))
                  if oracle_score_fn else None)
        rows.append({
            "pair_id": row.pair_id,
            "sentence_id": sid,
            "polarity_stratum": row.polarity_stratum,
            "gold": str(row.gold_polarity).strip().upper(),
            "pred_end_to_end": e2e,
            "pred_oracle_pair": oracle,
        })
    return pd.DataFrame(rows, columns=["pair_id", "sentence_id", "polarity_stratum", "gold",
                                       "pred_end_to_end", "pred_oracle_pair"])


def classification_report(gold: Sequence[str], pred: Sequence[Optional[str]],
                          labels: Sequence[str] = POLARITY_LABELS) -> dict:
    """Akurasi + P/R/F1 per kelas. pred None = tidak ada prediksi (selalu salah)."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    per_class = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        per_class[label] = {**Counts(tp, fp, fn).prf(), "support": sum(g == label for g in gold)}
    binary = [per_class[l]["f1"] for l in ("POSITIVE", "NEGATIVE") if l in per_class]
    present = [per_class[l]["f1"] for l in labels if per_class[l]["support"] > 0]
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1_pos_neg": round(sum(binary) / len(binary), 4) if binary else 0.0,
        "macro_f1_present": round(sum(present) / len(present), 4) if present else 0.0,
        "per_class": per_class,
    }


def summarize_polarity(items: pd.DataFrame) -> dict:
    result = {}
    e2e = items["pred_end_to_end"].tolist()
    result["end_to_end"] = classification_report(items["gold"].tolist(), e2e)
    found = items[items["pred_end_to_end"].notna()]
    result["conditional"] = {
        **classification_report(found["gold"].tolist(), found["pred_end_to_end"].tolist()),
        "coverage": round(len(found) / len(items), 4) if len(items) else 0.0,
    }
    if items["pred_oracle_pair"].notna().any():
        result["oracle_pair"] = classification_report(items["gold"].tolist(), items["pred_oracle_pair"].tolist())
        result["oracle_pair_by_stratum"] = {
            s: classification_report(g["gold"].tolist(), g["pred_oracle_pair"].tolist())["accuracy"]
            for s, g in items.groupby("polarity_stratum")
        }
    return result


# ------------------------------------------------------------------ implicit

def implicit_items(gold: EvalGold, preds: Dict[str, List[PredPair]], implicit_df: pd.DataFrame) -> pd.DataFrame:
    scope = "DEV_IMPLICIT" if gold.split == "dev" else "FINAL_IMPLICIT"
    df = implicit_df[implicit_df["implicit_scope"] == scope]
    rows = []
    for row in df.itertuples(index=False):
        decision = str(row.decision).strip().upper()
        if decision == "AMBIGUOUS":
            continue
        span = (int(row.opinion_start), int(row.opinion_end))
        r8 = [p for p in preds.get(row.sentence_id, [])
              if p.is_implicit and span[0] <= p.opinion_head <= span[1]]
        rows.append({
            "implicit_id": row.implicit_id,
            "sentence_id": row.sentence_id,
            "decision": decision,
            "gold_category": str(row.gold_aspect_category).strip() if decision == "IMPLICIT" else None,
            "pred_category": r8[0].implicit_category if r8 else None,
        })
    return pd.DataFrame(rows, columns=["implicit_id", "sentence_id", "decision", "gold_category", "pred_category"])


def summarize_implicit(items: pd.DataFrame, baseline_category: str) -> dict:
    is_imp = items["decision"] == "IMPLICIT"
    has_pred = items["pred_category"].notna()
    detection = Counts(
        tp=int((is_imp & has_pred).sum()),
        fp=int((~is_imp & has_pred).sum()),
        fn=int((is_imp & ~has_pred).sum()),
    ).prf()
    imp = items[is_imp]
    categories = sorted(set(imp["gold_category"]))
    system = classification_report(imp["gold_category"].tolist(), imp["pred_category"].tolist(), categories)
    baseline = classification_report(imp["gold_category"].tolist(), [baseline_category] * len(imp), categories)
    return {
        "n_items": int(len(items)),
        "n_implicit": int(is_imp.sum()),
        "detection": detection,
        "category": {k: system[k] for k in ("accuracy", "macro_f1_present")},
        "majority_baseline": {"category": baseline_category,
                              **{k: baseline[k] for k in ("accuracy", "macro_f1_present")}},
        "per_category": system["per_class"],
    }
