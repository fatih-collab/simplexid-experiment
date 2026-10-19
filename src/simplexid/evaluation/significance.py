"""
Uji signifikansi berpasangan (H3.3).

- paired_bootstrap_f1 : resample KALIMAT (dengan pengembalian), hitung ulang F1 micro
  kedua sistem pada sampel yang sama -> distribusi selisih, CI 95%, p dua sisi.
- mcnemar_exact       : benar/salah per item (mis. polaritas per pasangan) dua sistem;
  uji binomial eksak pada pasangan diskordan.
- holm                : koreksi Holm-Bonferroni untuk banyak perbandingan.

Menggantikan K-fold + t-test pada rancangan lama (tidak tepat untuk sistem tanpa
training dan hanya 5 titik data).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import binomtest

from .metrics import Counts

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def paired_bootstrap_f1(
    counts_a: Dict[str, Counts],
    counts_b: Dict[str, Counts],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Selisih F1 micro (B - A) dengan bootstrap per kalimat. Kunci kedua dict harus sama."""
    if set(counts_a) != set(counts_b):
        raise ValueError("Sistem A dan B harus dievaluasi pada kalimat yang sama")
    ids = sorted(counts_a)
    a = np.array([[counts_a[s].tp, counts_a[s].fp, counts_a[s].fn] for s in ids], dtype=float)
    b = np.array([[counts_b[s].tp, counts_b[s].fp, counts_b[s].fn] for s in ids], dtype=float)

    observed = (_f1(*b.sum(0)[:, None]) - _f1(*a.sum(0)[:, None])).item()

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    batch = 1000
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        idx = rng.integers(0, len(ids), size=(size, len(ids)))
        sa = a[idx].sum(axis=1)   # (size, 3)
        sb = b[idx].sum(axis=1)
        deltas[start:start + size] = _f1(sb[:, 0], sb[:, 1], sb[:, 2]) - _f1(sa[:, 0], sa[:, 1], sa[:, 2])

    # p dua sisi: seberapa sering selisih bootstrap melewati nol ke arah berlawanan.
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "metric": "f1",
        "delta": round(observed, 4),
        "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
        "p_value": round(float(min(p_value, 1.0)), 4),
        "n_sentences": len(ids),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """b01 = A benar & B salah, b10 = A salah & B benar."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Panjang hasil A dan B harus sama")
    b01 = sum(bool(x) and not bool(y) for x, y in zip(correct_a, correct_b))
    b10 = sum(not bool(x) and bool(y) for x, y in zip(correct_a, correct_b))
    n = b01 + b10
    p = binomtest(b10, n, 0.5).pvalue if n else 1.0
    acc_a = sum(map(bool, correct_a)) / len(correct_a) if correct_a else 0.0
    acc_b = sum(map(bool, correct_b)) / len(correct_b) if correct_b else 0.0
    return {
        "metric": "accuracy",
        "delta": round(acc_b - acc_a, 4),
        "a_only_correct": b01,
        "b_only_correct": b10,
        "p_value": round(float(p), 4),
        "n_items": len(correct_a),
    }


def holm(p_values: Sequence[float]) -> List[float]:
    """p yang sudah dikoreksi Holm-Bonferroni (urutan sama dengan input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = round(running, 4)
    return adjusted
