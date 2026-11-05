"""
Lembar anotasi tertarget disambiguasi kontekstual R8: build | evaluate (development saja).

gold_implicit tidak bisa menguji konteks karena kalimatnya tanpa aspek eksplisit. Di sini
diambil sampel acak kasus yang kategorinya BERBEDA antara R8 statis (A3) dan kontekstual
(A4), hanya pada token yang berada di span opini TermA, untuk dianotasi secara buta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplexid.aspect_vocab import crossfit_vocabularies, fold_of  # noqa: E402
from simplexid.config import VARIANTS  # noqa: E402
from simplexid.dam_builder import HOASA_CATEGORIES, load_dam  # noqa: E402
from simplexid.evaluation.sc_targeted import read_sheet  # noqa: E402
from simplexid.evaluation.significance import mcnemar_exact  # noqa: E402
from simplexid.lexicon import load_lexicon  # noqa: E402
from simplexid.parse_cache import load_parse_cache  # noqa: E402
from simplexid.pipeline import run_pipeline  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "annotation" / "ctx_targeted"
SHEET_PATH = OUT_DIR / "ctx_targeted_dev.csv"
SYSTEM_PATH = OUT_DIR / "ctx_targeted_dev_system.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "h4" / "ctx_targeted_dev.json"
SEED = 42


def cmd_build(args) -> None:
    if SHEET_PATH.exists() and not args.overwrite:
        print(f"Skip existing (berisi anotasi?): {SHEET_PATH}  -- pakai --overwrite untuk membuat ulang")
        return
    terma = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "terma_processed.csv")
    parses = load_parse_cache(PROJECT_ROOT / "data" / "cache" / "parses", terma_df=terma)
    folds, lexicon, dam = crossfit_vocabularies(terma, parses), load_lexicon(), load_dam()

    rows = []
    for row in terma[terma["source_split"] == "train"].itertuples(index=False):
        sid = row.sentence_id
        spans = json.loads(row.opinions)

        def r8(variant):
            results = run_pipeline(None, VARIANTS[variant], "hotel", lexicon, parsed_sentence=parses[sid],
                                   aspect_vocabulary=folds[fold_of(sid)], dam=dam)
            return {r.pair.opinion_idx: r.pair for r in results if r.pair.aspect_idx < 0}

        static, contextual = r8("A3"), r8("A4")
        for idx, pair in contextual.items():
            if idx in static and static[idx].aspect_text != pair.aspect_text \
                    and any(s["start"] <= idx <= s["end"] for s in spans):
                rows.append({"item_id": f"{sid}__o{idx}", "sentence_id": sid, "sentence": row.sentence,
                             "opinion_text": pair.opinion_text, "pred_static": static[idx].aspect_text,
                             "pred_contextual": pair.aspect_text})
    pool = pd.DataFrame(rows)
    sample = pool.sample(n=min(args.n, len(pool)), random_state=SEED).reset_index(drop=True)
    sheet = sample[["item_id", "sentence_id", "sentence", "opinion_text"]].assign(
        decision="", gold_aspect_category="", annotation_status="", notes="")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    sample[["item_id", "pred_static", "pred_contextual"]].to_csv(SYSTEM_PATH, index=False, encoding="utf-8-sig")
    print(f"Kasus berbeda di span opini: {len(pool)} | sampel: {len(sheet)} -> {SHEET_PATH}")


def evaluate(sheet: pd.DataFrame, system: pd.DataFrame) -> dict:
    reviewed = sheet[sheet["annotation_status"].str.strip().str.upper() == "REVIEWED"].copy()
    decision = reviewed["decision"].str.strip().str.upper()
    bad = set(decision) - {"IMPLICIT", "NOT_IMPLICIT", "AMBIGUOUS"}
    if bad:
        raise ValueError(f"decision harus IMPLICIT/NOT_IMPLICIT/AMBIGUOUS, ditemukan {sorted(bad)}")
    implicit = reviewed[decision == "IMPLICIT"].copy()
    implicit["gold"] = implicit["gold_aspect_category"].str.strip()
    bad = set(implicit["gold"]) - set(HOASA_CATEGORIES)
    if bad:
        raise ValueError(f"gold_aspect_category di luar ontologi: {sorted(bad)}")
    merged = implicit.merge(system, on="item_id")
    test = mcnemar_exact((merged["pred_static"] == merged["gold"]).tolist(),
                         (merged["pred_contextual"] == merged["gold"]).tolist())
    return {"items_total": int(len(sheet)), "reviewed": int(len(reviewed)), "evaluated": int(len(merged)),
            "only_static_correct": test["a_only_correct"], "only_contextual_correct": test["b_only_correct"],
            "both_wrong": int(((merged["pred_static"] != merged["gold"]) & (merged["pred_contextual"] != merged["gold"])).sum()),
            "p_value": test["p_value"]}


def cmd_evaluate(_) -> None:
    result = evaluate(read_sheet(SHEET_PATH), read_sheet(SYSTEM_PATH))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--overwrite", action="store_true", help="Membuat ulang lembar (MENGHAPUS anotasi)")
    sub.add_parser("evaluate")
    args = parser.parse_args()
    try:
        {"build": cmd_build, "evaluate": cmd_evaluate}[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
