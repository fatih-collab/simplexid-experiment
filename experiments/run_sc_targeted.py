"""Lembar anotasi tertarget scope consistency: build | evaluate (development saja)."""

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
from simplexid.evaluation.sc_targeted import discordant_pairs, evaluate_annotations, read_sheet  # noqa: E402
from simplexid.lexicon import load_lexicon  # noqa: E402
from simplexid.parse_cache import load_parse_cache  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "annotation" / "sc_targeted"
SHEET_PATH = OUT_DIR / "sc_targeted_dev.csv"
SYSTEM_PATH = OUT_DIR / "sc_targeted_dev_system.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "h4" / "sc_targeted_dev.json"


def cmd_build(args) -> None:
    if SHEET_PATH.exists() and not args.overwrite:
        print(f"Skip existing (berisi anotasi?): {SHEET_PATH}  -- pakai --overwrite untuk membuat ulang")
        return
    terma = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "terma_processed.csv")
    parses = load_parse_cache(PROJECT_ROOT / "data" / "cache" / "parses", terma_df=terma)
    folds = crossfit_vocabularies(terma, parses)
    train = terma[terma["source_split"] == "train"]
    sheet, system = discordant_pairs(
        train["sentence_id"], parses, dict(zip(train["sentence_id"], train["sentence"])),
        VARIANTS["A5"], VARIANTS["A6"], load_lexicon(), lambda sid: folds[fold_of(sid)],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    system.to_csv(SYSTEM_PATH, index=False, encoding="utf-8-sig")
    print(f"Created: {SHEET_PATH} ({len(sheet)} baris)")


def cmd_evaluate(_) -> None:
    result = evaluate_annotations(read_sheet(SHEET_PATH), read_sheet(SYSTEM_PATH))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
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
