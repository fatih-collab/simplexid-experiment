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

from simplexid.gold import (  # noqa: E402
    GoldValidationError,
    build_auto_1a1o_sheet,
    build_gold_pairs,
    build_implicit_template,
    build_polarity_template,
    h2_status,
    save_split_files,
    validate_gold_pairs,
    validate_implicit_gold,
    validate_polarity_gold,
    validate_auto_1a1o,
    validate_review_log,
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_GOLD = PROJECT_ROOT / "data" / "gold"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"

TERMA_PATH = DATA_PROCESSED / "terma_processed.csv"
HOASA_PATH = DATA_PROCESSED / "hoasa_processed.csv"
REVIEW_PATH = DATA_GOLD / "gold_review_log.csv"
GOLD_PAIRS_PATH = DATA_GOLD / "gold_pairs.csv"
GOLD_POLARITY_PATH = DATA_GOLD / "gold_polarity.csv"
GOLD_IMPLICIT_PATH = DATA_GOLD / "gold_implicit.csv"
AUTO_1A1O_PATH = DATA_GOLD / "gold_dev_auto_1a1o.csv"


def _need(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File wajib tidak ditemukan: {path}\n"
            "Periksa struktur project sebelum menjalankan H2."
        )


def load_core() -> tuple[pd.DataFrame, pd.DataFrame]:
    _need(TERMA_PATH)
    _need(REVIEW_PATH)
    terma = pd.read_csv(TERMA_PATH)
    review = pd.read_csv(REVIEW_PATH)
    return terma, review


def cmd_pairs(_: argparse.Namespace) -> None:
    terma, review = load_core()
    print("[1/3] Validate gold_review_log.csv")
    summary_review = validate_review_log(review, terma, require_reviewed=True)
    print(json.dumps(summary_review, indent=2, ensure_ascii=False))

    print("\n[2/3] Build gold_pairs.csv")
    gold_pairs = build_gold_pairs(review, terma)

    print("\n[3/3] Validate gold_pairs.csv")
    summary_pairs = validate_gold_pairs(gold_pairs, terma, review)
    print(json.dumps(summary_pairs, indent=2, ensure_ascii=False))

    DATA_GOLD.mkdir(parents=True, exist_ok=True)
    gold_pairs.to_csv(GOLD_PAIRS_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {GOLD_PAIRS_PATH}")


def cmd_splits(_: argparse.Namespace) -> None:
    _need(TERMA_PATH)
    terma = pd.read_csv(TERMA_PATH)
    metadata = save_split_files(terma, DATA_SPLITS)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nSaved split files in: {DATA_SPLITS}")


def cmd_templates(args: argparse.Namespace) -> None:
    _need(TERMA_PATH)
    _need(GOLD_PAIRS_PATH)
    terma = pd.read_csv(TERMA_PATH)
    gold_pairs = pd.read_csv(GOLD_PAIRS_PATH)
    DATA_GOLD.mkdir(parents=True, exist_ok=True)

    if GOLD_POLARITY_PATH.exists() and not args.overwrite:
        print(f"Skip existing: {GOLD_POLARITY_PATH}")
    else:
        polarity = build_polarity_template(
            gold_pairs,
            final_sample_size=args.final_polarity_sample,
            dev_sample_size=args.dev_polarity_sample,
            seed=42,
        )
        polarity.to_csv(GOLD_POLARITY_PATH, index=False, encoding="utf-8-sig")
        print(f"Created polarity annotation file: {GOLD_POLARITY_PATH}")
        print(f"Rows: {len(polarity)}")

    if GOLD_IMPLICIT_PATH.exists() and not args.overwrite:
        print(f"Skip existing: {GOLD_IMPLICIT_PATH}")
    else:
        implicit = build_implicit_template(
            terma,
            dev_sample_size=args.dev_implicit_sample,
            seed=42,
        )
        implicit.to_csv(GOLD_IMPLICIT_PATH, index=False, encoding="utf-8-sig")
        print(f"Created implicit annotation file: {GOLD_IMPLICIT_PATH}")
        print(f"Rows: {len(implicit)}")

    print(
        "\nIMPORTANT: gold_polarity.csv dan gold_implicit.csv yang baru dibuat masih TEMPLATE "
        "annotation. Isi kolom gold/status secara manual sebelum final validation."
    )


def cmd_validate(_: argparse.Namespace) -> None:
    terma, review = load_core()
    print("Validate review...")
    print(validate_review_log(review, terma, require_reviewed=True))

    _need(GOLD_PAIRS_PATH)
    gold_pairs = pd.read_csv(GOLD_PAIRS_PATH)
    print("Validate explicit pairs...")
    print(validate_gold_pairs(gold_pairs, terma, review))

    if GOLD_POLARITY_PATH.exists():
        polarity = pd.read_csv(GOLD_POLARITY_PATH)
        print("Validate polarity...")
        print(validate_polarity_gold(polarity, require_reviewed=True))
    else:
        print("gold_polarity.csv: belum ada")

    if GOLD_IMPLICIT_PATH.exists():
        implicit = pd.read_csv(GOLD_IMPLICIT_PATH)
        print("Validate implicit...")
        print(validate_implicit_gold(implicit, require_reviewed=True))
    else:
        print("gold_implicit.csv: belum ada")

    print("\nH2 VALIDATION FINISHED")


def cmd_prepare(args: argparse.Namespace) -> None:
    cmd_pairs(args)
    print("\n" + "=" * 72 + "\n")
    cmd_splits(args)
    print("\n" + "=" * 72 + "\n")
    cmd_templates(args)


def cmd_auto_1a1o(args: argparse.Namespace) -> None:
    terma, review = load_core()
    if AUTO_1A1O_PATH.exists() and not args.overwrite:
        auto = pd.read_csv(AUTO_1A1O_PATH)
        print(f"Validate existing (tidak ditimpa): {AUTO_1A1O_PATH}")
    else:
        auto = build_auto_1a1o_sheet(terma, review)
        auto.to_csv(AUTO_1A1O_PATH, index=False, encoding="utf-8-sig")
        print(f"Created: {AUTO_1A1O_PATH}")
    print(json.dumps(validate_auto_1a1o(auto, terma, review), indent=2))


def cmd_status(_: argparse.Namespace) -> None:
    status = h2_status(PROJECT_ROOT)
    print(f"Project root: {PROJECT_ROOT}\n")
    for name, exists in status.items():
        mark = "OK" if exists else "MISSING"
        print(f"{mark:7} {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SimplEx-ID H2 gold-data pipeline runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pairs", help="Validate review log and generate gold_pairs.csv")
    sub.add_parser("splits", help="Generate development/final split files")

    p_templates = sub.add_parser(
        "templates",
        help="Generate polarity and implicit annotation templates",
    )
    p_templates.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing gold_polarity.csv / gold_implicit.csv",
    )
    p_templates.add_argument("--final-polarity-sample", type=int, default=600)
    p_templates.add_argument("--dev-polarity-sample", type=int, default=400)
    p_templates.add_argument("--dev-implicit-sample", type=int, default=250)

    p_prepare = sub.add_parser(
        "prepare",
        help="Run pairs + splits + annotation-template preparation",
    )
    p_prepare.add_argument("--overwrite", action="store_true")
    p_prepare.add_argument("--final-polarity-sample", type=int, default=600)
    p_prepare.add_argument("--dev-polarity-sample", type=int, default=400)
    p_prepare.add_argument("--dev-implicit-sample", type=int, default=250)

    p_auto = sub.add_parser(
        "auto-1a1o",
        help="Create/validate the development AUTO 1A1O gold sheet",
    )
    p_auto.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate gold_dev_auto_1a1o.csv (ERASES manual verification)",
    )

    sub.add_parser("validate", help="Validate all H2 final files that exist")
    sub.add_parser("status", help="Show file-level H2 progress")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "pairs": cmd_pairs,
        "splits": cmd_splits,
        "templates": cmd_templates,
        "prepare": cmd_prepare,
        "validate": cmd_validate,
        "status": cmd_status,
        "auto-1a1o": cmd_auto_1a1o,
    }
    try:
        commands[args.command](args)
    except (GoldValidationError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
