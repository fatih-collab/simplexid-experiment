from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplexid.oracle import (  # noqa: E402
    SOURCE_AUTO_1A1O,
    SOURCE_REVIEWED,
    build_dev_gold,
    oracle_records,
    rule_reach,
    summarize,
)
from simplexid.parse_cache import ParseCacheError, build_parse_cache, load_parse_cache  # noqa: E402
from simplexid.parser import UDPipeParser  # noqa: E402
from simplexid.rules import expand_r3, expand_r4, extract_explicit_pairs  # noqa: E402

TERMA_PATH = PROJECT_ROOT / "data" / "processed" / "terma_processed.csv"
GOLD_PAIRS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_pairs.csv"
REVIEW_PATH = PROJECT_ROOT / "data" / "gold" / "gold_review_log.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "parses"
REPORT_DIR = PROJECT_ROOT / "reports" / "h32"
AUDIT_DIR = PROJECT_ROOT / "data" / "annotation" / "parser_audit"
AUDIT_SHEET = AUDIT_DIR / "parser_audit_sheet.csv"

NEGATION_CUES = {"tidak", "tak", "bukan", "belum", "kurang", "gak", "ga", "nggak", "enggak", "tanpa"}


def load_terma() -> pd.DataFrame:
    return pd.read_csv(TERMA_PATH)


def load_cache(terma: pd.DataFrame) -> dict:
    return load_parse_cache(CACHE_DIR, terma_df=terma)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_cache(args: argparse.Namespace) -> None:
    terma = load_terma()
    if not args.rebuild:
        try:
            load_cache(terma)
            print(f"Cache valid, tidak dibuat ulang: {CACHE_DIR} (pakai --rebuild untuk memaksa)")
            return
        except ParseCacheError as exc:
            print(f"Cache dibuat ulang: {exc}")
    parser = UDPipeParser()
    manifest = build_parse_cache(
        terma, CACHE_DIR, parser,
        progress=lambda i, n: print(f"  parsed {i}/{n}"),
    )
    print(json.dumps(manifest, indent=2))
    load_cache(terma)  # verifikasi baca-ulang
    # Cache tidak masuk git (bisa dibuat ulang); manifest-nya dicatat sebagai bukti provenance.
    write_json(REPORT_DIR / "parse_cache_manifest.json", manifest)
    print(f"\nSaved & verified: {CACHE_DIR}")


def cmd_stats(_: argparse.Namespace) -> None:
    """Inventaris label/POS seluruh split -- tidak memakai label gold."""
    terma = load_terma()
    parses = load_cache(terma)
    split_of = dict(zip(terma["sentence_id"], terma["source_split"]))

    deps, pos = Counter(), Counter()
    roots_per_sentence = Counter()
    n_tokens = Counter()
    for sentence_id, sent in parses.items():
        split = split_of[sentence_id]
        n_tokens[split] += len(sent.tokens)
        roots_per_sentence[sum(t.head_idx == t.idx for t in sent.tokens)] += 1
        deps.update(t.dep for t in sent.tokens)
        pos.update(t.pos for t in sent.tokens)

    labels_used_by_rules = ["amod", "nsubj", "obj", "obl", "compound", "relcl", "acl", "conj", "advmod"]
    stats = {
        "n_sentences": len(parses),
        "n_tokens_by_split": dict(n_tokens),
        "roots_per_sentence": dict(roots_per_sentence),
        "dep_labels": dict(deps.most_common()),
        "upos": dict(pos.most_common()),
        "rule_label_presence": {label: deps.get(label, 0) for label in labels_used_by_rules},
    }
    write_json(REPORT_DIR / "parse_stats.json", stats)
    print(json.dumps(stats["rule_label_presence"], indent=2))
    print(f"\nSaved: {REPORT_DIR / 'parse_stats.json'}")


def cmd_oracle(args: argparse.Namespace) -> None:
    if args.split != "train" and not args.allow_final:
        raise ValueError(
            "Oracle pada split final hanya boleh di H6 (setelah config dibekukan). "
            "Pakai --allow-final jika memang sudah di H6."
        )
    terma = load_terma()
    parses = load_cache(terma)
    gold = build_dev_gold(terma, pd.read_csv(GOLD_PAIRS_PATH), pd.read_csv(REVIEW_PATH), split=args.split)
    records = oracle_records(gold, parses)

    def current_rules(sent):
        pairs = extract_explicit_pairs(sent)
        return pairs + expand_r3(sent, pairs) + expand_r4(sent, pairs)

    reach = rule_reach(gold, parses, current_rules)
    rec_df = pd.DataFrame([r.__dict__ for r in records])
    rec_df["current_rules_head_match"] = reach

    tag = "dev" if args.split == "train" else args.split
    summary = {"split": args.split, "all": summarize(records)}
    for source in (SOURCE_REVIEWED, SOURCE_AUTO_1A1O):
        subset = [r for r in records if r.source == source]
        if subset:
            summary[source] = summarize(subset)
    summary["current_rules_head_match_recall"] = {
        "all": round(float(rec_df["current_rules_head_match"].mean()), 4),
        **{s: round(float(g["current_rules_head_match"].mean()), 4)
           for s, g in rec_df.groupby("source")},
    }

    write_json(REPORT_DIR / f"oracle_{tag}.json", summary)
    rec_df.to_csv(REPORT_DIR / f"oracle_{tag}_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summarize(records, top_patterns=200)["top_patterns"]).to_csv(
        REPORT_DIR / f"oracle_{tag}_patterns.csv", index=False, encoding="utf-8-sig"
    )
    _write_oracle_markdown(REPORT_DIR / f"oracle_{tag}.md", summary)
    print((REPORT_DIR / f"oracle_{tag}.md").read_text(encoding="utf-8"))


def _write_oracle_markdown(path: Path, summary: dict) -> None:
    s = summary["all"]
    lines = [
        f"# H3.2 Oracle report — split `{summary['split']}`",
        "",
        "Generated by `experiments/run_parse.py oracle`. Do not edit by hand.",
        "",
        f"Gold pairs analysed: **{s['n_pairs']}** "
        + ", ".join(f"{k}={summary[k]['n_pairs']}" for k in (SOURCE_REVIEWED, SOURCE_AUTO_1A1O) if k in summary),
        "",
        "## Recall upper bound by dependency distance (aspect head ↔ opinion head)",
        "",
        "| Max distance | All | " + " | ".join(k for k in (SOURCE_REVIEWED, SOURCE_AUTO_1A1O) if k in summary) + " |",
        "|---|---|" + "---|" * sum(k in summary for k in (SOURCE_REVIEWED, SOURCE_AUTO_1A1O)),
    ]
    for k, v in s["recall_upper_bound_by_distance"].items():
        extra = " | ".join(
            f"{summary[src]['recall_upper_bound_by_distance'][k]:.1%}"
            for src in (SOURCE_REVIEWED, SOURCE_AUTO_1A1O) if src in summary
        )
        lines.append(f"| {k} | {v:.1%} | {extra} |")
    reach = summary["current_rules_head_match_recall"]
    lines += [
        "",
        "## Current rules (R0–R7 + R3 + R4), head-match recall preview",
        "",
        ", ".join(f"{k}: **{v:.1%}**" for k, v in reach.items()),
        "",
        "Preview only — the official metric is computed by the H3.3 harness.",
        "",
        "## Head POS",
        "",
        f"- Aspect heads: {s['aspect_head_pos']}",
        f"- Opinion heads: {s['opinion_head_pos']}",
        "",
        "## Distance distribution",
        "",
        f"{s['distance_distribution']}",
        "",
        "## Top dependency path patterns (aspect → opinion)",
        "",
        "`^dep` = up from child to head, `vdep` = down from head to child.",
        "",
        "| # | Pattern | Count | Share | Cumulative |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(s["top_patterns"], start=1):
        lines.append(f"| {i} | `{row['pattern']}` | {row['count']} | {row['share']:.1%} | {row['cumulative']:.1%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_audit_sheet(args: argparse.Namespace) -> None:
    if AUDIT_SHEET.exists() and not args.overwrite:
        print(f"Skip existing (berisi anotasi?): {AUDIT_SHEET}  -- pakai --overwrite untuk membuat ulang")
        return
    terma = load_terma()
    parses = load_cache(terma)
    dev = terma[(terma["source_split"] == "train")].copy()
    dev["n_tok"] = dev["tokens"].apply(lambda t: len(json.loads(t)))
    dev = dev[(dev["n_tok"] >= 4) & (dev["n_tok"] <= 25)]

    def stratum(row) -> str:
        sent = parses[row.sentence_id]
        words = {t.text.lower() for t in sent.tokens}
        if any(t.dep == "conj" for t in sent.tokens):
            return "COORD"
        if words & NEGATION_CUES:
            return "NEGATION"
        if (row.n_aspects or 0) > 1:
            return "MULTI_ASPECT"
        return "OTHER"

    dev["stratum"] = [stratum(r) for r in dev.itertuples(index=False)]
    per = args.n // 4
    chosen = []
    for i, name in enumerate(["COORD", "NEGATION", "MULTI_ASPECT", "OTHER"]):
        pool = dev[dev["stratum"] == name]
        chosen.append(pool.sample(n=min(per, len(pool)), random_state=42 + i))
    sample = pd.concat(chosen).sort_values("sentence_id")

    rows = []
    for row in sample.itertuples(index=False):
        sent = parses[row.sentence_id]
        for t in sent.tokens:
            is_root = t.head_idx == t.idx
            rows.append({
                "sentence_id": row.sentence_id,
                "stratum": row.stratum,
                "sentence": row.sentence,
                "token_no": t.idx + 1,
                "token": t.text,
                "upos": t.pos,
                "head_no": 0 if is_root else t.head_idx + 1,
                "head_token": "ROOT" if is_root else sent.tokens[t.head_idx].text,
                "deprel": t.dep,
                "upos_ok": "",
                "head_ok": "",
                "deprel_ok": "",
                "correct_head_no": "",
                "correct_deprel": "",
                "notes": "",
            })
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUDIT_SHEET, index=False, encoding="utf-8-sig")
    print(f"Created: {AUDIT_SHEET}")
    print(f"Sentences: {sample['sentence_id'].nunique()}  tokens: {len(rows)}")
    print(sample["stratum"].value_counts().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimplEx-ID H3.2 parse cache & parser audit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cache", help="Parse TermA once and write the CoNLL-U cache")
    p.add_argument("--rebuild", action="store_true")
    sub.add_parser("stats", help="Dependency label / UPOS inventory (no gold labels used)")
    p = sub.add_parser("oracle", help="Oracle upper bound and path patterns for gold pairs")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--allow-final", action="store_true")
    p = sub.add_parser("audit-sheet", help="Create the manual parser-audit annotation sheet")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    # Konsol Windows default cp1252 tidak bisa mencetak karakter laporan (↔, →, …).
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    commands = {
        "cache": cmd_cache,
        "stats": cmd_stats,
        "oracle": cmd_oracle,
        "audit-sheet": cmd_audit_sheet,
    }
    try:
        commands[args.command](args)
    except (ParseCacheError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
