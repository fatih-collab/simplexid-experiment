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

from simplexid.config import KEY_COMPARISONS, VARIANTS  # noqa: E402
from simplexid.evaluation.gold_loader import load_eval_gold  # noqa: E402
from simplexid.evaluation.matching import LEVELS, from_extraction_results  # noqa: E402
from simplexid.evaluation.metrics import (  # noqa: E402
    extraction_counts,
    implicit_items,
    per_rule_precision,
    polarity_items,
    summarize_extraction,
    summarize_implicit,
    summarize_polarity,
)
from simplexid.evaluation.run_record import (  # noqa: E402
    FrozenConfigError,
    check_frozen,
    freeze,
    log_final_run,
    new_run_dir,
    write_manifest,
)
from simplexid.evaluation.significance import holm, mcnemar_exact, paired_bootstrap_f1  # noqa: E402
from simplexid.gold import GoldValidationError  # noqa: E402
from simplexid.lexicon import DEFAULT_LEXICON, load_lexicon  # noqa: E402
from simplexid.oracle import span_head  # noqa: E402
from simplexid.parse_cache import ParseCacheError, load_parse_cache  # noqa: E402
from simplexid.pipeline import run_pipeline  # noqa: E402
from simplexid.scoring import compute_factors  # noqa: E402

DOMAIN = "hotel"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "parses"
POLARITY_PATH = PROJECT_ROOT / "data" / "gold" / "gold_polarity.csv"
IMPLICIT_PATH = PROJECT_ROOT / "data" / "gold" / "gold_implicit.csv"


def majority_implicit_category(implicit_df: pd.DataFrame) -> str:
    """Baseline mayoritas SELALU ditentukan dari development, tidak pernah dari final."""
    dev = implicit_df[(implicit_df["implicit_scope"] == "DEV_IMPLICIT")
                      & (implicit_df["decision"].str.upper() == "IMPLICIT")]
    return dev["gold_aspect_category"].value_counts().idxmax()


def evaluate(split: str, variant_names, lexicon_name: str, label: str, n_resamples: int) -> Path:
    gold = load_eval_gold(split, PROJECT_ROOT)
    terma = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "terma_processed.csv")
    parses = load_parse_cache(CACHE_DIR, terma_df=terma)
    lexicon = load_lexicon(lexicon_name)
    polarity_df = pd.read_csv(POLARITY_PATH)
    implicit_df = pd.read_csv(IMPLICIT_PATH)
    baseline_category = majority_implicit_category(implicit_df)

    run_dir = new_run_dir(PROJECT_ROOT, split, label)
    print(f"Run dir: {run_dir}")
    print(json.dumps(gold.summary(), indent=2))

    metrics, counts_head, pol_items, imp_items = {}, {}, {}, {}
    for name in variant_names:
        variant = VARIANTS[name]
        preds = {
            sid: from_extraction_results(run_pipeline(
                gold.tokens[sid], variant, DOMAIN, lexicon, parsed_sentence=parses[sid]))
            for sid in gold.sentence_ids
        }
        with (run_dir / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as f:
            for sid in gold.sentence_ids:
                f.write(json.dumps({"sentence_id": sid, "variant": name,
                                    "pairs": [p.to_dict() for p in preds[sid]]}, ensure_ascii=False) + "\n")

        def oracle_score(sid, aspect_span, opinion_span, _v=variant):
            sent = parses[sid]
            head = span_head(sent, *opinion_span)
            return compute_factors(sent, head, lexicon, distance_weighted=_v.distance_weighted_formula).score_final

        extraction = {}
        for level in LEVELS:
            per_sentence = extraction_counts(gold, preds, level)
            extraction[level] = summarize_extraction(gold, per_sentence)
            if level == "head":
                counts_head[name] = per_sentence
        pol_items[name] = polarity_items(gold, preds, polarity_df, oracle_score)
        imp_items[name] = implicit_items(gold, preds, implicit_df)
        metrics[name] = {
            "description": variant.description,
            "extraction": extraction,
            "per_rule_precision_head": per_rule_precision(gold, preds, "head"),
            "polarity": summarize_polarity(pol_items[name]),
            "implicit": summarize_implicit(imp_items[name], baseline_category),
        }
        h = extraction["head"]["overall"]
        print(f"  {name}: head P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f} | "
              f"polarity oracle acc={metrics[name]['polarity']['oracle_pair']['accuracy']:.3f} | "
              f"implicit cat acc={metrics[name]['implicit']['category']['accuracy']:.3f}")

    significance = compare(variant_names, counts_head, pol_items, imp_items, n_resamples)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    significance.to_csv(run_dir / "significance.csv", index=False, encoding="utf-8-sig")
    for name in variant_names:
        pol_items[name].to_csv(run_dir / f"polarity_items_{name}.csv", index=False, encoding="utf-8-sig")
        imp_items[name].to_csv(run_dir / f"implicit_items_{name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"sentence_id": s, "stratum": gold.stratum[s], "source": gold.source[s], **vars(c)}
                      for s, c in counts_head[name].items()]).to_csv(
            run_dir / f"sentence_counts_head_{name}.csv", index=False, encoding="utf-8-sig")
    write_manifest(run_dir, PROJECT_ROOT, {
        "split": split,
        "label": label,
        "variants": {n: vars(VARIANTS[n]) for n in variant_names},
        "lexicon": lexicon_name,
        "lexicon_is_placeholder": lexicon_name == DEFAULT_LEXICON,
        "gold": gold.summary(),
        "implicit_majority_baseline": baseline_category,
        "bootstrap_resamples": n_resamples,
    })
    write_summary(run_dir, split, metrics, significance, lexicon_name)
    print(f"\nSaved: {run_dir}")
    return run_dir


def compare(variant_names, counts_head, pol_items, imp_items, n_resamples) -> pd.DataFrame:
    rows = []
    for a, b, desc in KEY_COMPARISONS:
        if a not in variant_names or b not in variant_names:
            continue
        rows.append({"comparison": f"{a}_vs_{b}", "description": desc, "test": "paired_bootstrap",
                     "target": "pair_f1_head",
                     **paired_bootstrap_f1(counts_head[a], counts_head[b], n_resamples=n_resamples)})
        for mode in ("pred_oracle_pair", "pred_end_to_end"):
            pa, pb = pol_items[a], pol_items[b]
            rows.append({"comparison": f"{a}_vs_{b}", "description": desc, "test": "mcnemar_exact",
                         "target": f"polarity_{mode.removeprefix('pred_')}",
                         **mcnemar_exact((pa[mode] == pa["gold"]).tolist(), (pb[mode] == pb["gold"]).tolist())})
        ia = imp_items[a][imp_items[a]["decision"] == "IMPLICIT"]
        ib = imp_items[b][imp_items[b]["decision"] == "IMPLICIT"]
        rows.append({"comparison": f"{a}_vs_{b}", "description": desc, "test": "mcnemar_exact",
                     "target": "implicit_category",
                     **mcnemar_exact((ia["pred_category"] == ia["gold_category"]).tolist(),
                                     (ib["pred_category"] == ib["gold_category"]).tolist())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["p_value"].tolist())
    return df


def write_summary(run_dir: Path, split: str, metrics: dict, significance: pd.DataFrame, lexicon: str) -> None:
    lines = [f"# Evaluation summary — `{split}`", "",
             f"Run: `{run_dir.name}` · lexicon: `{lexicon}`", ""]
    if lexicon == DEFAULT_LEXICON:
        lines += ["> **Polarity numbers use the PLACEHOLDER lexicon and are not reportable.**", ""]
    lines += ["## Pair extraction (micro)", "",
              "| Variant | strict F1 | head P | head R | **head F1** | overlap F1 |",
              "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        e = m["extraction"]
        h = e["head"]["overall"]
        lines.append(f"| {name} | {e['strict']['overall']['f1']:.3f} | {h['precision']:.3f} | {h['recall']:.3f} | "
                     f"**{h['f1']:.3f}** | {e['overlap']['overall']['f1']:.3f} |")
    lines += ["", "## Head F1 by stratum", ""]
    strata = sorted({s for m in metrics.values() for s in m["extraction"]["head"]["by_stratum"]})
    lines += ["| Variant | " + " | ".join(strata) + " |", "|---|" + "---|" * len(strata)]
    for name, m in metrics.items():
        by = m["extraction"]["head"]["by_stratum"]
        cells = []
        for s in strata:
            if s not in by:
                cells.append("-")
            elif s == "EMPTY":  # gold tanpa pasangan: hanya FP yang bermakna
                cells.append(f"FP={by[s]['fp']} ({by[s]['n_sentences']} sent.)")
            else:
                cells.append(f"P={by[s]['precision']:.2f} R={by[s]['recall']:.2f} F1={by[s]['f1']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Polarity (gold_polarity subset) and implicit aspects", "",
              "| Variant | oracle-pair acc | end-to-end acc | conditional acc (coverage) | implicit detect F1 | implicit cat acc | majority baseline acc |",
              "|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        p, i = m["polarity"], m["implicit"]
        lines.append(f"| {name} | {p['oracle_pair']['accuracy']:.3f} | {p['end_to_end']['accuracy']:.3f} | "
                     f"{p['conditional']['accuracy']:.3f} ({p['conditional']['coverage']:.0%}) | "
                     f"{i['detection']['f1']:.3f} | {i['category']['accuracy']:.3f} | "
                     f"{i['majority_baseline']['accuracy']:.3f} |")
    if not significance.empty:
        lines += ["", "## Key comparisons (B − A)", "",
                  "| Comparison | Target | Δ | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
        for r in significance.itertuples(index=False):
            ci = r.ci95 if isinstance(r.ci95, list) else "-"
            lines.append(f"| {r.comparison} | {r.target} | {r.delta:+.4f} | {ci} | {r.p_value:.4f} | {r.p_holm:.4f} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_dev(args):
    evaluate("dev", args.variants, args.lexicon, args.label, args.resamples)


def cmd_final(args):
    check_frozen(PROJECT_ROOT)
    run_dir = evaluate("final", args.variants, args.lexicon, args.label, args.resamples)
    log_final_run(PROJECT_ROOT, run_dir, args.variants)


def cmd_freeze(args):
    print(json.dumps(freeze(PROJECT_ROOT, note=args.note), indent=2))


def cmd_gold_summary(args):
    print(json.dumps(load_eval_gold(args.split, PROJECT_ROOT).summary(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimplEx-ID H3.3 evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("dev", "Evaluate variants on development"),
                            ("final", "Evaluate on the LOCKED final test (requires freeze)")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
        p.add_argument("--lexicon", default=DEFAULT_LEXICON)
        p.add_argument("--label", default="eval")
        p.add_argument("--resamples", type=int, default=10_000)
    p = sub.add_parser("freeze", help="Freeze code/resources/gold hashes before the final run (H6)")
    p.add_argument("--note", required=True, help="Why the configuration is being frozen")
    p = sub.add_parser("gold-summary", help="Show the evaluation gold universe for a split")
    p.add_argument("--split", default="dev", choices=["dev", "final"])
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    commands = {"dev": cmd_dev, "final": cmd_final, "freeze": cmd_freeze, "gold-summary": cmd_gold_summary}
    try:
        commands[args.command](args)
    except (FrozenConfigError, ParseCacheError, GoldValidationError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
