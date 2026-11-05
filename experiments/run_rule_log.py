"""
Log "satu perubahan, satu pengukuran" untuk rules (H4.2-H4.3), hanya pada dev.

Setiap langkah menghidupkan satu opsi RuleOptions di atas langkah sebelumnya, pada
varian A2 (R0-R7 + R3 + R4), lalu mengukur selisih F1 terhadap langkah sebelumnya.
Hasil: reports/h4/rule_changes.md dan rule_changes.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplexid.aspect_vocab import crossfit_vocabularies, fold_of  # noqa: E402
from simplexid.config import VARIANTS  # noqa: E402
from simplexid.evaluation.gold_loader import load_eval_gold  # noqa: E402
from simplexid.evaluation.matching import from_extraction_results  # noqa: E402
from simplexid.evaluation.metrics import (  # noqa: E402
    extraction_counts,
    per_rule_precision,
    summarize_extraction,
)
from simplexid.evaluation.significance import paired_bootstrap_f1  # noqa: E402
from simplexid.lexicon import DEFAULT_LEXICON, load_lexicon  # noqa: E402
from simplexid.parse_cache import load_parse_cache  # noqa: E402
from simplexid.pipeline import run_pipeline  # noqa: E402
from simplexid.rules import BASE_RULES  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "h4"
STEPS = [
    ("base", None, "Perilaku awal"),
    ("passive_subject", "H4.2", "R1 menerima nsubj:pass"),
    ("acl_relative", "H4.2", "R7 memakai acl"),
    ("extended_aspect_pos", "H4.2", "Aspek boleh PROPN/X"),
    ("lexicon_opinion", "H4.2", "Kata leksikon ber-POS NOUN/NUM boleh jadi opini"),
    ("extended_r4", "H4.2", "R4 menerima VERB & kata leksikon"),
    ("phrase_spans", "H4.2", "Span aspek/opini mencakup compound & modifier"),
    ("subjective_verbs", "H4.3", "VERB opini R1/R2/R5/R7 harus di leksikon atau dimodifikasi"),
    ("distribution_guard", "H4.3", "R3/R4 melewati konjungtor yang punya pasangan sendiri"),
    ("aspect_vocabulary", "H4.3", "Head aspek harus di kosakata train (cross-fit)"),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    gold = load_eval_gold("dev", PROJECT_ROOT)
    terma = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "terma_processed.csv")
    parses = load_parse_cache(PROJECT_ROOT / "data" / "cache" / "parses", terma_df=terma)
    lexicon = load_lexicon(DEFAULT_LEXICON)
    folds = crossfit_vocabularies(terma, parses)
    variant = VARIANTS["A2"]

    rows, previous, options = [], None, BASE_RULES
    for option, phase, description in STEPS:
        if option != "base":
            options = replace(options, **{option: True})
        preds = {
            sid: from_extraction_results(run_pipeline(
                gold.tokens[sid], variant, "hotel", lexicon, parsed_sentence=parses[sid],
                rules=options, aspect_vocabulary=folds[fold_of(sid)]))
            for sid in gold.sentence_ids
        }
        head = extraction_counts(gold, preds, "head")
        summary = summarize_extraction(gold, head)
        strict = summarize_extraction(gold, extraction_counts(gold, preds, "strict"))["overall"]
        overall = summary["overall"]
        row = {
            "option": option, "phase": phase, "description": description,
            "head_p": overall["precision"], "head_r": overall["recall"], "head_f1": overall["f1"],
            "strict_f1": strict["f1"],
            "fp_empty": summary["by_stratum"]["EMPTY"]["fp"],
            "complex_f1": summary["by_stratum"]["COMPLEX"]["f1"],
            "simple_f1": summary["by_stratum"]["SIMPLE"]["f1"],
            "per_rule": per_rule_precision(gold, preds, "head"),
        }
        if previous is not None:
            test = paired_bootstrap_f1(previous, head, n_resamples=2000)
            row.update(delta_f1=test["delta"], ci95=test["ci95"], p_value=test["p_value"])
        rows.append(row)
        previous = head
        print(f"{option:22} P={row['head_p']:.3f} R={row['head_r']:.3f} F1={row['head_f1']:.3f} "
              f"strict={row['strict_f1']:.3f} FP_empty={row['fp_empty']} dF1={row.get('delta_f1', 0):+.4f}")

    def evaluate_options(opts):
        preds = {
            sid: from_extraction_results(run_pipeline(
                gold.tokens[sid], variant, "hotel", lexicon, parsed_sentence=parses[sid],
                rules=opts, aspect_vocabulary=folds[fold_of(sid)]))
            for sid in gold.sentence_ids
        }
        return extraction_counts(gold, preds, "head"), summarize_extraction(gold, extraction_counts(gold, preds, "head"))

    # Leave-one-out dari set penuh: dampak mematikan satu opsi, dasar keputusan DEFAULT_RULES.
    full_counts, _ = evaluate_options(options)
    leave_one_out = []
    for option, _, description in STEPS[1:]:
        counts, summary = evaluate_options(replace(options, **{option: False}))
        test = paired_bootstrap_f1(counts, full_counts, n_resamples=2000)
        overall = summary["overall"]
        leave_one_out.append({"option": option, "description": description, "f1_without": overall["f1"],
                              "p_without": overall["precision"], "r_without": overall["recall"],
                              "gain_from_option": test["delta"], "ci95": test["ci95"], "p_value": test["p_value"]})
        print(f"without {option:22} F1={overall['f1']:.3f} gain={test['delta']:+.4f} {test['ci95']} p={test['p_value']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rule_changes.json").write_text(json.dumps(
        {"split": "dev", "variant": "A2", "lexicon": DEFAULT_LEXICON, "gold": gold.summary(),
         "final_options": asdict(options), "steps": rows, "leave_one_out": leave_one_out}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# H4 rule change log (development only)", "",
        f"Generated by `experiments/run_rule_log.py`. Variant A2, lexicon `{DEFAULT_LEXICON}`, gold `{gold.version}`.",
        "Each row adds one option on top of the previous row. Δ and CI are head F1 vs the previous row (bootstrap 2,000).",
        "", "| Option | Phase | head P | head R | head F1 | Δ F1 | 95% CI | p | strict F1 | FP on empty gold |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['option']}` | {r['phase'] or '-'} | {r['head_p']:.3f} | {r['head_r']:.3f} | **{r['head_f1']:.3f}** | "
            f"{r.get('delta_f1', 0):+.4f} | {r.get('ci95', '-')} | {r.get('p_value', '-')} | {r['strict_f1']:.3f} | {r['fp_empty']} |")
    lines += ["", "## Leave-one-out from the full option set", "",
              "Gain = head F1 with the option minus head F1 without it (bootstrap 2,000). "
              "Only options with a positive gain enter `DEFAULT_RULES`, except `phrase_spans`, which only affects strict matching.",
              "", "| Option | F1 without | Gain from option | 95% CI | p |", "|---|---|---|---|---|"]
    for r in leave_one_out:
        lines.append(f"| `{r['option']}` | {r['f1_without']:.3f} | {r['gain_from_option']:+.4f} | {r['ci95']} | {r['p_value']} |")
    lines += ["", "## Per-rule head precision at the last step", "", "| Rule | Predicted | Correct | Precision |", "|---|---|---|---|"]
    for rule, v in rows[-1]["per_rule"].items():
        lines.append(f"| {rule} | {v['predicted']} | {v['correct']} | {v['precision']:.3f} |")
    (OUT_DIR / "rule_changes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'rule_changes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
