# Parser audit — annotation guide (H3.2)

`parser_audit_sheet.csv` contains 100 development sentences (TermA train), 25 per stratum
(`COORD`, `NEGATION`, `MULTI_ASPECT`, `OTHER`), one row per token, parsed by
UDPipe `indonesian-gsd-ud-2.5-191206`. The result estimates parser accuracy (UPOS accuracy,
UAS, LAS) on hotel reviews for the *limitations* section and for H4 error analysis.

Regenerate only with `python experiments/run_parse.py audit-sheet --overwrite` — this
**erases annotations**.

## Columns to fill

| Column | Value | Meaning |
|---|---|---|
| `upos_ok` | `1` / `0` | Is `upos` the correct Universal POS for this token in context? |
| `head_ok` | `1` / `0` | Is `head_no` the correct head? (`0` = ROOT) |
| `deprel_ok` | `1` / `0` | Is `deprel` correct **given the correct head**? |
| `correct_head_no` | number | Only when `head_ok = 0` |
| `correct_deprel` | UD label | Only when `deprel_ok = 0` |
| `notes` | free text | Optional |

## Conventions

- Follow UD v2 guidelines for Indonesian: https://universaldependencies.org/id/
- Predicative adjectives without copula ("kamar bersih") are commonly `amod` in UD Indonesian-GSD;
  judge against the GSD convention, not against what SimplEx-ID rules expect.
- Punctuation rows: mark `1` unless clearly attached to the wrong clause.
- Judge the sentence as tokenised by TermA; do not re-tokenise.
- If unsure, mark your best guess and write `UNSURE` in `notes`.

## Recommended

A second annotator labels a subset (e.g. 20 sentences) independently so agreement can be reported.
