## H2 — Gold Data Preparation

### Dataset
- TermA processed: 5,000 sentences
- Development split: 3,000 TermA train sentences
- Final local test: 1,000 TermA validation sentences
- Official TermA test tidak digunakan untuk local gold metrics.

### Explicit Aspect–Opinion Gold
- Gold review: 1,314 sentences
- Gold aspect–opinion pairs: 3,202 pairs
- Duplicate pair: 0
- Missing core values: 0
- Span mismatch: 0
- Validation: PASS

### Polarity Gold
- Gold polarity: 1,000 aspect–opinion pairs
- Validation: PASS

### Implicit Aspect Gold
- Implicit candidates: 451
- Valid implicit gold rows: 407
- Validation: PASS

### Split Policy
- TermA train → development
- TermA validation → locked final evaluation
- Random seed: 42

### H2 Status
PASS

## H3.1 — Packaging & Parser Repair (2026-10-13)

### Defects found and fixed
- UDPipe model in `.venv` was an HTML page (LINDAT moved to DSpace 7; spacy-udpipe 1.0.0
  downloads from the dead legacy URL). Re-downloaded via the new API, MD5 verified
  (`e0167ecc9d9a953bc14726b6e928725e`), stored in `resources/models/udpipe/` with `SOURCE.md`.
- `spacy_udpipe` runs UDPipe as the tokenizer only (`nlp.pipe_names == []`), so the old
  `parse_tokens()` never produced dependency relations for pretokenized TermA input.
  Parser rewritten on `ufal.udpipe` directly ("horizontal" input keeps TermA tokenization;
  one TermA row = one tree). Added CoNLL-U export/import for the H3.2 parse cache.
- `ufal.udpipe` Pipeline crashed (abort) when the Python `Model` object was GC'd;
  the parser now keeps a reference.
- Package modules imported `parser`, `rules`, ... as top-level modules, which silently
  resolved to the legacy `src/*.py` copies. Converted to relative imports.

### Structure changes
- Removed legacy `src/*.py` (history in git). `evaluate.py` and `run_experiment.py`
  were not migrated: they evaluated the masked official test split with a heuristic pairing
  and will be rebuilt in H3.3 on top of `gold_pairs/gold_polarity/gold_implicit`.
- `secondary_tagging.py` moved into the package; `test_offline.py` became
  `tests/test_core_offline.py`.

### Notes for later phases
- Model license is CC BY-NC-SA 4.0 (non-commercial): fine for research, not for
  commercial deployment.
- Smoke test on dev sentences shows multi-token span mismatches and PROPN aspects
  (e.g. "AC") ignored by rules that require NOUN — address in H3.3 (matching) and H4 (rules).

### H3.1 Status
PASS — 24 tests passing, package imports without legacy modules.


## H3.2 — Parse Cache, Oracle Upper Bound, Parser Audit (2026-10-19)

### Parse cache
- All 5,000 TermA sentences parsed once (pretokenized, UDPipe id-gsd 2.5) into
  `data/cache/parses/terma.conllu`; every row is exactly one tree (5,000 roots).
- Cache is regenerable and not in git; provenance in `reports/h32/parse_cache_manifest.json`
  (model SHA-256, input SHA-256, cache SHA-256). Loading fails if model, data, or cache change.

### Label inventory (`reports/h32/parse_stats.json`)
- `relcl` occurs 0 times (GSD 2.5 uses `acl`): **R7 can never fire** as written.
- UD v2 has no `neg` label; negation ("tidak") is `advmod` + UPOS `PART`.
- `nsubj:pass` (848) is not matched by R1, which checks `nsubj` exactly.

### Oracle — development split only (`reports/h32/oracle_dev.md`)
- 1,593 gold pairs (1,125 reviewed + 468 trivial 1-aspect-1-opinion, reported separately).
- Recall upper bound by aspect-head↔opinion-head distance: ≤1 = 73.5%, ≤2 = 94.0%, ≤3 = 98.9%.
- Current rules (R0–R7 + R3 + R4), head-match preview: **51.0%** → ~43 pts headroom within distance 2.
- Pairs whose path uses `conj`: 195 (12.2%); current rules reach only 30.8% of them
  (relevant to the R3/R4 claim).
- Largest misses: opinion words tagged NOUN/NUM by the parser ("kotor", "oke", "bau")
  attached via `compound`/`nummod` (≈180 pairs); `nsubj:pass` (37/40 missed); `acl` (28/29).
  23.8% of opinion heads are not ADJ/VERB; 11.3% of aspect heads are not NOUN.
- Validation-split oracle intentionally NOT computed (allowed only in H6 via `--allow-final`).

### Parser audit
- `data/annotation/parser_audit/parser_audit_sheet.csv`: 100 dev sentences (25 per stratum
  COORD / NEGATION / MULTI_ASPECT / OTHER), 1,144 tokens; guide in the same folder.
  Status: awaiting human annotation.

### H3.2 Status
PASS (automated parts) — 31 tests passing. Parser audit annotation pending.


## H3.3 — Evaluation Harness (2026-10-19)

### Evaluation gold
- Final (TermA validation): 1,000 sentences, 2,077 pairs, 100% reviewed (822 spanned + 178 empty).
  Loader fails if any spanned final sentence is unreviewed. Gold version `4a7ac755fa0a90e8`.
- Development (TermA train): 1,478 sentences / 1,593 pairs = 492 reviewed + 468 AUTO 1A1O
  + 518 without spans. 1,522 unreviewed complex sentences excluded. Gold version `4f58f591c82b18c7`.
- AUTO 1A1O decision (user-approved 2026-10-19): dev only, never final. Evidence: 228/228 reviewed
  1A1O sentences were A1-O1 (rule-of-three upper error bound ≈1.3%). Sheet
  `data/gold/gold_dev_auto_1a1o.csv` (status AUTO) is being verified by the user; rows set to
  REVIEWED change the gold source and version hash.

### Harness
- Matching strict / head (primary) / overlap, one-to-one maximum bipartite.
- Polarity: end-to-end, conditional, oracle-pair. Implicit: detection + category vs dev-majority baseline.
- Paired bootstrap (10k, seed 42), exact McNemar, Holm. Final split locked behind
  `configs/frozen.lock`; final runs logged to `results/final_runs.log`.

### Baseline on dev (`results/20261019-122748_dev_h33-baseline`, commit 549de12, clean tree, placeholder lexicon)
| | head P | head R | head F1 |
|---|---|---|---|
| V0 | 0.295 | 0.471 | 0.363 |
| V3 | 0.291 | 0.510 | 0.371 |
| V4–V9 | identical to V3 on extraction |

- Consistency check: V3 head recall 0.510 equals the independent H3.2 oracle preview (51.0%).
- Precision is the main weakness: 707 FPs on 519 sentences whose gold has no pair; per-rule head
  precision R2 (obj) 11% on 712 predictions, R3 12%, R1 25%, R0 40%, R4 61%.
- R3 lowers F1 (V0 0.363 → V1 0.358), R4 raises it (V2 0.376).
- Polarity oracle-pair accuracy 0.282 for every variant: the 17-word placeholder lexicon leaves most
  opinions at score 0, so DW scoring (V6 vs V7) and scope consistency (V8 vs V9) cannot show any
  effect yet. Not reportable.
- Implicit: R8 recall 3.5% vs majority baseline accuracy 75.5%; V4 ≡ V5 as predicted
  (no context for disambiguation in sentence-level implicit candidates).
- Duplicated variants confirmed empirically (V5 ≡ V6, V7 ≡ V8, V8 ≡ V9 on all metrics).

### H3.3 Status
PASS — 53 tests passing; reproducible baseline recorded. Inputs for H4: precision on no-span sentences,
R2/R3 precision, lexicon, ablation redesign, R8 redesign.
