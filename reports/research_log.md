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


## H4.0–H4.3 — Ablation Redesign, Lexicon, Rules (2026-11-05)

All numbers are development-only and provisional: `gold_dev_auto_1a1o.csv` is still 468/468 AUTO in
the project (the user reports having reviewed it, but the reviewed file is not in the repository), and
dev gold_polarity/gold_implicit have not been verified. Final test untouched.

### H4.0 Ablation
- Variants: cumulative A0–A6, leave-one-out L_*, S_R4; unique flags enforced by test (this caught
  L_SC ≡ A5 during the redesign). Old mapping: V0=A0, V1=A1, V2=S_R4, V3=A2, V4=A3, V5=A4, V7=A5, V9=A6.
- SC now has a real inheritance baseline. SC can only affect R4 pairs: R3 pairs share the opinion token.
- DW defect found: modifiers were direct children only, so distance was always 1 and DW could never
  change a sign. DW now follows modifier (advmod) chains. Wider traversal changed 3 dev labels, all wrongly.
- Parser-independent gold subsets (auto, to verify in H5): COORD_ASPECT 46, COORD_OPINION 108, ASYM_MOD 33 sentences.

### H4.1 Lexicon (decision D1 applied as recommended; D2 still open)
- `merged_v1` = SentiStrengthID, then ID-OpinionWords (±0.6). Files not in git: no explicit license.
- Oracle-pair polarity accuracy on 400 dev items: placeholder 0.29, ss_v1 0.60, ow_v1 0.60, merged_v1 0.68,
  plus negated-unknown fallback 0.75 (NEGATION stratum 0.53 → 0.81).
- Remaining errors are mostly domain: 71 uncovered words ("ada", "strategis"), and wrong entries for hotels
  ("lengkap", "terjangkau", "lancar" negative; "dingin" negative though "ac dingin" is positive). Argues for D2.

### H4.2–H4.3 Rules (`reports/h4/rule_changes.md`)
- A2 head F1 0.371 → 0.588 (P 0.291 → 0.617, R 0.510 → 0.561); strict F1 0.194 → 0.418; FP on no-pair
  sentences 739 → 148.
- Leave-one-out gains: aspect_vocabulary +0.109, lexicon_opinion +0.054, subjective_verbs +0.039,
  distribution_guard +0.008, extended_aspect_pos +0.007, passive_subject +0.005 (all p<0.05);
  acl_relative +0.003 and extended_r4 +0.001 not significant; phrase_spans only affects strict.
- Caveat: the aspect vocabulary comes from TermA train annotations (scored cross-fit on dev), so the full
  system is not annotation-free. Without it, A2 head F1 = 0.478 (`--no-aspect-vocab`). Claims must say which.

### Ablation on dev (`results/20261105-182511_dev_h43-ablation`, commit 7c835f2, clean tree)
| Mechanism | Comparison | Result |
|---|---|---|
| R3 | A0→A1 overall | +0.001 F1, n.s. |
| R3 | on COORD_ASPECT | +0.058 F1, CI [0.020, 0.101], p_Holm 0.013 |
| R4 | A1→A2 overall | +0.017 F1, CI [0.011, 0.024], p_Holm <0.001 |
| R4 | on COORD_OPINION | +0.093 F1, CI [0.068, 0.120], p_Holm <0.001 |
| R8 static | A2→A3 implicit category | 0.000 → 0.048, far below majority baseline 0.755 |
| R8 contextual | A3→A4 | 0 change |
| DW | A4→A5 polarity | 0 change |
| SC | A5→A6 polarity | no significant change |

### Reach of the unproven mechanisms (`experiments/diagnose_mechanism_reach.py`, all 3,000 dev sentences, 5,357 pairs, no gold)
- SC changes 44 labels (0.8%); only a handful fall inside the 400 labelled polarity items, so the
  current gold cannot test SC. Targeted annotation of those pairs is required to get evidence either way.
- DW changes 4 labels and 407 magnitudes only. Label-based metrics cannot show a magnitude effect.
- Contextual R8 differs from static on 7 of 209 pairs. R8 needs a redesign (H4.4).

### Status
H4.0–H4.3 complete on dev; 85 tests passing. H4.4–H4.6 not started. Commits held locally pending the
public-repository decision below.

### Open risk: repository is public
`github.com/fatih-collab/simplexid-experiment` is public, and code implementing all claimed mechanisms has
been pushed since 2026-09-23 (the code in `src/`, then `src/simplexid/`). This may count as prior public
disclosure for the patent. Needs an IP consultant's assessment; no pushes made since this was noticed.

### Decisions and follow-ups (2026-11-05, after H4.3)
- Repository visibility: the user decided to keep the repository public and resume pushing. The disclosure
  risk above remains recorded.
- SC targeted annotation created: `data/annotation/sc_targeted/` (44 dev pairs where SC and inheritance
  disagree, blind sheet, `experiments/run_sc_targeted.py`). Awaiting human annotation. Exact McNemar needs
  at least 29 of 44 discordant items favouring one side for p<0.05 (more if invalid pairs are removed).
- R8 evaluation note: all 250 DEV_IMPLICIT rows come from sentences with no explicit aspect (by template
  design), and only 4 contain an ambiguous DAM word, so gold_implicit cannot test contextual disambiguation.
- Leakage check: 225 TermA train and 82 TermA validation rows are text-identical to a HoASA row (dedupe check,
  labels not read). Any HoASA-derived resource must exclude rows identical to final-split rows.

## H4.4 — R8 Rebuilt from HoASA (2026-11-05)

Decisions: merged_v1 stays the lexicon; domain adaptation deferred (user, 2026-11-05). 468 dev 1A1O rows set to
REVIEWED on the user's statement that they were reviewed; no pair changed. Dev gold version `ee6000ff47fdad69`.
Dev gold_polarity (14 MEDIUM) and gold_implicit (195 MEDIUM/LOW) remain unverified, so numbers are provisional.

- DAM from HoASA reviews: 2,371 usable after removing rows text-identical to labelled TermA. 153 words;
  other lexicon words default to `general`. min_count 5, lift 1.2 chosen on DEV_IMPLICIT.
- Results (`results/20261105-184806_dev_h44-ablation`, commit 6eb86ab, clean tree): A3 implicit detection P 0.94 / R 0.81 (was recall 0.035);
  category accuracy 0.533 and macro-F1 0.569, vs majority baseline accuracy 0.755 and macro-F1 0.215.
  R8 beats the baseline on macro-F1 but not on accuracy; 'general' recall is only 0.44.
- Precision outside candidate sentences is weak: over all dev sentences R8 emits 3,815 pairs, 59.2% on tokens
  inside a TermA opinion span. gold_implicit does not measure this.
- Contextual disambiguation: changes 702 pairs (430 inside opinion spans); A3→A4 on gold_implicit −0.009, n.s.,
  on sentences that have little context. Blind sheet of 60 sampled cases: `data/annotation/ctx_targeted/`. Awaiting annotation.
- H4.4 status: step 1–3 done; R8 contextual and SC claims await human annotation.
