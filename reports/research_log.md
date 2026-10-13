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
