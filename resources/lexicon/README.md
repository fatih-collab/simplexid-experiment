# Sentiment lexicons

| File | Status | Source |
|---|---|---|
| `seed_lexicon_v0.tsv` | **PLACEHOLDER — not valid for reported polarity results** | 17 words copied from the legacy `run_experiment.py` placeholder |

Format: tab-separated, header `word<TAB>score`, score in [-1, 1], lowercase lemma.

H4 replaces the placeholder with a published Indonesian lexicon (e.g. InSet or SentiStrengthID)
after its license is checked. Record the source, version, license and SHA-256 here.
Every run manifest stores the hash of `resources/`, so results always name the lexicon they used.
