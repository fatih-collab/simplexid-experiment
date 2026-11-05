# Sentiment lexicons

| Name (`--lexicon`) | Status | Source |
|---|---|---|
| `seed_lexicon_v0.tsv` | **PLACEHOLDER: not valid for reported polarity results** | 17 words copied from the legacy `run_experiment.py` |
| `ss_v1` | Research use | SentiStrengthID, `sentiwords_id.txt` (scale −5..+5, divided by 5) |
| `ow_v1` | Research use | ID-OpinionWords (binary, mapped to ±0.6) |
| `merged_v1` (default) | Research use | `ss_v1`, then `ow_v1` for words it does not cover |

## Third-party files (not in git)

Place in `resources/lexicon/third_party/`. They carry **no license file**, so they are not
redistributed from this repository. Cite Wahid & Azhari (2016) and, for ID-OpinionWords,
Liu, Hu & Cheng (2005), as the upstream READMEs request. Written permission from the authors is
needed before any commercial use; SentiStrengthID also derives from SentiStrength (Thelwall).

| Local file | Upstream | Commit | SHA-256 |
|---|---|---|---|
| `sentiwords_id.txt` | `masdevid/sentistrength_id/sentiwords_id.txt` | `00132290` | `d5e41780…4055` |
| `opinionwords_positive.txt` | `masdevid/ID-OpinionWords/positive.txt` | `163a4500` | `6b3fde60…60d3` |
| `opinionwords_negative.txt` | `masdevid/ID-OpinionWords/negative.txt` | `163a4500` | `c1a3f45e…0a80` |

Full checksums are in `src/simplexid/lexicon.py` and verified on load. Downloaded 2026-11-05 from
`https://raw.githubusercontent.com/<owner>/<repo>/master/<file>`.

Every run manifest stores the hash of `resources/`, so results always name the lexicon they used.
