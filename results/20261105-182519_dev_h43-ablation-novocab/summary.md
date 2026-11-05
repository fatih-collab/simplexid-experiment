# Evaluation summary — `dev`

Run: `20261105-182519_dev_h43-ablation-novocab` · lexicon: `merged_v1`

## Pair extraction (micro)

| Variant | strict F1 | head P | head R | **head F1** | overlap F1 |
|---|---|---|---|---|---|
| A0 | 0.328 | 0.397 | 0.571 | **0.469** | 0.490 |
| A1 | 0.327 | 0.388 | 0.581 | **0.465** | 0.486 |
| A2 | 0.338 | 0.393 | 0.612 | **0.478** | 0.500 |
| A6 | 0.338 | 0.393 | 0.612 | **0.478** | 0.500 |

## Head F1 by stratum

| Variant | COMPLEX | EMPTY | SIMPLE |
|---|---|---|---|
| A0 | P=0.54 R=0.53 F1=0.53 | FP=520 (519 sent.) | P=0.48 R=0.66 F1=0.55 |
| A1 | P=0.52 R=0.54 F1=0.53 | FP=540 (519 sent.) | P=0.47 R=0.66 F1=0.55 |
| A2 | P=0.53 R=0.59 F1=0.56 | FP=553 (519 sent.) | P=0.47 R=0.66 F1=0.55 |
| A6 | P=0.53 R=0.59 F1=0.56 | FP=553 (519 sent.) | P=0.47 R=0.66 F1=0.55 |

## Head F1 on coordination subsets

| Variant | COORD_ASPECT | COORD_OPINION | ASYM_MOD |
|---|---|---|---|
| A0 | 0.349 (n=46) | 0.555 (n=108) | 0.598 (n=33) |
| A1 | 0.403 (n=46) | 0.550 (n=108) | 0.587 (n=33) |
| A2 | 0.407 (n=46) | 0.642 (n=108) | 0.673 (n=33) |
| A6 | 0.407 (n=46) | 0.642 (n=108) | 0.673 (n=33) |

## Polarity (gold_polarity subset) and implicit aspects

| Variant | oracle-pair acc | end-to-end acc | conditional acc (coverage) | implicit detect F1 | implicit cat acc | majority baseline acc |
|---|---|---|---|---|---|---|
| A0 | 0.750 | 0.480 | 0.842 (57%) | 0.000 | 0.000 | 0.755 |
| A1 | 0.750 | 0.485 | 0.844 (57%) | 0.000 | 0.000 | 0.755 |
| A2 | 0.748 | 0.527 | 0.847 (62%) | 0.000 | 0.000 | 0.755 |
| A6 | 0.750 | 0.527 | 0.847 (62%) | 0.067 | 0.035 | 0.755 |

## Key comparisons (B − A)

| Comparison | Target | Δ | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| A0_vs_A6 | pair_f1_head | +0.0098 | [0.003, 0.0168] | 0.0038 | 0.0114 |
| A0_vs_A6 | polarity_end_to_end | +0.0475 | - | 0.0000 | 0.0000 |
| A0_vs_A6 | polarity_oracle_pair | +0.0000 | - | 1.0000 | 1.0000 |
| A0_vs_A1 | pair_f1_head | -0.0034 | [-0.0073, 0.0005] | 0.0862 | 0.1724 |
| A0_vs_A1 | pair_f1_head_coord_aspect | +0.0548 | [0.0192, 0.0968] | 0.0016 | 0.0064 |
| A1_vs_A2 | pair_f1_head | +0.0132 | [0.0078, 0.0189] | 0.0000 | 0.0000 |
| A1_vs_A2 | pair_f1_head_coord_opinion | +0.0918 | [0.0679, 0.1185] | 0.0000 | 0.0000 |
