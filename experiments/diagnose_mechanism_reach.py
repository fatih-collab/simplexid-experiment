"""Berapa pasangan di seluruh dev yang labelnya bisa diubah SC / DW / R8 kontekstual (tanpa memakai gold)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from simplexid.aspect_vocab import crossfit_vocabularies, fold_of
from simplexid.config import VARIANTS
from simplexid.dam_builder import load_dam
from simplexid.evaluation.metrics import polarity_of
from simplexid.lexicon import load_lexicon
from simplexid.parse_cache import load_parse_cache
from simplexid.pipeline import run_pipeline
t = pd.read_csv("data/processed/terma_processed.csv"); pc = load_parse_cache("data/cache/parses"); lex = load_lexicon(); dam = load_dam()
folds = crossfit_vocabularies(t, pc); ids = t[t.source_split == "train"].sentence_id.tolist()
def run(v, s): return {(r.pair.aspect_idx, r.pair.opinion_idx): r for r in run_pipeline(None, VARIANTS[v], "hotel", lex, parsed_sentence=pc[s], aspect_vocabulary=folds[fold_of(s)], dam=dam)}
tot = r4 = 0; sc, dw, dwmag = [], [], 0; ctxdiff = r8 = 0; ctx = []; r8_in_span = 0
spans = {r.sentence_id: json.loads(r.opinions) for r in t[t.source_split == "train"].itertuples()}
for s in ids:
    a4, a5, a6, a3 = run("A4", s), run("A5", s), run("A6", s), run("A3", s)
    for k, r in a6.items():
        if k[0] < 0: continue
        tot += 1; r4 += r.pair.rule == "R4"
        if polarity_of(a5[k].factors.score_final) != polarity_of(r.factors.score_final):
            sc.append((s, r.pair.aspect_text, r.pair.opinion_text, polarity_of(a5[k].factors.score_final), polarity_of(r.factors.score_final)))
    for k, r in a5.items():
        if k[0] < 0: continue
        if polarity_of(a4[k].factors.score_final) != polarity_of(r.factors.score_final): dw.append((s, r.pair.aspect_text, r.pair.opinion_text, polarity_of(a4[k].factors.score_final), polarity_of(r.factors.score_final)))
        elif a4[k].factors.score_final != r.factors.score_final: dwmag += 1
    for k, r in a4.items():
        if k[0] == -1:
            r8_in_span += any(o["start"] <= k[1] <= o["end"] for o in spans[s])
            r8 += 1; ctxdiff += a3[k].pair.aspect_text != r.pair.aspect_text
            if a3[k].pair.aspect_text != r.pair.aspect_text: ctx.append((s, r.pair.opinion_text, a3[k].pair.aspect_text, r.pair.aspect_text))
print(f"dev sentences={len(ids)} explicit pairs={tot} R4 pairs={r4}")
print(f"SC label changes: {len(sc)}"); [print(" ", x) for x in sc[:15]]
print(f"DW label changes: {len(dw)} | magnitude-only changes: {dwmag}"); [print(" ", x) for x in dw[:10]]
print(f"R8 pairs={r8} | opinion token inside a TermA opinion span: {r8_in_span} ({r8_in_span/max(r8,1):.1%}) | contextual != static: {ctxdiff}"); [print(" ", x) for x in ctx[:12]]
