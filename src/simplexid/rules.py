"""
Implementasi R0-R7 (ekstraksi eksplisit) dan R3/R4 (ekspansi konjungtif).

CATATAN TAGSET: skema Universal Dependencies standar memakai label "obj"
(bukan "dobj" ala Stanford Dependencies) dan "nmod"/"case" untuk frasa
preposisional. UDPIPE_LABEL_MAP di bawah menjembatani penamaan pada draf
paten (R1=nsubj, R2=dobj, R5=prep+pobj) dengan label UD aktual yang keluar
dari parser -- SESUAIKAN jika model UDPipe Bahasa Indonesia yang dipakai
menghasilkan skema label yang sedikit berbeda (periksa dengan mencetak
`token.dep` pada beberapa kalimat contoh).
"""

from dataclasses import dataclass
from typing import List, Optional
from .parser import ParsedSentence, Token

UDPIPE_LABEL_MAP = {
    "dobj": "obj",          # Stanford -> UD v2
    "pobj": "obl",          # objek preposisi -> oblique nominal di UD v2
}


@dataclass
class Pair:
    aspect_idx: int
    opinion_idx: int
    rule: str                 # "R0".."R8" -- untuk traceability/explainability
    aspect_text: str = ""
    opinion_text: str = ""


def _dep(tok: Token) -> str:
    """Normalisasi label dependensi via UDPIPE_LABEL_MAP."""
    return UDPIPE_LABEL_MAP.get(tok.dep, tok.dep)


def extract_explicit_pairs(sent: ParsedSentence) -> List[Pair]:
    """
    Kelompok Aturan Pertama: R0 (amod), R1 (nsubj), R2 (obj/dobj),
    R5 (prep+pobj/obl), R6 (compound), R7 (relcl).

    Iterasi mengikuti Algoritma 1 di draf paten: untuk tiap token t dengan
    pos(t) in {ADJ, VERB}, cek relasi dependensinya untuk menentukan aturan
    yang cocok.
    """
    pairs = []
    for t in sent.tokens:
        if t.pos not in ("ADJ", "VERB"):
            continue
        dep = _dep(t)
        head = sent.tokens[t.head_idx] if t.head_idx != t.idx else None

        # R0: amod -- t (ADJ) memodifikasi head (NOUN) langsung
        if dep == "amod" and head and head.pos == "NOUN":
            pairs.append(Pair(head.idx, t.idx, "R0"))

        # R1: nsubj -- t adalah head predikat, subjek NOUN terhubung via nsubj
        subj = [c for c in sent.tokens if c.head_idx == t.idx and _dep(c) == "nsubj"]
        if subj and subj[0].pos == "NOUN":
            pairs.append(Pair(subj[0].idx, t.idx, "R1"))

        # R2: obj -- verba dgn objek NOUN
        if t.pos == "VERB":
            objs = [c for c in sent.tokens if c.head_idx == t.idx and _dep(c) == "obj"]
            for o in objs:
                if o.pos == "NOUN":
                    pairs.append(Pair(o.idx, t.idx, "R2"))

        # R5: prep+pobj/obl -- opini terhubung ke nomina via frasa preposisional
        if dep == "obl" and head and head.pos == "NOUN":
            pairs.append(Pair(head.idx, t.idx, "R5"))

        # R7: relcl -- opini dalam klausa relatif yg menerangkan nomina
        if dep == "relcl" and head and head.pos == "NOUN":
            pairs.append(Pair(head.idx, t.idx, "R7"))

    # R6: compound noun -- gabungkan dua NOUN beruntun dgn relasi compound
    _apply_compound(sent, pairs)

    for p in pairs:
        p.aspect_text = sent.tokens[p.aspect_idx].text
        p.opinion_text = sent.tokens[p.opinion_idx].text
    return _dedupe(pairs)


def _apply_compound(sent: ParsedSentence, pairs: List[Pair]):
    """
    R6: jika aspek pada suatu pair adalah head dari relasi 'compound',
    gabungkan dengan modifier-nya menjadi frasa nomina majemuk. Ini mengubah
    aspect_text pasangan yang sudah ada (bukan menambah pair baru).
    """
    compound_map = {}  # head_idx -> modifier token
    for t in sent.tokens:
        if _dep(t) == "compound":
            compound_map[t.head_idx] = t
    for p in pairs:
        if p.aspect_idx in compound_map:
            mod = compound_map[p.aspect_idx]
            p.aspect_text = f"{mod.text} {sent.tokens[p.aspect_idx].text}"


def _dedupe(pairs: List[Pair]) -> List[Pair]:
    seen = set()
    out = []
    for p in pairs:
        key = (p.aspect_idx, p.opinion_idx)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def expand_r3(sent: ParsedSentence, pairs: List[Pair]) -> List[Pair]:
    """
    R3: Ekspansi konjungtif sisi ASPEK.
    Untuk pasangan (o, a1) yang sudah ada, cari a2 dengan dep(a1,a2)=conj
    dan pos(a2)=NOUN -> tambahkan (o, a2).
    """
    new_pairs = []
    for p in pairs:
        a1 = sent.tokens[p.aspect_idx]
        conj_targets = [c for c in sent.tokens
                         if c.head_idx == a1.idx and _dep(c) == "conj" and c.pos == "NOUN"]
        for a2 in conj_targets:
            new_pairs.append(Pair(a2.idx, p.opinion_idx, "R3",
                                   aspect_text=a2.text, opinion_text=p.opinion_text))
    return new_pairs


def expand_r4(sent: ParsedSentence, pairs: List[Pair]) -> List[Pair]:
    """
    R4: Ekspansi konjungtif sisi OPINI.
    Untuk pasangan (o1, a) yang sudah ada, cari o2 dengan dep(o1,o2)=conj
    dan pos(o2)=ADJ -> tambahkan (o2, a).
    """
    new_pairs = []
    for p in pairs:
        o1 = sent.tokens[p.opinion_idx]
        conj_targets = [c for c in sent.tokens
                         if c.head_idx == o1.idx and _dep(c) == "conj" and c.pos == "ADJ"]
        for o2 in conj_targets:
            new_pairs.append(Pair(p.aspect_idx, o2.idx, "R4",
                                   aspect_text=p.aspect_text, opinion_text=o2.text))
    return new_pairs


def find_orphan_opinions(sent: ParsedSentence, pairs: List[Pair]) -> List[Token]:
    """Token ber-POS ADJ/VERB yang belum jadi opini di `pairs` manapun (kandidat R8)."""
    paired_opinion_idx = {p.opinion_idx for p in pairs}
    return [t for t in sent.tokens
            if t.pos in ("ADJ", "VERB") and t.idx not in paired_opinion_idx]
