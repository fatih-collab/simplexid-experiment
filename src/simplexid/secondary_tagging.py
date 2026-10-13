"""
Deteksi semi-otomatis kandidat kalimat COORD / IMPLICIT_AMBIG / ASYM_MOD
(Bagian 3 rancangan eksperimen) -- mempercepat anotasi manual, BUKAN
pengganti anotasi manual (tetap perlu verifikasi 2 annotator + kappa).
"""

from dataclasses import dataclass
from typing import List
from .parser import ParsedSentence, parse_tokens
from .rules import extract_explicit_pairs, find_orphan_opinions
from .dam import lookup
from .scoring import NEGATION_WORDS, AMPLIFIER_WORDS, DOWNTONER_WORDS


@dataclass
class TagCandidate:
    sentence_id: str
    tags: List[str]
    notes: str


def detect_coord(sent: ParsedSentence) -> bool:
    """
    COORD: ada relasi 'conj' yang menghubungkan dua token ber-POS NOUN atau
    dua token ber-POS ADJ (kandidat R3/R4).
    """
    for t in sent.tokens:
        if t.dep == "conj":
            head = sent.tokens[t.head_idx]
            if t.pos == head.pos and t.pos in ("NOUN", "ADJ"):
                return True
    return False


def detect_implicit_ambig(sent: ParsedSentence, domain: str) -> bool:
    """
    IMPLICIT_AMBIG: ada orphan opinion word dengan >1 kandidat kategori
    aspek implisit di DAM_DICT untuk domain tsb.
    """
    pairs = extract_explicit_pairs(sent)
    orphans = find_orphan_opinions(sent, pairs)
    for o in orphans:
        candidates = lookup(o.lemma, domain)
        if len(candidates) > 1:
            return True
    return False


def detect_asym_mod(sent: ParsedSentence) -> bool:
    """
    ASYM_MOD: ada pasangan konjungtif (conj antar ADJ) di mana modifier
    (negasi/intensifier) HANYA terhubung ke salah satu dari dua konjungtor.
    """
    for t in sent.tokens:
        if t.dep == "conj":
            head = sent.tokens[t.head_idx]
            if t.pos == "ADJ" and head.pos == "ADJ":
                head_mods = {c.lemma.lower() for c in sent.tokens if c.head_idx == head.idx
                             and c.dep in ("advmod", "neg")}
                t_mods = {c.lemma.lower() for c in sent.tokens if c.head_idx == t.idx
                          and c.dep in ("advmod", "neg")}
                relevant = NEGATION_WORDS | AMPLIFIER_WORDS | DOWNTONER_WORDS
                if (head_mods & relevant) != (t_mods & relevant):
                    return True
    return False


def tag_sentence(sentence_id: str, tokens: List[str], domain: str) -> TagCandidate:
    sent = parse_tokens(tokens)
    tags = []
    if detect_coord(sent):
        tags.append("COORD")
    if detect_implicit_ambig(sent, domain):
        tags.append("IMPLICIT_AMBIG")
    if detect_asym_mod(sent):
        tags.append("ASYM_MOD")
    return TagCandidate(sentence_id=sentence_id, tags=tags,
                         notes="Kandidat otomatis -- perlu verifikasi manual")


if __name__ == "__main__":
    # Contoh pakai pada beberapa kalimat contoh dari draf paten
    examples = {
        "contoh-1": "Kecepatan dan stabilitas koneksi sangat baik".split(),
        "contoh-2": "Sinyalnya sangat lemah dan tidak stabil".split(),
    }
    for sid, toks in examples.items():
        result = tag_sentence(sid, toks, domain="hotel")
        print(sid, "->", result.tags)
