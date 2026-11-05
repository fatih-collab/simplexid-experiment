"""
Implementasi R0-R7 (ekstraksi eksplisit) dan R3/R4 (ekspansi konjungtif).

Label dependensi mengikuti keluaran model UDPipe id-gsd 2.5 (UD v2). Perubahan
rule dikontrol RuleOptions agar tiap perubahan bisa diukur sendiri-sendiri
(experiments/run_rule_log.py); BASE_RULES adalah perilaku awal, DEFAULT_RULES
adalah set yang dipakai sistem.
"""

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Optional, Tuple

from .cues import ALL_CUES
from .parser import ParsedSentence, Token
from .scoring import token_score

UDPIPE_LABEL_MAP = {
    "dobj": "obj",          # Stanford -> UD v2
    "pobj": "obl",          # objek preposisi -> oblique nominal di UD v2
}

OPINION_POS = ("ADJ", "VERB")
# POS yang sering diberikan parser pada kata opini ("kotor" NOUN, "oke" NUM).
MISTAGGED_OPINION_POS = ("NOUN", "NUM", "PROPN", "ADV", "X")
# Relasi tempat kata opini salah-POS itu menempel ke aspek.
MISTAGGED_OPINION_RELATIONS = ("compound", "nummod", "flat")
# POS head aspek selain NOUN ("AC" = X/PROPN).
EXTRA_ASPECT_POS = ("PROPN", "X")
ASPECT_SPAN_RELATIONS = ("compound", "flat", "compound:plur")


@dataclass(frozen=True)
class RuleOptions:
    # H4.2 recall
    passive_subject: bool = False      # R1 menerima nsubj:pass
    acl_relative: bool = False         # R7 memakai acl (model tidak menghasilkan relcl)
    extended_aspect_pos: bool = False  # aspek boleh PROPN/X
    lexicon_opinion: bool = False      # kata di leksikon ber-POS NOUN/NUM/.. boleh jadi opini
    extended_r4: bool = False          # R4 menerima konjungtor VERB dan kata di leksikon
    phrase_spans: bool = False         # span mencakup compound (aspek) dan modifier (opini)
    # H4.3 precision
    distribution_guard: bool = False   # R3/R4 tidak memakai konjungtor yang punya pasangan sendiri
    subjective_verbs: bool = False     # VERB opini R1/R2/R5/R7 harus di leksikon atau dinegasi
    aspect_vocabulary: bool = False    # head aspek harus ada di kosakata aspek


BASE_RULES = RuleOptions()
# Keputusan berdasarkan leave-one-out di dev (reports/h4/rule_changes.md).
DEFAULT_RULES = RuleOptions(
    passive_subject=True, acl_relative=True, extended_aspect_pos=True, lexicon_opinion=True,
    extended_r4=True, phrase_spans=True, distribution_guard=True, subjective_verbs=True,
    aspect_vocabulary=True,
)


@dataclass
class Pair:
    aspect_idx: int
    opinion_idx: int
    rule: str                 # "R0".."R8" -- untuk traceability/explainability
    aspect_text: str = ""
    opinion_text: str = ""
    source_opinion_idx: Optional[int] = None  # opini sumber untuk pasangan hasil R4
    aspect_span: Optional[Tuple[int, int]] = None
    opinion_span: Optional[Tuple[int, int]] = None


def _dep(tok: Token) -> str:
    """Normalisasi label dependensi via UDPIPE_LABEL_MAP."""
    return UDPIPE_LABEL_MAP.get(tok.dep, tok.dep)


def _children(sent: ParsedSentence) -> Dict[int, List[Token]]:
    children: Dict[int, List[Token]] = {t.idx: [] for t in sent.tokens}
    for t in sent.tokens:
        if t.head_idx != t.idx:
            children[t.head_idx].append(t)
    return children


def _is_aspect(tok: Token, options: RuleOptions, vocabulary: Optional[AbstractSet[str]] = None) -> bool:
    if tok.pos != "NOUN" and not (options.extended_aspect_pos and tok.pos in EXTRA_ASPECT_POS):
        return False
    if options.aspect_vocabulary and vocabulary is not None:
        return tok.text.lower() in vocabulary or tok.lemma.lower() in vocabulary
    return True


def _in_lexicon(tok: Token, lexicon: Optional[dict]) -> bool:
    return bool(lexicon) and token_score(tok, lexicon) != 0.0


def _is_negated(tok: Token, children: Dict[int, List[Token]]) -> bool:
    return any(_dep(c) == "advmod" and (c.text.lower() in ALL_CUES or c.lemma.lower() in ALL_CUES)
               for c in children[tok.idx])


def _is_mistagged_opinion(tok: Token, options: RuleOptions, lexicon: Optional[dict]) -> bool:
    return options.lexicon_opinion and tok.pos in MISTAGGED_OPINION_POS and _in_lexicon(tok, lexicon)


def phrase_span(sent: ParsedSentence, idx: int, relations, cue_only: bool = False) -> Tuple[int, int]:
    """Perluas idx ke token bersebelahan yang merupakan anak/induknya lewat `relations`."""
    def linked(a: Token, b: Token) -> bool:
        child, head = (a, b) if a.head_idx == b.idx else (b, a) if b.head_idx == a.idx else (None, None)
        if child is None or _dep(child) not in relations:
            return False
        return not cue_only or child.text.lower() in ALL_CUES or child.lemma.lower() in ALL_CUES

    start = end = idx
    while start > 0 and linked(sent.tokens[start - 1], sent.tokens[start]):
        start -= 1
    while end + 1 < len(sent.tokens) and linked(sent.tokens[end], sent.tokens[end + 1]):
        end += 1
    return start, end


def _finalize(sent: ParsedSentence, pair: Pair, options: RuleOptions) -> None:
    if options.phrase_spans:
        pair.aspect_span = phrase_span(sent, pair.aspect_idx, ASPECT_SPAN_RELATIONS)
        pair.opinion_span = phrase_span(sent, pair.opinion_idx, ("advmod",), cue_only=True)
    else:
        pair.aspect_span = (pair.aspect_idx, pair.aspect_idx)
        pair.opinion_span = (pair.opinion_idx, pair.opinion_idx)
    a, o = pair.aspect_span, pair.opinion_span
    pair.aspect_text = " ".join(t.text for t in sent.tokens[a[0]:a[1] + 1])
    pair.opinion_text = " ".join(t.text for t in sent.tokens[o[0]:o[1] + 1])


def extract_explicit_pairs(sent: ParsedSentence, options: RuleOptions = BASE_RULES,
                           lexicon: Optional[dict] = None,
                           aspect_vocabulary: Optional[AbstractSet[str]] = None) -> List[Pair]:
    """
    Kelompok Aturan Pertama: R0 (amod), R1 (nsubj), R2 (obj), R5 (obl), R6 (compound,
    lewat span aspek), R7 (klausa relatif). lexicon hanya dipakai opsi yang membutuhkannya.
    """
    children = _children(sent)
    subject_labels = ("nsubj", "nsubj:pass") if options.passive_subject else ("nsubj",)
    relative_labels = ("relcl", "acl") if options.acl_relative else ("relcl",)

    def aspect(tok: Token) -> bool:
        return _is_aspect(tok, options, aspect_vocabulary)

    def subjective(tok: Token) -> bool:
        if not options.subjective_verbs or tok.pos != "VERB":
            return True
        return _in_lexicon(tok, lexicon) or _is_negated(tok, children)

    pairs = []
    for t in sent.tokens:
        head = sent.tokens[t.head_idx] if t.head_idx != t.idx else None
        dep = _dep(t)

        if _is_mistagged_opinion(t, options, lexicon):
            if head and dep in MISTAGGED_OPINION_RELATIONS and aspect(head):
                pairs.append(Pair(head.idx, t.idx, "R0"))
            if dep == "amod" and head and aspect(head):
                pairs.append(Pair(head.idx, t.idx, "R0"))
            continue
        if t.pos not in OPINION_POS:
            continue

        # R0: t memodifikasi head langsung
        if dep == "amod" and head and aspect(head):
            pairs.append(Pair(head.idx, t.idx, "R0"))

        if subjective(t):
            # R1: subjek t
            subj = [c for c in children[t.idx] if _dep(c) in subject_labels]
            if subj and aspect(subj[0]):
                pairs.append(Pair(subj[0].idx, t.idx, "R1"))
            # R2: objek verba
            if t.pos == "VERB":
                pairs += [Pair(o.idx, t.idx, "R2") for o in children[t.idx] if _dep(o) == "obj" and aspect(o)]
            # R5: oblique ke nomina
            if dep == "obl" and head and aspect(head):
                pairs.append(Pair(head.idx, t.idx, "R5"))
            # R7: klausa relatif yang menerangkan nomina
            if dep in relative_labels and head and aspect(head):
                pairs.append(Pair(head.idx, t.idx, "R7"))

    pairs = _dedupe(pairs)
    for p in pairs:
        _finalize(sent, p, options)
    return pairs


def _dedupe(pairs: List[Pair]) -> List[Pair]:
    seen = set()
    out = []
    for p in pairs:
        key = (p.aspect_idx, p.opinion_idx)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def expand_r3(sent: ParsedSentence, pairs: List[Pair], options: RuleOptions = BASE_RULES,
              aspect_vocabulary: Optional[AbstractSet[str]] = None) -> List[Pair]:
    """
    R3: ekspansi konjungtif sisi ASPEK. Dari (a1, o), konjungtor a2 dari a1 -> (a2, o).
    distribution_guard: a2 yang sudah punya opininya sendiri dilewati; pada
    "kamar bersih , ac dingin" konjungsinya antar-klausa, "bersih" tidak menilai "ac".
    """
    children = _children(sent)
    own_opinion = {p.aspect_idx for p in pairs}
    new_pairs = []
    for p in pairs:
        for a2 in children[p.aspect_idx]:
            if _dep(a2) != "conj" or not _is_aspect(a2, options, aspect_vocabulary):
                continue
            if options.distribution_guard and a2.idx in own_opinion:
                continue
            pair = Pair(a2.idx, p.opinion_idx, "R3")
            _finalize(sent, pair, options)
            new_pairs.append(pair)
    return new_pairs


def _is_r4_conjunct(tok: Token, options: RuleOptions, lexicon: Optional[dict]) -> bool:
    if tok.pos == "ADJ":
        return True
    return options.extended_r4 and (tok.pos == "VERB" or _is_mistagged_opinion(tok, options, lexicon)
                                    or _in_lexicon(tok, lexicon))


def expand_r4(sent: ParsedSentence, pairs: List[Pair], options: RuleOptions = BASE_RULES,
              lexicon: Optional[dict] = None) -> List[Pair]:
    """
    R4: ekspansi konjungtif sisi OPINI. Dari (a, o1), konjungtor o2 dari o1 -> (a, o2).
    distribution_guard: o2 yang punya subjek sendiri menilai subjek itu, bukan a.
    """
    children = _children(sent)
    new_pairs = []
    for p in pairs:
        for o2 in children[p.opinion_idx]:
            if _dep(o2) != "conj" or not _is_r4_conjunct(o2, options, lexicon):
                continue
            if options.distribution_guard and any(_dep(c) in ("nsubj", "nsubj:pass") for c in children[o2.idx]):
                continue
            pair = Pair(p.aspect_idx, o2.idx, "R4", source_opinion_idx=p.opinion_idx)
            _finalize(sent, pair, options)
            new_pairs.append(pair)
    return new_pairs


def conj_source_opinion(sent: ParsedSentence, opinion_idx: int, options: RuleOptions = BASE_RULES,
                        lexicon: Optional[dict] = None) -> Optional[int]:
    """Opini sumber bila token ini akan dihasilkan R4 (dipakai mode oracle-pair tanpa SC)."""
    t = sent.tokens[opinion_idx]
    if _dep(t) != "conj" or t.head_idx == t.idx or not _is_r4_conjunct(t, options, lexicon):
        return None
    head = sent.tokens[t.head_idx]
    if head.pos in OPINION_POS or _is_mistagged_opinion(head, options, lexicon):
        return t.head_idx
    return None


def find_orphan_opinions(sent: ParsedSentence, pairs: List[Pair]) -> List[Token]:
    """Token ber-POS ADJ/VERB yang belum jadi opini di `pairs` manapun (kandidat R8)."""
    paired_opinion_idx = {p.opinion_idx for p in pairs}
    return [t for t in sent.tokens
            if t.pos in OPINION_POS and t.idx not in paired_opinion_idx]
