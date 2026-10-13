"""
Wrapper UDPipe untuk Bahasa Indonesia via spacy-udpipe.

Menghasilkan struktur token sederhana (Token) dengan field yang dibutuhkan
rules.py dan scoring.py: teks, lemma, pos (Universal POS), head index, dan
label relasi dependensi -- serta fungsi jarak antar-token pada pohon
dependensi (dipakai formula berbobot-jarak di scoring.py).
"""

from dataclasses import dataclass
from typing import List, Optional
import functools


@dataclass
class Token:
    idx: int             # index token dlm kalimat (0-based)
    text: str
    lemma: str
    pos: str              # Universal POS: NOUN, ADJ, VERB, ADV, CCONJ, ...
    dep: str               # label relasi dependensi ke head: amod, nsubj, dobj, conj, ...
    head_idx: int           # index token head (== idx jika root)


@dataclass
class ParsedSentence:
    tokens: List[Token]

    def token_text(self):
        return [t.text for t in self.tokens]


_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy_udpipe
        try:
            _NLP = spacy_udpipe.load("id")
        except Exception:
            spacy_udpipe.download("id")
            _NLP = spacy_udpipe.load("id")
    return _NLP


def parse(sentence: str) -> ParsedSentence:
    """Parse satu kalimat Bahasa Indonesia menjadi ParsedSentence."""
    nlp = _get_nlp()
    doc = nlp(sentence)
    tokens = []
    for i, tok in enumerate(doc):
        tokens.append(Token(
            idx=i,
            text=tok.text,
            lemma=tok.lemma_,
            pos=tok.pos_,
            dep=tok.dep_,
            head_idx=tok.head.i,
        ))
    return ParsedSentence(tokens=tokens)


def parse_tokens(tokens: List[str]) -> ParsedSentence:
    """
    Parse dari daftar token yang SUDAH ditokenisasi (dipakai saat menyelaraskan
    dengan token TermA, yang tokenisasinya sudah given dan tidak boleh diubah
    ulang oleh tokenizer UDPipe agar span index tetap valid).
    """
    nlp = _get_nlp()
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=tokens)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    parsed = []
    for i, tok in enumerate(doc):
        parsed.append(Token(
            idx=i, text=tok.text, lemma=tok.lemma_, pos=tok.pos_,
            dep=tok.dep_, head_idx=tok.head.i,
        ))
    return ParsedSentence(tokens=parsed)


def build_adjacency(sent: ParsedSentence) -> dict:
    """Bangun adjacency list tak berarah dari pohon dependensi (untuk BFS jarak)."""
    adj = {t.idx: set() for t in sent.tokens}
    for t in sent.tokens:
        if t.head_idx != t.idx:
            adj[t.idx].add(t.head_idx)
            adj[t.head_idx].add(t.idx)
    return adj


def dependency_distance(sent: ParsedSentence, i: int, j: int) -> Optional[int]:
    """
    Jarak (jumlah edge) antara token i dan j pada pohon dependensi via BFS.
    Mengembalikan None jika tidak terhubung (seharusnya tidak terjadi pada
    satu pohon dependensi kalimat yang valid, tapi dijaga untuk keamanan).

    Dipakai oleh scoring.py untuk faktor Neg(o, d_neg) dan Int(o, d_int)
    pada varian distance_weighted_formula=True.
    """
    if i == j:
        return 0
    adj = build_adjacency(sent)
    from collections import deque
    visited = {i}
    q = deque([(i, 0)])
    while q:
        cur, d = q.popleft()
        for nxt in adj[cur]:
            if nxt == j:
                return d + 1
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, d + 1))
    return None


def find_dependents(sent: ParsedSentence, head_idx: int, dep_label: str) -> List[Token]:
    """Cari semua token yang punya head=head_idx dan dep==dep_label."""
    return [t for t in sent.tokens if t.head_idx == head_idx and t.dep == dep_label
            and t.idx != head_idx]


def find_heads_with_dep(sent: ParsedSentence, dep_label: str) -> List[Token]:
    """Cari semua token yang relasi dependensinya (ke head-nya) == dep_label."""
    return [t for t in sent.tokens if t.dep == dep_label]
