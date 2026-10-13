"""
Wrapper UDPipe untuk Bahasa Indonesia (ufal.udpipe, tanpa spaCy).

Menghasilkan struktur token sederhana (Token) dengan field yang dibutuhkan
rules.py dan scoring.py: teks, lemma, pos (Universal POS), head index, dan
label relasi dependensi -- serta fungsi jarak antar-token pada pohon
dependensi (dipakai formula berbobot-jarak di scoring.py).

Catatan desain:
- spacy-udpipe TIDAK dipakai lagi: di library tsb seluruh UDPipe berjalan
  sebagai tokenizer sehingga `nlp.pipeline` kosong, dan parse dari token
  yang sudah ditokenisasi (parse_tokens versi lama) tidak pernah menghasilkan
  relasi dependensi.
- Model dibaca dari path eksplisit (resources/models/udpipe/), bukan dari
  site-packages, agar versi model tercatat dan hasil bisa direproduksi.
  Override lewat environment variable SIMPLEXID_UDPIPE_MODEL.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "resources" / "models" / "udpipe" / "indonesian-gsd-ud-2.5-191206.udpipe"
)
MODEL_NAME = DEFAULT_MODEL_PATH.name


@dataclass
class Token:
    idx: int             # index token dlm kalimat (0-based)
    text: str
    lemma: str
    pos: str              # Universal POS: NOUN, ADJ, VERB, ADV, CCONJ, ...
    dep: str               # label relasi dependensi ke head: amod, nsubj, obj, conj, ...
    head_idx: int           # index token head (== idx jika root)
    feats: str = "_"        # fitur morfologis UD, mis. "Polarity=Neg"


@dataclass
class ParsedSentence:
    tokens: List[Token]

    def token_text(self):
        return [t.text for t in self.tokens]

    def to_conllu(self) -> str:
        """Serialisasi ke CoNLL-U (dipakai cache parse, data/cache/parses/)."""
        lines = []
        for t in self.tokens:
            head = 0 if t.head_idx == t.idx else t.head_idx + 1
            lines.append("\t".join([
                str(t.idx + 1), t.text, t.lemma, t.pos, "_", t.feats or "_",
                str(head), t.dep, "_", "_",
            ]))
        return "\n".join(lines) + "\n"


def from_conllu(block: str) -> ParsedSentence:
    """Baca SATU kalimat CoNLL-U menjadi ParsedSentence (multiword/empty node diabaikan)."""
    tokens: List[Token] = []
    for line in block.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        idx = int(cols[0]) - 1
        head = int(cols[6])
        tokens.append(Token(
            idx=idx,
            text=cols[1],
            lemma=cols[2],
            pos=cols[3],
            dep=cols[7],
            head_idx=idx if head == 0 else head - 1,
            feats=cols[5],
        ))
    return ParsedSentence(tokens=tokens)


class UDPipeParser:
    """Model UDPipe yang dimuat sekali; satu instance per path model."""

    def __init__(self, model_path: str | Path | None = None):
        from ufal.udpipe import Model, Pipeline

        path = Path(model_path or os.environ.get("SIMPLEXID_UDPIPE_MODEL") or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Model UDPipe tidak ditemukan: {path}\n"
                "Lihat resources/models/udpipe/SOURCE.md untuk cara mengunduh."
            )
        model = Model.load(str(path))
        if model is None:
            raise RuntimeError(
                f"Model UDPipe gagal dimuat: {path}\n"
                "File kemungkinan rusak (mis. halaman HTML hasil download gagal). "
                "Verifikasi checksum sesuai resources/models/udpipe/SOURCE.md."
            )
        self.model_path = path
        # Pipeline hanya menyimpan pointer C++ ke model: referensi Python WAJIB
        # dipertahankan, kalau tidak model di-GC dan process() crash (abort).
        self._model = model
        self._raw = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        # "horizontal": satu baris = satu kalimat, token dipisah spasi -> tokenisasi TermA
        # dipertahankan apa adanya sehingga span index gold tetap valid.
        self._pretok = Pipeline(model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    def _run(self, pipeline, text: str) -> str:
        from ufal.udpipe import ProcessingError

        error = ProcessingError()
        out = pipeline.process(text, error)
        if error.occurred():
            raise RuntimeError(f"UDPipe error: {error.message}")
        return out

    def parse(self, sentence: str) -> ParsedSentence:
        """Parse teks mentah. Jika UDPipe memecah >1 kalimat, semua token digabung
        dalam satu ParsedSentence; root kalimat berikutnya tetap root (head==idx)."""
        conllu = self._run(self._raw, sentence)
        tokens: List[Token] = []
        for block in conllu.strip().split("\n\n"):
            offset = len(tokens)
            for t in from_conllu(block).tokens:
                t.idx += offset
                t.head_idx += offset
                tokens.append(t)
        return ParsedSentence(tokens=tokens)

    def parse_tokens(self, tokens: List[str]) -> ParsedSentence:
        """Parse token yang SUDAH ditokenisasi (TermA). Jumlah & urutan token dijamin sama."""
        if any(not t or any(ch.isspace() for ch in t) for t in tokens):
            raise ValueError(f"Token kosong/berisi spasi tidak didukung: {tokens}")
        conllu = self._run(self._pretok, " ".join(tokens))
        parsed = from_conllu(conllu)
        if [t.text for t in parsed.tokens] != list(tokens):
            raise RuntimeError(
                f"Tokenisasi berubah saat parsing: {tokens} -> {parsed.token_text()}"
            )
        return parsed


_PARSER: Optional[UDPipeParser] = None


def get_parser() -> UDPipeParser:
    global _PARSER
    if _PARSER is None:
        _PARSER = UDPipeParser()
    return _PARSER


def parse(sentence: str) -> ParsedSentence:
    """Parse satu kalimat Bahasa Indonesia menjadi ParsedSentence."""
    return get_parser().parse(sentence)


def parse_tokens(tokens: List[str]) -> ParsedSentence:
    """
    Parse dari daftar token yang SUDAH ditokenisasi (dipakai saat menyelaraskan
    dengan token TermA, yang tokenisasinya sudah given dan tidak boleh diubah
    ulang oleh tokenizer UDPipe agar span index tetap valid).
    """
    return get_parser().parse_tokens(tokens)


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
