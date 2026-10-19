"""
Cache hasil parse UDPipe dalam format CoNLL-U (H3.2).

Setiap kalimat TermA di-parse SEKALI dengan model yang tercatat, lalu semua
eksperimen membaca cache ini. manifest.json menyimpan hash model dan hash
input sehingga cache otomatis ditolak bila model atau data berubah.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

from .parser import ParsedSentence, UDPipeParser, from_conllu

CACHE_FILE = "terma.conllu"
MANIFEST_FILE = "manifest.json"


class ParseCacheError(RuntimeError):
    """Cache tidak ada, rusak, atau tidak cocok dengan model/data saat ini."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(terma_df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    return [(row.sentence_id, json.loads(row.tokens)) for row in terma_df.itertuples(index=False)]


def input_hash(rows: Iterable[Tuple[str, List[str]]]) -> str:
    """Hash urutan (sentence_id, tokens) -- berubah jika tokenisasi/ID berubah."""
    h = hashlib.sha256()
    for sentence_id, tokens in rows:
        h.update(json.dumps([sentence_id, tokens], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_parse_cache(
    terma_df: pd.DataFrame,
    cache_dir: str | Path,
    parser: UDPipeParser,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Parse semua kalimat terma_df lalu tulis terma.conllu + manifest.json."""
    from ufal.udpipe import __version__ as udpipe_version  # type: ignore[attr-defined]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(terma_df)

    blocks = []
    for i, (sentence_id, tokens) in enumerate(rows, start=1):
        parsed = parser.parse_tokens(tokens)
        blocks.append(f"# sent_id = {sentence_id}\n# text = {' '.join(tokens)}\n{parsed.to_conllu()}")
        if progress and (i % 500 == 0 or i == len(rows)):
            progress(i, len(rows))

    tmp = cache_dir / (CACHE_FILE + ".tmp")
    tmp.write_text("\n".join(blocks), encoding="utf-8")
    tmp.replace(cache_dir / CACHE_FILE)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sentences": len(rows),
        "splits": {k: int(v) for k, v in terma_df["source_split"].value_counts().items()},
        "input_sha256": input_hash(rows),
        "model_file": parser.model_path.name,
        "model_sha256": sha256_file(parser.model_path),
        "ufal_udpipe_version": udpipe_version,
        "input_format": "horizontal (pretokenized TermA, one row = one tree)",
        "cache_sha256": sha256_file(cache_dir / CACHE_FILE),
    }
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_parse_cache(
    cache_dir: str | Path,
    terma_df: pd.DataFrame | None = None,
    model_path: str | Path | None = None,
) -> Dict[str, ParsedSentence]:
    """
    Baca cache menjadi {sentence_id: ParsedSentence}.

    terma_df / model_path: jika diberikan, cache diverifikasi terhadap data
    dan model saat ini; ketidakcocokan menimbulkan ParseCacheError.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / CACHE_FILE
    manifest_path = cache_dir / MANIFEST_FILE
    if not cache_path.exists() or not manifest_path.exists():
        raise ParseCacheError(
            f"Cache parse belum ada di {cache_dir}. Jalankan: python experiments/run_parse.py cache"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(cache_path) != manifest["cache_sha256"]:
        raise ParseCacheError("terma.conllu berubah sejak dibuat (hash tidak cocok).")
    if model_path is not None and sha256_file(model_path) != manifest["model_sha256"]:
        raise ParseCacheError("Model UDPipe berbeda dengan model yang dipakai membuat cache.")
    if terma_df is not None and input_hash(_rows(terma_df)) != manifest["input_sha256"]:
        raise ParseCacheError("terma_processed.csv berbeda dengan input saat cache dibuat.")

    parses: Dict[str, ParsedSentence] = {}
    for block in cache_path.read_text(encoding="utf-8").strip().split("\n\n"):
        sentence_id = next(
            line.split("=", 1)[1].strip() for line in block.splitlines() if line.startswith("# sent_id")
        )
        parses[sentence_id] = from_conllu(block)

    if len(parses) != manifest["n_sentences"]:
        raise ParseCacheError(
            f"Jumlah kalimat cache {len(parses)} != manifest {manifest['n_sentences']}"
        )
    return parses
