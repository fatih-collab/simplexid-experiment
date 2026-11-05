"""
Pemuat leksikon sentimen (word -> skor -1..+1).

Nama yang dikenal:
- seed_lexicon_v0.tsv : placeholder 17 kata, tidak layak dilaporkan.
- ss_v1      : SentiStrengthID (masdevid), skala -5..+5 dinormalkan.
- ow_v1      : ID-OpinionWords (masdevid), biner -> +-OPINIONWORDS_WEIGHT.
- merged_v1  : ss_v1, lalu ow_v1 untuk kata yang tidak tercakup.

File pihak ketiga TIDAK ada di git karena tidak berlisensi eksplisit; unduh
sesuai resources/lexicon/README.md. Checksum diverifikasi saat dimuat.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_DIR = PROJECT_ROOT / "resources" / "lexicon"
THIRD_PARTY_DIR = LEXICON_DIR / "third_party"

PLACEHOLDER_LEXICON = "seed_lexicon_v0.tsv"
DEFAULT_LEXICON = "merged_v1"

SENTISTRENGTH_SCALE = 5.0
# ID-OpinionWords hanya punya tanda; 0.6 = 3/5, titik tengah magnitudo skala SentiStrength.
OPINIONWORDS_WEIGHT = 0.6

THIRD_PARTY_SHA256 = {
    "sentiwords_id.txt": "d5e417800747acd3c6cd278bbfca161b051cfd96a9576562bfb65a22aaee4055",
    "opinionwords_positive.txt": "6b3fde6027759bff420ef55773841ddbe077135e130f108660bfef0d875560d3",
    "opinionwords_negative.txt": "c1a3f45e3f402cb3b576258ec147d5855cc1dc76ec7a337d6ab705fd575e0a80",
}


def _third_party_text(name: str, directory: Path) -> str:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"Leksikon pihak ketiga tidak ditemukan: {path}\nUnduh sesuai resources/lexicon/README.md."
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if name in THIRD_PARTY_SHA256 and digest != THIRD_PARTY_SHA256[name]:
        raise ValueError(f"Checksum {name} tidak cocok: {digest}")
    return data.decode("utf-8", errors="replace")


def parse_sentistrength(text: str, scale: float = SENTISTRENGTH_SCALE) -> Dict[str, float]:
    lexicon = {}
    for line in text.splitlines():
        match = re.match(r"^(.*?):(-?\d+)\s*$", line.strip())
        if match:
            lexicon[match.group(1).strip().lower()] = float(match.group(2)) / scale
    return lexicon


def parse_binary(positive: str, negative: str, weight: float = OPINIONWORDS_WEIGHT) -> Dict[str, float]:
    lexicon = {w.strip().lower(): weight for w in positive.splitlines() if w.strip()}
    for word in negative.splitlines():
        if word.strip():
            lexicon[word.strip().lower()] = -weight
    return lexicon


def merge_lexicons(primary: Dict[str, float], fallback: Dict[str, float]) -> Dict[str, float]:
    return {**fallback, **primary}


def load_lexicon(name: str = DEFAULT_LEXICON, third_party_dir: Path = THIRD_PARTY_DIR) -> Dict[str, float]:
    if name in ("ss_v1", "merged_v1"):
        sentistrength = parse_sentistrength(_third_party_text("sentiwords_id.txt", third_party_dir))
    if name in ("ow_v1", "merged_v1"):
        opinionwords = parse_binary(_third_party_text("opinionwords_positive.txt", third_party_dir),
                                    _third_party_text("opinionwords_negative.txt", third_party_dir))
    if name == "ss_v1":
        return sentistrength
    if name == "ow_v1":
        return opinionwords
    if name == "merged_v1":
        return merge_lexicons(sentistrength, opinionwords)

    path = Path(name) if Path(name).is_absolute() else LEXICON_DIR / name
    with path.open(encoding="utf-8") as f:
        lexicon = {row["word"].strip().lower(): float(row["score"]) for row in csv.DictReader(f, delimiter="\t")}
    for word, score in lexicon.items():
        if not -1.0 <= score <= 1.0:
            raise ValueError(f"Skor leksikon di luar [-1, 1]: {word}={score}")
    return lexicon
