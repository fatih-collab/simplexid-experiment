"""
Pencatatan run & penguncian final test (H3.3).

- Setiap run menulis results/<timestamp>_<split>_<label>/manifest.json berisi commit
  git, status dirty, hash kode, hash resource, hash file gold, versi gold, dan hash
  cache parse -> setiap angka bisa ditelusuri asalnya.
- Final test hanya bisa dijalankan bila configs/frozen.lock ada DAN hash kode/
  resource saat ini sama dengan yang dikunci. Setiap run final ditambahkan ke
  results/final_runs.log (append-only) sebagai bukti berapa kali final dipakai.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from ..parse_cache import sha256_file

GOLD_FILES = [
    "data/processed/terma_processed.csv",
    "data/gold/gold_review_log.csv",
    "data/gold/gold_pairs.csv",
    "data/gold/gold_polarity.csv",
    "data/gold/gold_implicit.csv",
    "data/gold/gold_dev_auto_1a1o.csv",
]
LOCK_PATH = "configs/frozen.lock"
FINAL_LOG = "results/final_runs.log"


class FrozenConfigError(RuntimeError):
    """Final test diminta tetapi config belum dibekukan atau sudah berubah."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_info(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        # results/ diabaikan: folder run yang sedang ditulis tidak boleh membuat repo tampak dirty.
        dirty = bool(run("status", "--porcelain", "--", ".", ":(exclude)results"))
        return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def tree_hash(root: Path, patterns) -> str:
    """Hash isi semua file yang cocok dengan pola (urutan path deterministik)."""
    h = hashlib.sha256()
    files = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    for path in files:
        if "__pycache__" in path.parts or path.suffix == ".udpipe":
            continue
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def fingerprint(root: Path) -> dict:
    root = Path(root)
    cache_manifest = root / "data" / "cache" / "parses" / "manifest.json"
    return {
        "code_sha256": tree_hash(root, ["src/simplexid/**/*.py", "experiments/*.py"]),
        "resources_sha256": tree_hash(root, ["resources/**/*"]),
        "gold_sha256": {f: sha256_file(root / f) for f in GOLD_FILES if (root / f).exists()},
        "parse_cache": json.loads(cache_manifest.read_text(encoding="utf-8"))["cache_sha256"]
        if cache_manifest.exists() else None,
    }


def new_run_dir(root: Path, split: str, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) / "results" / f"{stamp}_{split}_{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(run_dir: Path, root: Path, extra: Dict) -> dict:
    manifest = {"created_at": _now(), "git": git_info(root), **fingerprint(root), **extra}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def freeze(root: Path, note: str = "") -> dict:
    root = Path(root)
    lock = {"frozen_at": _now(), "git": git_info(root), "note": note, **fingerprint(root)}
    path = root / LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def check_frozen(root: Path) -> dict:
    root = Path(root)
    path = root / LOCK_PATH
    if not path.exists():
        raise FrozenConfigError(
            f"{LOCK_PATH} belum ada. Final test hanya boleh dijalankan setelah config dibekukan (H6): "
            "python experiments/run_eval.py freeze"
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    current = fingerprint(root)
    changed = [k for k in ("code_sha256", "resources_sha256", "gold_sha256", "parse_cache") if lock.get(k) != current[k]]
    if changed:
        raise FrozenConfigError(f"Berubah sejak dibekukan: {changed}. Bekukan ulang secara sadar dan catat alasannya.")
    return lock


def log_final_run(root: Path, run_dir: Path, variants) -> None:
    path = Path(root) / FINAL_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), "run_dir": run_dir.name, "variants": list(variants),
                            "git": git_info(Path(root))}) + "\n")
