"""Environment provenance sidecars for generated artifacts.

The payload files under ``artifacts/`` are the evidence: a dirty ``git
status`` on one of them means a value actually changed, and every doc claim
that cites an artifact is trusting that signal. Environment metadata (git
commit, python/numpy/BLAS versions, platform, wall-clock time) is not part of
that signal -- it is not even deterministic *given* a deterministic payload:
two honest runs on two different machines can produce byte-identical output
with entirely different environment strings, and re-running the same writer
on the same machine an hour later changes only the timestamp. Stamping any of
that into a payload dict would make every routine re-run dirty every payload
file forever, destroying the one signal this repo relies on to distinguish
"nothing changed" from "something changed." So provenance always goes into a
sibling file, ``<payload-stem>.provenance.json``, and the payload writers
never gain a new key.

See TECHNICAL_NOTE.md Sec. 5 for the measured cross-environment deltas this
was written to explain, and for why the existing (pre-2026-08-07) artifacts
carry a hand-authored "unknown_historical" sidecar instead of a collected
one.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "collect_environment",
    "write_provenance_sidecar",
    "write_unknown_provenance_sidecar",
]

_REPO_ROOT = Path(__file__).resolve().parent


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _blas_info() -> dict[str, Any] | None:
    # np.show_config(mode="dicts") is numpy >= 1.26. Older numpy raises
    # TypeError on the unexpected kwarg; report "unavailable" rather than
    # failing the writer over a metadata nicety.
    try:
        cfg = np.show_config(mode="dicts")
        if isinstance(cfg, dict):
            return cfg.get("Build Dependencies", {}).get("blas") or None
    except Exception:
        pass
    return None


def collect_environment() -> dict[str, Any]:
    """Snapshot of everything that can make a re-run's numbers differ
    without the writer's own logic changing."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blas": _blas_info(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _payload_hash(payload_path: Path) -> str | None:
    if not payload_path.exists():
        return None
    h = hashlib.blake2b(digest_size=32)
    h.update(payload_path.read_bytes())
    return h.hexdigest()


def write_provenance_sidecar(
    payload_path: str | Path,
    *,
    environment: dict[str, Any] | None = None,
    **extra: Any,
) -> Path:
    """Write ``<payload>.provenance.json`` next to a just-written artifact.

    Never touches the payload file's own bytes, so its git diff stays a pure
    content signal. ``payload_path`` may be a JSON file (most writers) or any
    other artifact (e.g. the vnnlib manifest); the sidecar name is derived
    from its stem regardless of extension.

    ``environment`` defaults to a fresh :func:`collect_environment` snapshot
    -- the normal path, used by every writer below. Pass an explicit dict
    only to backfill a sidecar for bytes this process did not itself produce
    (see :func:`write_unknown_provenance_sidecar`).

    ``extra`` lets a caller attach artifact-specific notes, e.g. which script
    or CLI invocation produced the payload.
    """
    payload_path = Path(payload_path)
    sidecar_path = payload_path.with_name(payload_path.stem + ".provenance.json")
    record: dict[str, Any] = {
        "payload_file": payload_path.name,
        "payload_blake2b": _payload_hash(payload_path),
        **(collect_environment() if environment is None else environment),
        **extra,
    }
    sidecar_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return sidecar_path


_UNKNOWN_ENVIRONMENT: dict[str, Any] = {
    "timestamp_utc": None,
    "git_commit": None,
    "git_dirty": None,
    "python_version": None,
    "numpy_version": None,
    "blas": None,
    "platform": None,
    "machine": None,
    "processor": None,
}


def write_unknown_provenance_sidecar(payload_path: str | Path, *, note: str) -> Path:
    """Backfill a sidecar for an artifact already committed before
    provenance tracking existed.

    The real environment that produced these bytes is not recoverable, and
    this function does not guess at it -- every environment field is
    ``null``. ``payload_blake2b`` is real (hashed from the file as currently
    committed), so a later, properly-collected sidecar for a re-generated
    version of the same artifact can at least be compared against this one
    to see whether the *bytes* changed, even though nothing here says what
    produced the original ones.
    """
    return write_provenance_sidecar(
        payload_path,
        environment=dict(_UNKNOWN_ENVIRONMENT),
        provenance_status="unknown_historical",
        note=note,
    )
