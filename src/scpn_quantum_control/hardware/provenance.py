# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Provenance capture
"""Run-time provenance capture for hardware and simulator results.

Every `HardwareRunner` JSON write embeds the output of
`capture_provenance()` under the `provenance` key. An outsider can
then trace the numbers in a result file back to:

- the exact commit of `scpn-quantum-control` that produced them,
- the `scpn_quantum_engine` (Rust) version if installed,
- the Qiskit + Qiskit-IBM-Runtime versions,
- the Python interpreter version and platform,
- the runner host-name (anonymised to the first 8 chars of its
  SHA-256 hash when `SCPN_ANONYMOUS_HOSTNAME=1` is set, otherwise
  the raw string — decided per deployment),
- the wall-clock timestamp in UTC ISO-8601.

The function is deliberately best-effort: any field that cannot be
resolved (e.g. running outside a git checkout) falls back to
`"unknown"` rather than raising. Provenance must never block a
submit or save.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

__all__ = ["capture_provenance"]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _optional_engine_version() -> str:
    try:
        mod = importlib.import_module("scpn_quantum_engine")
    except ImportError:
        return "not installed"
    # PyO3 modules usually expose __version__; fall back to metadata.
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    return _pkg_version("scpn-quantum-engine")


def _hostname() -> str:
    try:
        host = socket.gethostname()
    except OSError:
        return "unknown"
    if _anonymous_hostname_enabled():
        return "h" + hashlib.sha256(host.encode("utf-8")).hexdigest()[:8]
    return host


def _anonymous_hostname_enabled() -> bool:
    """Single point for the anonymous-hostname toggle.

    Prefers the typed :class:`SCPNConfig` when ``pydantic-settings`` is
    available; otherwise falls back to the legacy
    ``SCPN_ANONYMOUS_HOSTNAME=1`` environment variable. The fallback
    keeps ``capture_provenance()`` working on minimal installs that
    haven't pulled the ``[config]`` extra.
    """
    try:
        from ..config import get_config

        return bool(get_config().anonymous_hostname)
    except Exception:
        return os.environ.get("SCPN_ANONYMOUS_HOSTNAME") == "1"


def capture_provenance() -> dict:
    """Return a JSON-serialisable dict describing the current run.

    All fields are best-effort; missing values fall back to
    `"unknown"` / `"not installed"` rather than raising.
    """
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "short": _git("rev-parse", "--short", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": _git("describe", "--tags", "--always", "--dirty"),
            "dirty": _git("status", "--porcelain") != "",
        },
        "versions": {
            "scpn_quantum_control": _pkg_version("scpn-quantum-control"),
            "scpn_quantum_engine": _optional_engine_version(),
            "qiskit": _pkg_version("qiskit"),
            "qiskit_ibm_runtime": _pkg_version("qiskit-ibm-runtime"),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": _hostname(),
        },
    }
