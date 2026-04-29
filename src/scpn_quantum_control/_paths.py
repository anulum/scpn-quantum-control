# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project resource path helpers
"""Locate repository-scoped data files from source and installed layouts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

DATA_ROOT_ENV = "SCPN_QUANTUM_CONTROL_DATA_ROOT"


def project_data_root(*required_paths: str) -> Path:
    """Return the project root containing optional repository data markers.

    Source checkouts resolve naturally from ``src/scpn_quantum_control``.
    Installed Docker images resolve through the process working directory
    (``/app`` in this repository's image). Packaged deployments that keep
    data elsewhere can set ``SCPN_QUANTUM_CONTROL_DATA_ROOT``.
    """
    required = tuple(Path(item) for item in required_paths)
    fallback = Path(__file__).resolve().parents[2]
    for candidate in _candidate_roots(fallback):
        if not required or all((candidate / item).exists() for item in required):
            return candidate
    return fallback


def project_data_path(relative_path: str) -> Path:
    """Return an absolute path to a repository-scoped data resource."""
    root = project_data_root(relative_path)
    return root / relative_path


def _candidate_roots(fallback: Path) -> Iterator[Path]:
    seen: set[Path] = set()

    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        yield from _dedupe([Path(env_root).expanduser().resolve()], seen)

    module_path = Path(__file__).resolve()
    yield from _dedupe([fallback, *module_path.parents], seen)

    cwd = Path.cwd().resolve()
    yield from _dedupe([cwd, *cwd.parents], seen)


def _dedupe(paths: list[Path], seen: set[Path]) -> Iterator[Path]:
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        yield path


__all__ = ["DATA_ROOT_ENV", "project_data_path", "project_data_root"]
