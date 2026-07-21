# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — reviewed QPY artefact loader (campaign scripts)
"""Reviewed QPY loader for repository-local campaign artefacts.

QPY deserialisation is an unsafe-surface API
(``tools/audit_serialization_surface.py``); every call site must go through
an explicitly reviewed wrapper. The HAL wrapper
(``hardware/hal_qiskit._reviewed_qpy_load_circuits``) covers trusted
in-process bytes but sits behind the full package import chain, which the
isolated ``.venv-iqm`` cannot import. This module is the standalone reviewed
wrapper for the campaign scripts: no package imports, loadable by file path
in either virtual environment.

Review boundary: it refuses anything that is not a ``.qpy`` file inside this
repository's ``data/`` tree — the campaign artefacts written by
``scripts/iqm_layout_transfer_harness.py prepare`` from committed code. It
never loads bytes from outside the repository, from network input, or from
user-supplied paths outside ``data/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_ROOT = (_REPO_ROOT / "data").resolve()


def reviewed_qpy_load_circuits(path: str | Path) -> list[Any]:
    """Load circuits from a repository-local ``data/*.qpy`` artefact.

    Fails closed on any path outside the repository ``data/`` tree or
    without the ``.qpy`` suffix.
    """
    resolved = Path(path).resolve()
    if resolved.suffix != ".qpy":
        raise ValueError(f"refusing non-QPY artefact: {resolved.name}")
    if not resolved.is_relative_to(_ARTIFACT_ROOT):
        raise ValueError(f"refusing QPY artefact outside the repository data tree: {resolved}")
    from qiskit import qpy

    with resolved.open("rb") as stream:
        return list(qpy.load(stream))
