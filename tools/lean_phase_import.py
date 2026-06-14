# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lean phase-module loader
"""Load ``scpn_quantum_control.phase`` leaf modules without the heavy package init.

Importing any name from ``scpn_quantum_control.phase`` executes
``scpn_quantum_control/__init__.py`` (the full hardware-abstraction, mitigation
and analysis surface) followed by ``scpn_quantum_control/phase/__init__.py``
(around fifty framework-bridge submodules). For pure-NumPy phase tooling such as
the QNode affinity benchmark that costs roughly seven seconds of start-up and
pulls optional heavy dependencies (``mitiq``, ``torch``, ``qiskit``) the tool
never calls. ``mitiq`` alone accounts for about 3.2 s of that import.

This loader binds lightweight package shells into ``sys.modules`` so a single
leaf module and its in-package relatives load from the real source files while
the heavy ``__init__`` bodies stay dormant. When the full package is already
imported in the interpreter the real package objects are reused, so behaviour is
identical under pytest where other modules import the package eagerly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT_PACKAGE = "scpn_quantum_control"
_PHASE_PACKAGE = f"{_ROOT_PACKAGE}.phase"
_PHASE_DIR = Path(__file__).resolve().parents[1] / "src" / "scpn_quantum_control" / "phase"
_ROOT_DIR = _PHASE_DIR.parent


def _ensure_package(qualified_name: str, directory: Path) -> ModuleType:
    """Return the package for ``qualified_name``, installing a lean shell if absent.

    A shell is a bare package object carrying ``__path__`` so the import system
    can locate submodules, but without running the real ``__init__`` body. If the
    real package is already imported it is reused unchanged.
    """
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    shell = ModuleType(qualified_name)
    shell.__path__ = [str(directory)]  # type: ignore[attr-defined]
    shell.__package__ = qualified_name
    shell.__lean_shell__ = True  # marker: __init__ body deliberately not executed
    sys.modules[qualified_name] = shell
    return shell


def load_phase_module(submodule: str) -> ModuleType:
    """Load ``scpn_quantum_control.phase.<submodule>`` without the heavy init surfaces.

    ``submodule`` must be a bare module name (no dots or path separators). The
    leaf module and any of its in-package relatives load from the real source
    tree. Returns the already-loaded module if it is present in ``sys.modules``.
    """
    if not submodule.isidentifier():
        raise ValueError(f"submodule must be a bare identifier, got {submodule!r}")
    qualified_name = f"{_PHASE_PACKAGE}.{submodule}"
    cached = sys.modules.get(qualified_name)
    if cached is not None:
        return cached
    _ensure_package(_ROOT_PACKAGE, _ROOT_DIR)
    _ensure_package(_PHASE_PACKAGE, _PHASE_DIR)
    file_path = _PHASE_DIR / f"{submodule}.py"
    if not file_path.is_file():
        raise ModuleNotFoundError(f"no phase module {submodule!r} at {file_path}")
    spec = importlib.util.spec_from_file_location(qualified_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module
