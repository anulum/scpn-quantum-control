# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust Source Import Guard
"""Prevent the Rust source directory from shadowing the native extension.

The repository root is on ``sys.path`` during local and CI test runs. Without
this guard, Python can import ``scpn_quantum_engine`` as an empty namespace
package from the Rust crate directory even when the PyO3 extension is not built.
That looks like an installed accelerator but has none of the native symbols.

When a real extension module is installed elsewhere on ``sys.path`` this guard
delegates to it. Otherwise it raises ``ModuleNotFoundError`` so optional Rust
paths are correctly treated as unavailable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REQUIRED_EXPORTS = (
    "build_knm",
    "kuramoto_euler",
    "order_parameter",
)


def _candidate_extension(entry: str) -> importlib.machinery.ModuleSpec | None:
    """Return a native extension spec for this module from ``entry``."""
    spec = importlib.machinery.PathFinder.find_spec(__name__, [entry])
    if spec is None or spec.origin is None:
        return None
    if Path(spec.origin).resolve() == Path(__file__).resolve():
        return None
    if not any(spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        return None
    return spec


def _load_native_extension() -> ModuleType | None:
    """Load an installed native extension while skipping this source tree."""
    source_parent = Path(__file__).resolve().parent.parent
    for raw_entry in sys.path:
        entry = raw_entry or "."
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved == source_parent:
            continue
        spec = _candidate_extension(str(resolved))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[__name__] = module
        spec.loader.exec_module(module)
        return module
    return None


_native = _load_native_extension()
if _native is None or any(
    not callable(getattr(_native, name, None)) for name in _REQUIRED_EXPORTS
):
    sys.modules.pop(__name__, None)
    raise ModuleNotFoundError("scpn_quantum_engine native extension is not built", name=__name__)

globals().update(_native.__dict__)
