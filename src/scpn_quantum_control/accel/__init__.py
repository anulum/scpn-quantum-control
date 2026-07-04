# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Deprecated Kuramoto accelerator shim
"""Backward-compatible re-export shim for the relocated Kuramoto accelerators.

The accelerated Kuramoto primitives now live in the standalone :mod:`oscillatools.accel`
distribution. This shim keeps ``scpn_quantum_control.accel`` — both the aggregated public
names and the individual submodules (``scpn_quantum_control.accel.networked_kuramoto`` …) —
importable for the deprecation window, forwarding every name to :mod:`oscillatools.accel`
and preserving object identity for already-loaded submodules. It emits a single
:class:`DeprecationWarning` naming the new import path. See ``DEPRECATIONS.md``.
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings

import oscillatools.accel as _target
from oscillatools.accel import *  # noqa: F403  (deliberate re-export of the relocated surface)

# Resolve submodule imports (``scpn_quantum_control.accel.<name>``) against the relocated
# package directory, and alias every already-loaded submodule so the two dotted paths refer
# to the same module object (preserving ``is`` identity across the shim boundary).
__path__ = list(_target.__path__)
_prefix = _target.__name__ + "."
for _name in [_n for _n in _sys.modules if _n.startswith(_prefix)]:
    _sys.modules[__name__ + "." + _name[len(_prefix) :]] = _sys.modules[_name]

__all__ = list(getattr(_target, "__all__", []))

_warnings.warn(
    "scpn_quantum_control.accel has moved to oscillatools.accel; import from "
    "`oscillatools.accel` instead. This compatibility shim will be removed no earlier "
    "than the next major release.",
    DeprecationWarning,
    stacklevel=2,
)
