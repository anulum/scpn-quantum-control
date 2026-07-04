# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Deprecated Kuramoto facade shim
"""Backward-compatible re-export shim for the relocated Kuramoto toolkit facade.

The unified Kuramoto facade now lives in the standalone :mod:`oscillatools` distribution.
This shim keeps ``scpn_quantum_control.kuramoto`` importable for the deprecation window,
forwarding every name to :mod:`oscillatools`, and emits a single :class:`DeprecationWarning`
naming the new import path. See ``DEPRECATIONS.md``.
"""

from __future__ import annotations

import warnings as _warnings

import oscillatools as _target
from oscillatools import *  # noqa: F403  (deliberate re-export of the relocated facade)

__all__ = list(getattr(_target, "__all__", []))

_warnings.warn(
    "scpn_quantum_control.kuramoto has moved to oscillatools; import from `oscillatools` "
    "instead. This compatibility shim will be removed no earlier than the next major release.",
    DeprecationWarning,
    stacklevel=2,
)
