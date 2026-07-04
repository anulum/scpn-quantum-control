# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Deprecated Kuramoto neural-operator shim
"""Backward-compatible re-export shim for the relocated Kuramoto neural operator.

The DeepONet Kuramoto neural-operator surrogate now lives in the standalone
:mod:`oscillatools.neural_operator` module. This shim keeps
``scpn_quantum_control.forecasting.kuramoto_neural_operator`` importable for the deprecation
window, forwarding every name to :mod:`oscillatools.neural_operator`, and emits a single
:class:`DeprecationWarning` naming the new import path. See ``DEPRECATIONS.md``.
"""

from __future__ import annotations

import warnings as _warnings

import oscillatools.neural_operator as _target
from oscillatools.neural_operator import *  # noqa: F403  (deliberate re-export)

__all__ = list(getattr(_target, "__all__", []))

_warnings.warn(
    "scpn_quantum_control.forecasting.kuramoto_neural_operator has moved to "
    "oscillatools.neural_operator; import from `oscillatools.neural_operator` instead. "
    "This compatibility shim will be removed no earlier than the next major release.",
    DeprecationWarning,
    stacklevel=2,
)
