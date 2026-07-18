# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA toy-composition probe
"""KYMA toy compositional-generalisation probe.

A small, honesty-gated probe asking whether reusable Kuramoto *motifs*
(learned in-phase / anti-phase relations on cluster pairs) **compose** on a
held-out conjunction never seen jointly in training, and whether a
Kuramoto-dynamics substrate composes better than a parameter-matched
non-motif MLP baseline.

The metric, threshold, and seed count are fixed by the frozen pre-registration
``KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md`` — this package implements
that contract without post-hoc metric shopping.

The Kuramoto integrator here is a purpose-built JAX reimplementation of the
canonical model ``dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)`` (same model the
SCPN-QUANTUM-CONTROL toolkit compiles to hardware), chosen for clean autodiff
through the RK4 solver during motif training.
"""

from __future__ import annotations

from .dynamics import cluster_order_parameter, integrate_kuramoto, kuramoto_rhs
from .task import (
    ProbeConfig,
    TrialBatch,
    build_trials,
    success_mask,
)

__all__ = [
    "ProbeConfig",
    "TrialBatch",
    "build_trials",
    "cluster_order_parameter",
    "integrate_kuramoto",
    "kuramoto_rhs",
    "success_mask",
]
