# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-language acceleration package
"""Multi-language acceleration chain.

Implements the rule codified in ``feedback_multi_language_accel.md``:
every compute function may have one or more acceleration backends
(Rust, Julia, Go, Mojo) with the *measured fastest* at the top of the
fallback chain. Python is always the final fallback.

Current tiers shipped from this package:

* :mod:`.julia` — Julia bindings via ``juliacall``. Activated on first
  successful import; the first Julia call incurs a one-off JIT cost
  (~20 s) that amortises across the process lifetime.

Tiers that exist elsewhere in the repo:

* Rust is shipped via the ``scpn_quantum_engine`` PyO3 wheel at the
  repo root; the dispatchers in this package forward to it by name.

Tiers tracked as future work (empty modules not shipped until a
compute-module actually needs them):

* ``.go`` — for standalone fan-out daemons, tracked under
  ``hardware/async_runner.py`` follow-ups.
* ``.mojo`` — for GPU / MLIR hot paths, not yet relevant at the
  qubit-counts we run today.
"""

from __future__ import annotations

from .daido_observables import (
    daido_order_parameter,
    daido_order_parameter_gradient,
    daido_order_parameter_hessian,
    last_daido_gradient_tier_used,
    last_daido_hessian_tier_used,
    last_daido_tier_used,
)
from .dispatcher import (
    MultiLangDispatcher,
    available_tiers,
    dispatch,
)
from .kuramoto_energy import (
    kuramoto_interaction_energy,
    kuramoto_interaction_energy_gradient,
    kuramoto_interaction_energy_hessian,
    last_kuramoto_interaction_energy_gradient_tier_used,
    last_kuramoto_interaction_energy_hessian_tier_used,
    last_kuramoto_interaction_energy_tier_used,
)
from .kuramoto_mean_field import (
    last_mean_field_force_tier_used,
    last_mean_field_jacobian_tier_used,
    mean_field_force,
    mean_field_jacobian,
)
from .local_order import (
    last_local_order_parameter_jacobian_tier_used,
    last_local_order_parameter_tier_used,
    local_order_parameter,
    local_order_parameter_jacobian,
)
from .mean_phase_observables import (
    last_mean_phase_gradient_tier_used,
    last_mean_phase_hessian_tier_used,
    last_mean_phase_tier_used,
    mean_phase,
    mean_phase_gradient,
    mean_phase_hessian,
)
from .networked_kuramoto import (
    last_networked_kuramoto_force_tier_used,
    last_networked_kuramoto_jacobian_tier_used,
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)
from .order_parameter_observables import (
    last_gradient_tier_used,
    last_hessian_tier_used,
    last_tier_used,
    order_parameter,
    order_parameter_gradient,
    order_parameter_hessian,
)
from .sakaguchi_kuramoto import (
    last_sakaguchi_force_tier_used,
    last_sakaguchi_jacobian_tier_used,
    sakaguchi_force,
    sakaguchi_jacobian,
)

__all__ = [
    "MultiLangDispatcher",
    "available_tiers",
    "daido_order_parameter",
    "daido_order_parameter_gradient",
    "daido_order_parameter_hessian",
    "dispatch",
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "kuramoto_interaction_energy_hessian",
    "last_kuramoto_interaction_energy_gradient_tier_used",
    "last_kuramoto_interaction_energy_hessian_tier_used",
    "last_kuramoto_interaction_energy_tier_used",
    "last_daido_gradient_tier_used",
    "last_daido_hessian_tier_used",
    "last_daido_tier_used",
    "last_gradient_tier_used",
    "last_hessian_tier_used",
    "last_local_order_parameter_jacobian_tier_used",
    "last_local_order_parameter_tier_used",
    "local_order_parameter",
    "local_order_parameter_jacobian",
    "last_mean_field_force_tier_used",
    "last_mean_field_jacobian_tier_used",
    "last_mean_phase_gradient_tier_used",
    "last_mean_phase_hessian_tier_used",
    "last_mean_phase_tier_used",
    "last_tier_used",
    "mean_field_force",
    "mean_field_jacobian",
    "mean_phase",
    "mean_phase_gradient",
    "mean_phase_hessian",
    "networked_kuramoto_force",
    "networked_kuramoto_jacobian",
    "last_networked_kuramoto_force_tier_used",
    "last_networked_kuramoto_jacobian_tier_used",
    "order_parameter",
    "order_parameter_gradient",
    "order_parameter_hessian",
    "sakaguchi_force",
    "sakaguchi_jacobian",
    "last_sakaguchi_force_tier_used",
    "last_sakaguchi_jacobian_tier_used",
]

import numpy as np
from numpy.typing import NDArray


def rust_random_state(n_qubits: int, seed: int = 42) -> NDArray[np.complex128]:
    """Return a normalized complex random state vector for fallback tests."""
    np.random.seed(seed)
    state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)
