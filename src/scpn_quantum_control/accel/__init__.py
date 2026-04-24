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

from .dispatcher import (
    MultiLangDispatcher,
    available_tiers,
    dispatch,
    last_tier_used,
    order_parameter,
)

__all__ = [
    "MultiLangDispatcher",
    "available_tiers",
    "dispatch",
    "last_tier_used",
    "order_parameter",
]

import numpy as np


def rust_random_state(n_qubits: int, seed: int = 42):
    np.random.seed(seed)
    state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    return state / np.linalg.norm(state)
