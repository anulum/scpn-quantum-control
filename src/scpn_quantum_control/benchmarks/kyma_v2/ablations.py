# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2.1 ablations (fix-removal controls)
"""Ablation models that each remove ONE of the two v2 fixes (KYMA v2.1 #1).

* **A1 — no coupling gating.** A "shared-K" student whose coupling does **not**
  depend on the code (one global learned coupling), the input entering only as an
  additive frequency drive — v1's architecture on the v2 task. A single shared K
  cannot be attractive (in-phase) and frustrated (anti-phase) at once, so this
  should fail to realise both relations (reproducing v1's defect 1).

* **A2 — separable readout.** Not a model but a task setting (``bridge_mode =
  "r1_only"``, :mod:`.coupling`): the readout node is bridged to the R1 cluster
  only, so the label depends on a single relation and is separable — a
  param-matched MLP should then compose it (reproducing v1's defect 2). A2 is run
  through the ordinary :mod:`.probe` with that config, so it needs no model here.

Both ablations predict the substrate's +43 pp advantage disappears — evidence
that each fix is load-bearing.
"""

from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .coupling import base_coupling_matrix, partners_for
from .dynamics import integrate_kuramoto_batched, phase_label
from .models import _adam_descent, _circular_loss  # shared training utilities
from .task import (
    CLUSTERS,
    N_CLUSTER_OSC,
    N_OSC,
    N_PAIRS,
    N_RELATIONS,
    READOUT_OSCILLATOR,
    ProbeConfigV2,
    TrialBatchV2,
)

_Params = dict[str, jax.Array]

# Basis over every edge among the 16 clustered oscillators (the readout node is
# never a motif endpoint). A single shared symmetric coupling is a combination of
# these unit edges — the v1-style "one K for everything".
_CLUSTER_OSC = [o for c in CLUSTERS for o in c]
_ALL_EDGES = list(combinations(_CLUSTER_OSC, 2))
_N_SHARED = len(_ALL_EDGES)  # C(16,2) = 120


def _shared_edge_basis() -> NDArray[np.float64]:
    """``(N_SHARED, N_OSC, N_OSC)`` symmetric unit-edge matrices over cluster edges."""
    basis = np.zeros((_N_SHARED, N_OSC, N_OSC), dtype=np.float64)
    for e, (a, b) in enumerate(_ALL_EDGES):
        basis[e, a, b] = 1.0
        basis[e, b, a] = 1.0
    return basis


_SHARED_BASIS = _shared_edge_basis()


def shared_param_count() -> int:
    """Trainable-parameter count of the shared-K (no-gating) ablation student."""
    return _N_SHARED + N_RELATIONS * N_PAIRS * N_OSC  # shared K + code-drive


def shared_init(seed: int) -> _Params:
    """Small random shared coupling + per-``(relation, pair)`` frequency drive."""
    rng = np.random.default_rng(seed)
    return {
        "k_shared": jnp.asarray(rng.normal(0.0, 0.3, size=_N_SHARED)),
        "drive": jnp.asarray(rng.normal(0.0, 0.3, size=(N_RELATIONS, N_PAIRS, N_OSC))),
    }


def _shared_coupling(k_shared: jax.Array) -> jax.Array:
    """Expand the free shared-edge weights to a symmetric ``(N_OSC, N_OSC)`` matrix."""
    return jnp.einsum("e,eij->ij", k_shared, jnp.asarray(_SHARED_BASIS))


def _drive_for(drive: jax.Array, code: jax.Array) -> jax.Array:
    """``ω(code)`` — additive per-``(relation, pair)`` frequency drive, ``(n, N_OSC)``."""
    return jnp.einsum("nrp,rpo->no", code, drive)


def shared_final_phases(
    params: _Params, theta0: jax.Array, code: jax.Array, base: jax.Array, config: ProbeConfigV2
) -> jax.Array:
    """Integrate the shared-K student (one coupling for every trial; code → drive)."""
    coupling = base[None, :, :] + _shared_coupling(params["k_shared"])[None, :, :]
    omega = _drive_for(params["drive"], code)
    return integrate_kuramoto_batched(theta0, omega, coupling, config.dt, config.steps)


def train_shared(
    batch: TrialBatchV2,
    teacher_finals: NDArray[np.float64],
    config: ProbeConfigV2,
    seed: int,
    epochs: int = 1500,
    lr: float = 0.05,
) -> _Params:
    """Train the shared-K ablation to reproduce the teacher's final phases."""
    tr = ~batch.is_test
    theta0 = jnp.asarray(batch.theta0[tr])
    code = jnp.asarray(batch.code[tr])
    target = jnp.asarray(np.asarray(teacher_finals)[tr])
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    params = shared_init(seed)

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        def loss_fn(p: _Params) -> jax.Array:
            pred = shared_final_phases(p, theta0, code, base, config)
            return _circular_loss(pred, target)

        return _adam_descent(params, loss_fn, lr=lr, epochs=epochs)

    result: _Params = optimise()
    return result


def shared_predict(
    params: _Params, batch_slice: TrialBatchV2, config: ProbeConfigV2
) -> NDArray[np.int64]:
    """Predicted class per trial = quantised shared-K student readout phase."""
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    final = shared_final_phases(
        params, jnp.asarray(batch_slice.theta0), jnp.asarray(batch_slice.code), base, config
    )
    labels = phase_label(final, READOUT_OSCILLATOR, config.n_bins)
    return np.asarray(labels, dtype=np.int64)


# N_CLUSTER_OSC is imported to document intent (the shared basis spans exactly the
# clustered oscillators); assert the basis size matches to catch drift.
assert _N_SHARED == N_CLUSTER_OSC * (N_CLUSTER_OSC - 1) // 2
