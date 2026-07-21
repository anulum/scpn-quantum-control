# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 ground-truth teacher
"""The fixed gated-oscillator **teacher** that defines every trial's label.

The teacher is a gated Kuramoto substrate with hand-set motifs and the fixed
ambient coupling. It integrates the true composed dynamics and reads the label
off the achieved phase, so the ground truth is (a) realisable by a gated
substrate and (b) a non-separable, ``θ0``-dependent function of the two active
relations. Both the trainable student (:mod:`.models`) and the MLP baseline are
trained to reproduce these labels; the held-out conjunction is the test.

Motif construction (frozen):

* **R1 (in-phase)** on pair ``p`` — ``+g_sync`` on every intra-pair edge, so the
  eight oscillators phase-lock (``R → 1``).
* **R2 (anti-phase)** on pair ``p`` — ``+g_sync`` within each cluster and
  ``−g_sync`` between the two clusters, so the two clusters lock π apart
  (``R → 0``).

The teacher runs with zero natural-frequency drive (``ω = 0``): the achieved
phases are set purely by ``θ0`` and the gated coupling, which is what makes the
label a function of the input data.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import task
from .coupling import assemble_coupling, base_coupling_matrix, partners_for
from .dynamics import integrate_kuramoto_batched, phase_label
from .task import N_OSC, N_PAIRS, N_RELATIONS, READOUT_OSCILLATOR, ProbeConfigV2, TrialBatchV2


def teacher_gates(g_sync: float) -> NDArray[np.float64]:
    """Hand-set ``(N_RELATIONS, N_PAIRS, N_OSC, N_OSC)`` teacher motif gates."""
    gates = np.zeros((N_RELATIONS, N_PAIRS, N_OSC, N_OSC), dtype=np.float64)
    for p in range(N_PAIRS):
        gates[0, p] = g_sync * task.in_phase_mask(p).astype(np.float64)
        within, between = task.anti_phase_masks(p)
        gates[1, p] = g_sync * (within.astype(np.float64) - between.astype(np.float64))
    return gates


def teacher_final_phases(
    theta0: NDArray[np.float64],
    code: NDArray[np.float64],
    config: ProbeConfigV2,
) -> jnp.ndarray:
    """Integrate the teacher and return the ``(batch, N_OSC)`` achieved phases."""
    gates = jnp.asarray(teacher_gates(config.g_sync))
    partners = partners_for(config.held_out, config.bridge_mode)
    base = jnp.asarray(base_coupling_matrix(config.k_ambient, config.k_bridge, partners))
    coupling = assemble_coupling(jnp.asarray(code), gates, base)
    omega = jnp.zeros((theta0.shape[0], N_OSC))
    return integrate_kuramoto_batched(
        jnp.asarray(theta0), omega, coupling, config.dt, config.steps
    )


def teacher_labels(
    theta0: NDArray[np.float64],
    code: NDArray[np.float64],
    config: ProbeConfigV2,
) -> NDArray[np.int64]:
    """Ground-truth class label per trial (quantised readout phase)."""
    final = teacher_final_phases(theta0, code, config)
    labels = phase_label(final, READOUT_OSCILLATOR, config.n_bins)
    return np.asarray(labels, dtype=np.int64)


def label_batch(batch: TrialBatchV2, config: ProbeConfigV2) -> NDArray[np.int64]:
    """Teacher labels for a whole :class:`TrialBatchV2`."""
    return teacher_labels(batch.theta0, batch.code, config)
