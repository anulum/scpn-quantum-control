# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 gated-coupling assembly (fix 1)
"""Assemble the per-trial gated coupling ``K_eff(code)`` (KYMA v2 fix 1).

The control code gates a set of per-``(relation, pair)`` coupling perturbations
on top of a fixed **base coupling**::

    K_eff[trial] = K_base + Σ_r Σ_p  code[trial, r, p] · gates[r, p]

``gates`` is a ``(N_RELATIONS, N_PAIRS, N_OSC, N_OSC)`` tensor whose ``[r, p]``
slice is masked to cluster-pair ``p`` — the reusable motif for relation ``r`` on
pair ``p``. The *teacher* uses hand-set gates (:mod:`.teacher`); the *student*
learns them (:mod:`.models`).

The base coupling is fixed and shared so neither model can remove the interaction
that makes the readout non-separable (fix 2). It has two terms:

* an optional small **uniform ambient** coupling, and
* a **readout bridge** — sparse fixed edges from the readout oscillator to one
  representative of each other cluster.

The mechanism-only sanity check (:mod:`.design`) showed a *uniform* ambient
strong enough to make the readout non-separable also destroys the anti-phase
motif (attraction fights frustration). The **readout bridge** avoids that: it
touches only the readout node's links to other clusters, leaving each motif's
internal frustration intact, while still making the readout phase depend on the
other active relation's ``θ0``-dependent state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .task import CLUSTERS, HELD_OUT_R1_PAIR, HELD_OUT_R2_PAIR, N_OSC, PAIRS, READOUT_OSCILLATOR


def ambient_matrix(k_ambient: float) -> NDArray[np.float64]:
    """Uniform ambient coupling on every off-diagonal edge (zero diagonal)."""
    mat = np.full((N_OSC, N_OSC), float(k_ambient), dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    return mat


def partners_for(
    held_out: tuple[int, int] | None = None, bridge_mode: str = "both"
) -> tuple[int, ...]:
    """Readout-bridge partner oscillators for a held-out conjunction.

    The passive readout listens to a *single* oscillator of one held-out R1
    cluster and one held-out R2 cluster (never both clusters of an anti-phase
    pair), so it reads each relation's coherent branch phase without the
    mean-field cancellation that makes an anti-phase pair invisible.

    Args:
        held_out: the ``(r1_pair, r2_pair)`` split; ``None`` → the frozen default.
        bridge_mode: ``"both"`` (default, one partner per relation) or ``"r1_only"``
            (the A2 ablation — bridge to the R1 cluster only, making the readout
            depend on a single relation → separable).
    """
    r1_pair, r2_pair = (HELD_OUT_R1_PAIR, HELD_OUT_R2_PAIR) if held_out is None else held_out
    r1_partner = CLUSTERS[PAIRS[r1_pair][0]][0]
    r2_partner = CLUSTERS[PAIRS[r2_pair][0]][0]
    if bridge_mode == "r1_only":
        return (r1_partner,)
    return (r1_partner, r2_partner)


def readout_bridge_matrix(
    k_bridge: float, partners: tuple[int, ...] | None = None
) -> NDArray[np.float64]:
    """Sparse symmetric bridge from the passive readout node to its partners."""
    if partners is None:
        partners = partners_for()
    mat = np.zeros((N_OSC, N_OSC), dtype=np.float64)
    for partner in partners:
        mat[READOUT_OSCILLATOR, partner] = float(k_bridge)
        mat[partner, READOUT_OSCILLATOR] = float(k_bridge)
    return mat


def base_coupling_matrix(
    k_ambient: float, k_bridge: float, partners: tuple[int, ...] | None = None
) -> NDArray[np.float64]:
    """Fixed base coupling = uniform ambient + readout bridge."""
    return ambient_matrix(k_ambient) + readout_bridge_matrix(k_bridge, partners)


def assemble_coupling(code: jax.Array, gates: jax.Array, base: jax.Array) -> jax.Array:
    """Per-trial ``K_eff = base + Σ code·gates``.

    Args:
        code: ``(batch, N_RELATIONS, N_PAIRS)`` binary gate activations.
        gates: ``(N_RELATIONS, N_PAIRS, N_OSC, N_OSC)`` motif perturbations.
        base: ``(N_OSC, N_OSC)`` fixed base coupling (ambient + bridge).

    Returns:
        ``(batch, N_OSC, N_OSC)`` symmetric per-trial coupling, zero diagonal.
    """
    gated = jnp.einsum("nrp,rpij->nij", code, gates)
    return base[None, :, :] + gated


def symmetrise(gates: jax.Array) -> jax.Array:
    """Symmetrise a gate tensor and zero its diagonal (per ``[r, p]`` slice).

    The student stores free entries; this projects them onto the symmetric,
    zero-diagonal coupling manifold so ``K_eff`` is always a valid Kuramoto
    coupling regardless of the raw parameters.
    """
    sym = 0.5 * (gates + jnp.swapaxes(gates, -1, -2))
    eye = jnp.eye(N_OSC)
    return sym * (1.0 - eye)[None, None, :, :]
