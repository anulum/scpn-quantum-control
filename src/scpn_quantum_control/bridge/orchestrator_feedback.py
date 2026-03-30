# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Orchestrator Feedback
"""Orchestrator bidirectional feedback loop.

The existing orchestrator adapter (orchestrator_adapter.py) maps
quantum phases to orchestrator state, but changes don't drive
orchestrator decisions. This module closes the feedback loop:

    Quantum state → R_global, entanglement, stability
    → Orchestrator action: advance phase, hold, rollback

Actions:
    - R_global > 0.8 AND stable → "advance" (proceed to next phase)
    - R_global > 0.5 AND adjusting → "hold" (continue current phase)
    - R_global < 0.5 OR unstable → "rollback" (return to previous phase)

The orchestrator uses quantum observables as decision criteria for
the SCPN phase lifecycle, making quantum evolution a first-class
input to the cybernetic control loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..l16.quantum_director import compute_l16_lyapunov


@dataclass
class OrchestratorFeedback:
    """Feedback from quantum state to orchestrator."""

    action: str  # "advance", "hold", "rollback"
    r_global: float
    stability_score: float
    l16_action: str
    confidence: float  # 0-1, how certain the recommendation is
    reason: str


def compute_orchestrator_feedback(
    K: np.ndarray,
    omega: np.ndarray,
    r_advance: float = 0.8,
    r_hold: float = 0.5,
) -> OrchestratorFeedback:
    """Compute quantum-informed feedback for the orchestrator.

    Args:
        K: coupling matrix
        omega: natural frequencies
        r_advance: R threshold for phase advancement
        r_hold: R threshold for hold (below = rollback)
    """
    l16 = compute_l16_lyapunov(K, omega)
    r = l16.order_parameter
    stability = l16.stability_score

    if r >= r_advance and l16.action == "continue":
        action = "advance"
        confidence = min(r, stability)
        reason = f"R={r:.3f} >= {r_advance}, stable"
    elif r >= r_hold:
        action = "hold"
        confidence = (r - r_hold) / (r_advance - r_hold)
        reason = f"R={r:.3f} in [{r_hold}, {r_advance}), monitoring"
    else:
        action = "rollback"
        confidence = 1.0 - r / r_hold
        reason = f"R={r:.3f} < {r_hold}, desynchronised"

    return OrchestratorFeedback(
        action=action,
        r_global=r,
        stability_score=stability,
        l16_action=l16.action,
        confidence=float(confidence),
        reason=reason,
    )
