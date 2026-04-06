# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Syndrome Flow Analysis
"""Syndrome information flow between MS-QEC levels.

Analyses how error correction information propagates through the
SCPN hierarchy, weighted by inter-domain K_nm coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multiscale_qec import MultiscaleQECResult, QECLevel, knm_between_domains


@dataclass
class SyndromeFlow:
    """Syndrome propagation between QEC levels."""

    source_level: int
    target_level: int
    syndrome_weight: float  # K_nm coupling strength
    correction_capacity: float  # max correctable weight at target
    information_flow: float  # bits of syndrome per QEC round


def syndrome_flow_between_levels(
    K: np.ndarray,
    level_a: QECLevel,
    level_b: QECLevel,
) -> SyndromeFlow:
    """Compute syndrome information flow between two QEC levels.

    The syndrome weight is proportional to K_nm coupling between
    the corresponding SCPN domains.
    """
    coupling = knm_between_domains(K, level_a.layer_range, level_b.layer_range)
    correction_cap = (level_b.code_distance - 1) / 2.0
    info_flow = coupling * np.log2(max(level_b.code_distance, 2))

    return SyndromeFlow(
        source_level=level_a.level,
        target_level=level_b.level,
        syndrome_weight=coupling,
        correction_capacity=correction_cap,
        information_flow=info_flow,
    )


def syndrome_flow_analysis(
    K: np.ndarray,
    result: MultiscaleQECResult,
) -> list[SyndromeFlow]:
    """Analyse syndrome information flow between all adjacent levels."""
    flows: list[SyndromeFlow] = []
    for i in range(len(result.levels) - 1):
        flow = syndrome_flow_between_levels(K, result.levels[i], result.levels[i + 1])
        flows.append(flow)
    return flows
