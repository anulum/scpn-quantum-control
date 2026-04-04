# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiments
"""Concrete hardware experiments for IBM Quantum.

Each experiment function takes a HardwareRunner and returns results + classical
reference for comparison. Designed to fit within the 10-min/month free tier.

20 experiments total. See docs/EXPERIMENT_ROADMAP.md for budget allocation.

This module re-exports all experiment functions from their respective
sub-modules so that existing ``from .experiments import ...`` statements
continue to work unchanged.
"""

from __future__ import annotations

from ._experiment_helpers import (
    _build_evo_base,
    _build_xyz_circuits,
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _R_from_xyz,
)
from .experiment_control import (
    bell_test_4q_experiment,
    correlator_4q_experiment,
    qaoa_mpc_4_experiment,
    qkd_qber_4q_experiment,
    upde_16_snapshot_experiment,
)
from .experiment_dynamics import (
    kuramoto_4osc_experiment,
    kuramoto_4osc_trotter2_experiment,
    kuramoto_8osc_experiment,
    sync_threshold_experiment,
)
from .experiment_mitigation import (
    decoherence_scaling_experiment,
    kuramoto_4osc_zne_experiment,
    kuramoto_8osc_zne_experiment,
    noise_baseline_experiment,
    upde_16_dd_experiment,
    zne_higher_order_experiment,
)
from .experiment_vqe import (
    _run_vqe,
    ansatz_comparison_hw_experiment,
    vqe_4q_experiment,
    vqe_8q_experiment,
    vqe_8q_hardware_experiment,
    vqe_landscape_experiment,
)

__all__ = [
    # Helpers
    "_build_evo_base",
    "_build_xyz_circuits",
    "_correlator_from_counts",
    "_expectation_per_qubit",
    "_qaoa_cost_from_counts",
    "_R_from_xyz",
    "_run_vqe",
    # Dynamics
    "kuramoto_4osc_experiment",
    "kuramoto_4osc_trotter2_experiment",
    "kuramoto_8osc_experiment",
    "sync_threshold_experiment",
    # VQE
    "ansatz_comparison_hw_experiment",
    "vqe_4q_experiment",
    "vqe_8q_experiment",
    "vqe_8q_hardware_experiment",
    "vqe_landscape_experiment",
    # Mitigation
    "decoherence_scaling_experiment",
    "kuramoto_4osc_zne_experiment",
    "kuramoto_8osc_zne_experiment",
    "noise_baseline_experiment",
    "upde_16_dd_experiment",
    "zne_higher_order_experiment",
    # Control
    "bell_test_4q_experiment",
    "correlator_4q_experiment",
    "qaoa_mpc_4_experiment",
    "qkd_qber_4q_experiment",
    "upde_16_snapshot_experiment",
    # Registry
    "ALL_EXPERIMENTS",
]

ALL_EXPERIMENTS = {
    "kuramoto_4osc": kuramoto_4osc_experiment,
    "kuramoto_8osc": kuramoto_8osc_experiment,
    "vqe_4q": vqe_4q_experiment,
    "vqe_8q": vqe_8q_experiment,
    "qaoa_mpc_4": qaoa_mpc_4_experiment,
    "upde_16_snapshot": upde_16_snapshot_experiment,
    "kuramoto_4osc_zne": kuramoto_4osc_zne_experiment,
    "noise_baseline": noise_baseline_experiment,
    "kuramoto_8osc_zne": kuramoto_8osc_zne_experiment,
    "vqe_8q_hardware": vqe_8q_hardware_experiment,
    "upde_16_dd": upde_16_dd_experiment,
    "kuramoto_4osc_trotter2": kuramoto_4osc_trotter2_experiment,
    "sync_threshold": sync_threshold_experiment,
    "ansatz_comparison_hw": ansatz_comparison_hw_experiment,
    "zne_higher_order": zne_higher_order_experiment,
    "decoherence_scaling": decoherence_scaling_experiment,
    "vqe_landscape": vqe_landscape_experiment,
    "bell_test_4q": bell_test_4q_experiment,
    "correlator_4q": correlator_4q_experiment,
    "qkd_qber_4q": qkd_qber_4q_experiment,
}
