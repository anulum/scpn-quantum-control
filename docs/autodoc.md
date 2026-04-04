# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Auto-Generated API Reference

# Auto-Generated API Reference

Generated from source docstrings via mkdocstrings.

## Bridge

::: scpn_quantum_control.bridge.knm_hamiltonian
    options:
      members: [OMEGA_N_16, build_knm_paper27, build_kuramoto_ring, knm_to_hamiltonian, knm_to_ansatz]

::: scpn_quantum_control.bridge.snn_adapter
    options:
      members: [spike_train_to_rotations, quantum_measurement_to_current, SNNQuantumBridge, ArcaneNeuronBridge]

::: scpn_quantum_control.bridge.ssgf_adapter
    options:
      members: [ssgf_w_to_hamiltonian, ssgf_state_to_quantum, quantum_to_ssgf_state, SSGFQuantumLoop]

## Phase

::: scpn_quantum_control.phase.xy_kuramoto
    options:
      members: [QuantumKuramotoSolver]

::: scpn_quantum_control.phase.phase_vqe
    options:
      members: [PhaseVQE]

::: scpn_quantum_control.phase.trotter_upde
    options:
      members: [QuantumUPDESolver]

## Control

::: scpn_quantum_control.control.qaoa_mpc
    options:
      members: [QAOA_MPC]

::: scpn_quantum_control.control.q_disruption_iter
    options:
      members: [ITERFeatureSpec, normalize_iter_features, generate_synthetic_iter_data, from_fusion_core_shot, DisruptionBenchmark]

## QSNN

::: scpn_quantum_control.qsnn.qlif
    options:
      members: [QuantumLIFNeuron]

::: scpn_quantum_control.qsnn.qlayer
    options:
      members: [QuantumDenseLayer]

::: scpn_quantum_control.qsnn.training
    options:
      members: [QSNNTrainer]

## Identity

::: scpn_quantum_control.identity.ground_state
    options:
      members: [IdentityAttractor]

::: scpn_quantum_control.identity.coherence_budget
    options:
      members: [coherence_budget, fidelity_at_depth]

::: scpn_quantum_control.identity.entanglement_witness
    options:
      members: [chsh_from_statevector, disposition_entanglement_map]

::: scpn_quantum_control.identity.identity_key
    options:
      members: [identity_fingerprint, prove_identity, verify_identity]

::: scpn_quantum_control.identity.binding_spec
    options:
      members: [ARCANE_SAPIENCE_SPEC, ORCHESTRATOR_MAPPING, build_identity_attractor, solve_identity, quantum_to_orchestrator_phases, orchestrator_to_quantum_phases]

## Benchmarks

::: scpn_quantum_control.benchmarks.quantum_advantage
    options:
      members: [AdvantageResult, classical_benchmark, quantum_benchmark, estimate_crossover, run_scaling_benchmark]

## QEC

::: scpn_quantum_control.qec.fault_tolerant
    options:
      members: [LogicalQubit, RepetitionCodeUPDE]

::: scpn_quantum_control.qec.surface_code_upde
    options:
      members: [SurfaceCodeSpec, SurfaceCodeUPDE]

## Mitigation

::: scpn_quantum_control.mitigation.pec
    options:
      members: [PECResult, pauli_twirl_decompose, pec_sample]

::: scpn_quantum_control.mitigation.zne
    options:
      members: [ZNEResult, gate_fold_circuit, zne_extrapolate]

## Hardware

::: scpn_quantum_control.hardware.trapped_ion
    options:
      members: [trapped_ion_noise_model, transpile_for_trapped_ion]

::: scpn_quantum_control.hardware.fast_classical
    options:
      members: [fast_sparse_evolution]

::: scpn_quantum_control.hardware.runner
    options:
      members: [HardwareRunner, JobResult]

## Analysis

::: scpn_quantum_control.analysis.sync_witness
    options:
      members: [WitnessResult, evaluate_all_witnesses]

::: scpn_quantum_control.analysis.quantum_persistent_homology
    options:
      members: [QuantumPHResult, quantum_persistent_homology, ph_sync_scan]

::: scpn_quantum_control.analysis.berry_phase
    options:
      members: [BerryPhaseResult, berry_phase_scan]

::: scpn_quantum_control.analysis.finite_size_scaling
    options:
      members: [FSSResult, finite_size_scaling]

::: scpn_quantum_control.analysis.krylov_complexity
    options:
      members: [KrylovResult, krylov_vs_coupling]

::: scpn_quantum_control.analysis.magic_nonstabilizerness
    options:
      members: [MagicResult, magic_vs_coupling]

## Crypto

::: scpn_quantum_control.crypto
    options:
      show_root_heading: true

## Applications

::: scpn_quantum_control.applications
    options:
      show_root_heading: true

## Gauge

::: scpn_quantum_control.gauge
    options:
      show_root_heading: true

## TCBO

::: scpn_quantum_control.tcbo
    options:
      show_root_heading: true

## PGBO

::: scpn_quantum_control.pgbo
    options:
      show_root_heading: true

## L16

::: scpn_quantum_control.l16
    options:
      show_root_heading: true
