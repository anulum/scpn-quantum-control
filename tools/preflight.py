# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Local CI preflight — mirrors every CI gate so failures are caught before push.

Gates (in order):
  1. repository lint, format, documentation, generated-surface, policy, and security audits
  2. focused strict-typing and NumPy-docstring ratchets for promoted owner cohorts
  3. Rust formatting, version/export consistency, and repository typing gates
  4. exact MLIR-leaf, Phase-QNode-affinity, Phase-QNode-vector, Phase-QNode
     JAX, Studio Program-AD, and trace-value statement/branch coverage (default
     coverage mode only)
  5. repository pytest with the selected coverage mode
  6. Bandit security scan

Each exact owner gate uses an explicit responsibility-scoped test cohort, an
isolated coverage data file, and a 100% statement/branch threshold for only its
named production modules. ``--no-tests`` and ``--no-coverage`` skip those
executable coverage gates while retaining their static typing/docstring
ratchets.

Usage:
  python tools/preflight.py                # all gates (default)
  python tools/preflight.py --no-tests     # skip pytest entirely (quick lint pass)
  python tools/preflight.py --no-coverage  # run tests without coverage threshold
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable
from importlib import import_module
from os import X_OK, access, devnull, environ, pathsep
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools import advanced_witnesses_quality_gates as _advanced_witnesses_quality_gates
    from tools import (
        advantage_language_protocol_quality_gates as _advantage_language_protocol_quality_gates,
    )
    from tools import application_honesty_quality_gates as _application_honesty_quality_gates
    from tools import attested_result_pack_quality_gates as _attested_result_pack_quality_gates
    from tools import bench_cli_quality_gates as _bench_cli_quality_gates
    from tools import (
        campaign_harness_product_quality_gates as _campaign_harness_product_quality_gates,
    )
    from tools import chimera_control_quality_gates as _chimera_control_quality_gates
    from tools import (
        cloud_native_deployment_product_quality_gates as _cloud_native_deployment_quality_gates,
    )
    from tools import codesign_components_quality_gates as _codesign_components_quality_gates
    from tools import (
        competitive_baseline_watch_quality_gates as _competitive_baseline_watch_quality_gates,
    )
    from tools import (
        compiler_boundary_product_quality_gates as _compiler_boundary_product_quality_gates,
    )
    from tools import (
        control_stack_compose_product_quality_gates as _control_stack_compose_quality_gates,
    )
    from tools import coupling_recovery_quality_gates as _coupling_recovery_quality_gates
    from tools import coverage_frontier_quality_gates as _coverage_frontier_quality_gates
    from tools import (
        custom_derivatives_product_quality_gates as _custom_derivatives_product_quality_gates,
    )
    from tools import decisive_advantage_quality_gates as _decisive_advantage_quality_gates
    from tools import diff_contract_audit_quality_gates as _diff_contract_audit_quality_gates
    from tools import (
        differentiable_audit_contracts_quality_gates as _differentiable_audit_contracts_quality_gates,
    )
    from tools import (
        differentiable_notebook_curriculum_quality_gates as _differentiable_notebook_curriculum_quality_gates,
    )
    from tools import (
        differentiable_parameter_shift_quality_gates as _differentiable_parameter_shift_quality_gates,
    )
    from tools import differentiable_quality_gates as _differentiable_quality_gates
    from tools import (
        differentiable_sparse_derivatives_quality_gates as _differentiable_sparse_derivatives_quality_gates,
    )
    from tools import (
        differentiable_transform_support_matrix_artifact_quality_gates as _transform_support_matrix_artifact_quality_gates,
    )
    from tools import (
        dla_topology_objectives_quality_gates as _dla_topology_objectives_quality_gates,
    )
    from tools import (
        dla_topology_optimizer_quality_gates as _dla_topology_optimizer_quality_gates,
    )
    from tools import dla_topology_parity_quality_gates as _dla_topology_parity_quality_gates
    from tools import (
        dla_topology_projection_quality_gates as _dla_topology_projection_quality_gates,
    )
    from tools import dla_topology_schema_quality_gates as _dla_topology_schema_quality_gates
    from tools import enaqt_evidence_quality_gates as _enaqt_evidence_quality_gates
    from tools import (
        entanglement_sync_evidence_quality_gates as _entanglement_sync_evidence_quality_gates,
    )
    from tools import (
        error_mitigation_product_quality_gates as _error_mitigation_product_quality_gates,
    )
    from tools import experiment_mitigation_quality_gates as _experiment_mitigation_quality_gates
    from tools import external_validation_quality_gates as _external_validation_quality_gates
    from tools import (
        fault_tolerant_resource_product_quality_gates as _fault_tolerant_resource_product_quality_gates,
    )
    from tools import feedback_loop_quality_gates as _feedback_loop_quality_gates
    from tools import finite_size_scaling_quality_gates as _finite_size_scaling_quality_gates
    from tools import fusion_core_frc_bridge_quality_gates as _fusion_core_frc_bridge_quality_gates
    from tools import (
        geometric_control_product_quality_gates as _geometric_control_product_quality_gates,
    )
    from tools import governed_route_matrix_quality_gates as _governed_route_matrix_quality_gates
    from tools import (
        gradient_plan_explanation_artifact_quality_gates as _gradient_plan_explanation_artifact_quality_gates,
    )
    from tools import gradient_tape_quality_gates as _gradient_tape_quality_gates
    from tools import (
        hardware_experiment_control_quality_gates as _hardware_experiment_control_quality_gates,
    )
    from tools import (
        hardware_experiment_vqe_quality_gates as _hardware_experiment_vqe_quality_gates,
    )
    from tools import hardware_hal_quality_gates as _hardware_hal_quality_gates
    from tools import hardware_safe_execution_quality_gates as _hardware_safe_quality_gates
    from tools import (
        hermetic_reproduction_kit_quality_gates as _hermetic_reproduction_kit_quality_gates,
    )
    from tools import (
        hls_cosimulation_evidence_quality_gates as _hls_cosimulation_evidence_quality_gates,
    )
    from tools import identity_binding_spec_quality_gates as _identity_binding_spec_quality_gates
    from tools import (
        kuramoto_layout_cost_quality_gates as _kuramoto_layout_cost_quality_gates,
    )
    from tools import (
        kuramoto_layout_relaxation_quality_gates as _kuramoto_layout_relaxation_quality_gates,
    )
    from tools import kyma_dynamics_quality_gates as _kyma_dynamics_quality_gates
    from tools import (
        kyma_mechanism_benchmark_product_quality_gates as _kyma_mechanism_product_quality_gates,
    )
    from tools import kyma_v2_dynamics_quality_gates as _kyma_v2_dynamics_quality_gates
    from tools import (
        layout_method_comparison_quality_gates as _layout_method_comparison_quality_gates,
    )
    from tools import (
        metamorphic_ad_verification_quality_gates as _metamorphic_ad_verification_quality_gates,
    )
    from tools import (
        migration_guides_product_quality_gates as _migration_guides_product_quality_gates,
    )
    from tools import ml_dsa_seal_quality_gates as _ml_dsa_seal_quality_gates
    from tools import multi_hal_federation_product_quality_gates as _multi_hal_quality_gates
    from tools import (
        neural_operator_baseline_product_quality_gates as _neural_operator_baseline_product_quality_gates,
    )
    from tools import (
        neural_operator_cost_model_quality_gates as _neural_operator_cost_model_quality_gates,
    )
    from tools import (
        open_system_completeness_quality_gates as _open_system_completeness_quality_gates,
    )
    from tools import (
        open_system_objective_quality_gates as _open_system_objective_quality_gates,
    )
    from tools import openpulse_control_quality_gates as _openpulse_control_quality_gates
    from tools import phase_jax_qnode_quality_gates as _phase_jax_qnode_quality_gates
    from tools import phase_qnode_product_quality_gates as _phase_qnode_product_quality_gates
    from tools import phase_trainability_quality_gates as _phase_trainability_quality_gates
    from tools import (
        polyglot_parity_certificate_quality_gates as _polyglot_parity_certificate_quality_gates,
    )
    from tools import program_ad_adjoint_quality_gates as _program_ad_adjoint_quality_gates
    from tools import program_ad_array_indexing_quality_gates as _array_indexing_quality_gates
    from tools import (
        program_ad_fuzz_assurance_quality_gates as _program_ad_fuzz_assurance_quality_gates,
    )
    from tools import program_ad_quality_gates as _program_ad_quality_gates
    from tools import public_api_stability_quality_gates as _public_api_stability_quality_gates
    from tools import pulse_shaping_quality_gates as _pulse_shaping_quality_gates
    from tools import (
        qnode_circuit_contracts_quality_gates as _qnode_circuit_contracts_quality_gates,
    )
    from tools import qpu_compute_product_quality_gates as _qpu_compute_product_quality_gates
    from tools import qpu_compute_types_quality_gates as _qpu_compute_types_quality_gates
    from tools import (
        quantum_sync_oracle_product_quality_gates as _quantum_sync_oracle_product_quality_gates,
    )
    from tools import research_lane_registry_quality_gates as _research_lane_registry_quality_gates
    from tools import resource_budget_gate_quality_gates as _resource_budget_gate_quality_gates
    from tools import (
        scorecard_acceptance_engine_quality_gates as _scorecard_acceptance_engine_quality_gates,
    )
    from tools import (
        ssgf_geometry_gradient_quality_gates as _ssgf_geometry_gradient_quality_gates,
    )
    from tools import stable_core_product_quality_gates as _stable_core_product_quality_gates
    from tools import (
        stochastic_estimators_product_quality_gates as _stochastic_estimators_product_quality_gates,
    )
    from tools import (
        studio_executive_benchmark_quality_gates as _studio_executive_benchmark_quality_gates,
    )
    from tools import (
        studio_executive_differentiate_quality_gates as _studio_executive_differentiate_quality_gates,
    )
    from tools import studio_executive_product_quality_gates as _studio_executive_quality_gates
    from tools import (
        synchronisation_witness_quality_gates as _synchronisation_witness_quality_gates,
    )
    from tools import theory_hook_promotion_quality_gates as _theory_hook_promotion_quality_gates
    from tools import (
        thermo_readiness_product_quality_gates as _thermo_readiness_product_quality_gates,
    )
    from tools import (
        tn_mps_baseline_design_quality_gates as _tn_mps_baseline_design_quality_gates,
    )
    from tools import (
        topology_kernel_evidence_quality_gates as _topology_kernel_evidence_quality_gates,
    )
    from tools import topology_kernel_schema_quality_gates as _topology_kernel_schema_quality_gates
    from tools import (
        unsuitable_scenario_registry_quality_gates as _unsuitable_scenario_registry_quality_gates,
    )
    from tools import variational_metric_quality_gates as _variational_metric_quality_gates
    from tools import (
        visualisation_dashboard_product_quality_gates as _visualisation_dashboard_product_quality_gates,
    )
    from tools import (
        whole_program_ad_product_quality_gates as _whole_program_ad_product_quality_gates,
    )
    from tools import (
        whole_program_frontend_contracts_quality_gates as _whole_program_frontend_contracts_quality_gates,
    )
    from tools import (
        wirtinger_implicit_product_quality_gates as _wirtinger_implicit_product_quality_gates,
    )
else:
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    _advanced_witnesses_quality_gates = import_module("tools.advanced_witnesses_quality_gates")
    _attested_result_pack_quality_gates = import_module("tools.attested_result_pack_quality_gates")
    _application_honesty_quality_gates = import_module("tools.application_honesty_quality_gates")
    _advantage_language_protocol_quality_gates = import_module(
        "tools.advantage_language_protocol_quality_gates"
    )
    _campaign_harness_product_quality_gates = import_module(
        "tools.campaign_harness_product_quality_gates"
    )
    _chimera_control_quality_gates = import_module("tools.chimera_control_quality_gates")
    _whole_program_frontend_contracts_quality_gates = import_module(
        "tools.whole_program_frontend_contracts_quality_gates"
    )
    _variational_metric_quality_gates = import_module("tools.variational_metric_quality_gates")
    _qnode_circuit_contracts_quality_gates = import_module(
        "tools.qnode_circuit_contracts_quality_gates"
    )
    _topology_kernel_schema_quality_gates = import_module(
        "tools.topology_kernel_schema_quality_gates"
    )
    _topology_kernel_evidence_quality_gates = import_module(
        "tools.topology_kernel_evidence_quality_gates"
    )
    _geometric_control_product_quality_gates = import_module(
        "tools.geometric_control_product_quality_gates"
    )
    _error_mitigation_product_quality_gates = import_module(
        "tools.error_mitigation_product_quality_gates"
    )
    _tn_mps_baseline_design_quality_gates = import_module(
        "tools.tn_mps_baseline_design_quality_gates"
    )
    _cloud_native_deployment_quality_gates = import_module(
        "tools.cloud_native_deployment_product_quality_gates"
    )
    _compiler_boundary_product_quality_gates = import_module(
        "tools.compiler_boundary_product_quality_gates"
    )
    _coverage_frontier_quality_gates = import_module("tools.coverage_frontier_quality_gates")
    _competitive_baseline_watch_quality_gates = import_module(
        "tools.competitive_baseline_watch_quality_gates"
    )
    _whole_program_ad_product_quality_gates = import_module(
        "tools.whole_program_ad_product_quality_gates"
    )
    _control_stack_compose_quality_gates = import_module(
        "tools.control_stack_compose_product_quality_gates"
    )
    _public_api_stability_quality_gates = import_module("tools.public_api_stability_quality_gates")
    _qpu_compute_product_quality_gates = import_module("tools.qpu_compute_product_quality_gates")
    _wirtinger_implicit_product_quality_gates = import_module(
        "tools.wirtinger_implicit_product_quality_gates"
    )
    _custom_derivatives_product_quality_gates = import_module(
        "tools.custom_derivatives_product_quality_gates"
    )
    _decisive_advantage_quality_gates = import_module("tools.decisive_advantage_quality_gates")
    _coupling_recovery_quality_gates = import_module("tools.coupling_recovery_quality_gates")
    _diff_contract_audit_quality_gates = import_module("tools.diff_contract_audit_quality_gates")
    _differentiable_audit_contracts_quality_gates = import_module(
        "tools.differentiable_audit_contracts_quality_gates"
    )
    _transform_support_matrix_artifact_quality_gates = import_module(
        "tools.differentiable_transform_support_matrix_artifact_quality_gates"
    )
    _gradient_plan_explanation_artifact_quality_gates = import_module(
        "tools.gradient_plan_explanation_artifact_quality_gates"
    )
    _differentiable_quality_gates = import_module("tools.differentiable_quality_gates")
    _experiment_mitigation_quality_gates = import_module(
        "tools.experiment_mitigation_quality_gates"
    )
    _external_validation_quality_gates = import_module("tools.external_validation_quality_gates")
    _fault_tolerant_resource_product_quality_gates = import_module(
        "tools.fault_tolerant_resource_product_quality_gates"
    )
    _feedback_loop_quality_gates = import_module("tools.feedback_loop_quality_gates")
    _finite_size_scaling_quality_gates = import_module("tools.finite_size_scaling_quality_gates")
    _fusion_core_frc_bridge_quality_gates = import_module(
        "tools.fusion_core_frc_bridge_quality_gates"
    )
    _hardware_experiment_control_quality_gates = import_module(
        "tools.hardware_experiment_control_quality_gates"
    )
    _hardware_experiment_vqe_quality_gates = import_module(
        "tools.hardware_experiment_vqe_quality_gates"
    )
    _gradient_tape_quality_gates = import_module("tools.gradient_tape_quality_gates")
    _hardware_hal_quality_gates = import_module("tools.hardware_hal_quality_gates")
    _openpulse_control_quality_gates = import_module("tools.openpulse_control_quality_gates")
    _pulse_shaping_quality_gates = import_module("tools.pulse_shaping_quality_gates")
    _dla_topology_optimizer_quality_gates = import_module(
        "tools.dla_topology_optimizer_quality_gates"
    )
    _dla_topology_objectives_quality_gates = import_module(
        "tools.dla_topology_objectives_quality_gates"
    )
    _dla_topology_parity_quality_gates = import_module("tools.dla_topology_parity_quality_gates")
    _dla_topology_projection_quality_gates = import_module(
        "tools.dla_topology_projection_quality_gates"
    )
    _dla_topology_schema_quality_gates = import_module("tools.dla_topology_schema_quality_gates")
    _entanglement_sync_evidence_quality_gates = import_module(
        "tools.entanglement_sync_evidence_quality_gates"
    )
    _enaqt_evidence_quality_gates = import_module("tools.enaqt_evidence_quality_gates")
    _bench_cli_quality_gates = import_module("tools.bench_cli_quality_gates")
    _codesign_components_quality_gates = import_module("tools.codesign_components_quality_gates")
    _synchronisation_witness_quality_gates = import_module(
        "tools.synchronisation_witness_quality_gates"
    )
    _governed_route_matrix_quality_gates = import_module(
        "tools.governed_route_matrix_quality_gates"
    )
    _hermetic_reproduction_kit_quality_gates = import_module(
        "tools.hermetic_reproduction_kit_quality_gates"
    )
    _hls_cosimulation_evidence_quality_gates = import_module(
        "tools.hls_cosimulation_evidence_quality_gates"
    )
    _hardware_safe_quality_gates = import_module("tools.hardware_safe_execution_quality_gates")
    _identity_binding_spec_quality_gates = import_module(
        "tools.identity_binding_spec_quality_gates"
    )
    _ssgf_geometry_gradient_quality_gates = import_module(
        "tools.ssgf_geometry_gradient_quality_gates"
    )
    _metamorphic_ad_verification_quality_gates = import_module(
        "tools.metamorphic_ad_verification_quality_gates"
    )
    _kyma_mechanism_product_quality_gates = import_module(
        "tools.kyma_mechanism_benchmark_product_quality_gates"
    )
    _kyma_dynamics_quality_gates = import_module("tools.kyma_dynamics_quality_gates")
    _kyma_v2_dynamics_quality_gates = import_module("tools.kyma_v2_dynamics_quality_gates")
    _layout_method_comparison_quality_gates = import_module(
        "tools.layout_method_comparison_quality_gates"
    )
    _kuramoto_layout_cost_quality_gates = import_module("tools.kuramoto_layout_cost_quality_gates")
    _kuramoto_layout_relaxation_quality_gates = import_module(
        "tools.kuramoto_layout_relaxation_quality_gates"
    )
    _unsuitable_scenario_registry_quality_gates = import_module(
        "tools.unsuitable_scenario_registry_quality_gates"
    )
    _migration_guides_product_quality_gates = import_module(
        "tools.migration_guides_product_quality_gates"
    )
    _ml_dsa_seal_quality_gates = import_module("tools.ml_dsa_seal_quality_gates")
    _multi_hal_quality_gates = import_module("tools.multi_hal_federation_product_quality_gates")
    _neural_operator_cost_model_quality_gates = import_module(
        "tools.neural_operator_cost_model_quality_gates"
    )
    _neural_operator_baseline_product_quality_gates = import_module(
        "tools.neural_operator_baseline_product_quality_gates"
    )
    _differentiable_notebook_curriculum_quality_gates = import_module(
        "tools.differentiable_notebook_curriculum_quality_gates"
    )
    _differentiable_parameter_shift_quality_gates = import_module(
        "tools.differentiable_parameter_shift_quality_gates"
    )
    _differentiable_sparse_derivatives_quality_gates = import_module(
        "tools.differentiable_sparse_derivatives_quality_gates"
    )
    _program_ad_adjoint_quality_gates = import_module("tools.program_ad_adjoint_quality_gates")
    _qpu_compute_types_quality_gates = import_module("tools.qpu_compute_types_quality_gates")
    _open_system_completeness_quality_gates = import_module(
        "tools.open_system_completeness_quality_gates"
    )
    _open_system_objective_quality_gates = import_module(
        "tools.open_system_objective_quality_gates"
    )
    _phase_jax_qnode_quality_gates = import_module("tools.phase_jax_qnode_quality_gates")
    _phase_qnode_product_quality_gates = import_module("tools.phase_qnode_product_quality_gates")
    _phase_trainability_quality_gates = import_module("tools.phase_trainability_quality_gates")
    _polyglot_parity_certificate_quality_gates = import_module(
        "tools.polyglot_parity_certificate_quality_gates"
    )
    _array_indexing_quality_gates = import_module("tools.program_ad_array_indexing_quality_gates")
    _program_ad_fuzz_assurance_quality_gates = import_module(
        "tools.program_ad_fuzz_assurance_quality_gates"
    )
    _program_ad_quality_gates = import_module("tools.program_ad_quality_gates")
    _quantum_sync_oracle_product_quality_gates = import_module(
        "tools.quantum_sync_oracle_product_quality_gates"
    )
    _research_lane_registry_quality_gates = import_module(
        "tools.research_lane_registry_quality_gates"
    )
    _resource_budget_gate_quality_gates = import_module("tools.resource_budget_gate_quality_gates")
    _scorecard_acceptance_engine_quality_gates = import_module(
        "tools.scorecard_acceptance_engine_quality_gates"
    )
    _stable_core_product_quality_gates = import_module("tools.stable_core_product_quality_gates")
    _stochastic_estimators_product_quality_gates = import_module(
        "tools.stochastic_estimators_product_quality_gates"
    )
    _studio_executive_benchmark_quality_gates = import_module(
        "tools.studio_executive_benchmark_quality_gates"
    )
    _studio_executive_differentiate_quality_gates = import_module(
        "tools.studio_executive_differentiate_quality_gates"
    )
    _studio_executive_quality_gates = import_module("tools.studio_executive_product_quality_gates")
    _thermo_readiness_product_quality_gates = import_module(
        "tools.thermo_readiness_product_quality_gates"
    )
    _theory_hook_promotion_quality_gates = import_module(
        "tools.theory_hook_promotion_quality_gates"
    )
    _visualisation_dashboard_product_quality_gates = import_module(
        "tools.visualisation_dashboard_product_quality_gates"
    )

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_CARGO = which("cargo") or "cargo"
_PNPM = which("pnpm") or "pnpm"
_RUNTIME_SOURCE_ROOTS = (ROOT / "src", ROOT / "oscillatools" / "src")
_HELP_FLAGS = frozenset({"-h", "--help"})

STUDIO_PROGRAM_AD_QUALITY_RATCHET = _program_ad_quality_gates.STUDIO_PROGRAM_AD_QUALITY_RATCHET
STUDIO_PROGRAM_AD_COVERAGE_COHORT = _program_ad_quality_gates.STUDIO_PROGRAM_AD_COVERAGE_COHORT
STUDIO_PROGRAM_AD_BROWSER_TESTS = _program_ad_quality_gates.STUDIO_PROGRAM_AD_BROWSER_TESTS
STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE = (
    _program_ad_quality_gates.STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE
)
PHASE_JAX_QNODE_QUALITY_RATCHET = _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_QUALITY_RATCHET
PHASE_JAX_QNODE_COVERAGE_COHORT = _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_COVERAGE_COHORT
PHASE_JAX_QNODE_COVERAGE_DATA_FILE = (
    _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_COVERAGE_DATA_FILE
)

DIFFERENTIABLE_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/differentiable_architecture_map.py",
    "src/scpn_quantum_control/differentiable_claim_ledger.py",
    "src/scpn_quantum_control/differentiable_claim_rendering.py",
    "src/scpn_quantum_control/differentiable_competitive_baselines.py",
    "src/scpn_quantum_control/differentiable_dependency_environment_evidence.py",
    "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
    "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
    "src/scpn_quantum_control/differentiable_external_validation.py",
    "src/scpn_quantum_control/differentiable_finite_difference.py",
    "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
    "src/scpn_quantum_control/differentiable_transform_algebra.py",
    "src/scpn_quantum_control/program_ad_alias_contracts.py",
    "src/scpn_quantum_control/program_ad_registry.py",
    "src/scpn_quantum_control/program_ad_shape_transforms.py",
    "src/scpn_quantum_control/studio/evidence_bundle.py",
    "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
    "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
    "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
    "src/scpn_quantum_control/stable_core_product.py",
    "tests/test_differentiable_external_validation.py",
    "tests/test_differentiable_finite_difference.py",
    "tests/test_differentiable_competitive_baselines.py",
    "tests/test_differentiable_module_hardening_audit.py",
    "tests/test_differentiable_transform_algebra.py",
    "tests/test_program_ad_alias_contracts.py",
    "tests/test_program_ad_registry.py",
    "tests/test_program_ad_shape_transforms.py",
    "tests/test_phase_tensorflow_maintenance.py",
    "tests/test_differentiable_hardening_gate.py",
    "tests/test_stable_core_product.py",
    "tools/differentiable_support_matrix_page.py",
    "tests/test_differentiable_support_matrix_page.py",
    "tools/differentiable_reviewer_evidence_catalog.py",
    "tools/differentiable_reviewer_evidence_page.py",
    "tests/test_differentiable_reviewer_evidence_page.py",
]

REALTIME_RUNTIME_QUALITY_RATCHET = [
    "src/scpn_quantum_control/control/realtime_runtime.py",
    "tests/test_realtime_runtime.py",
    "tests/test_realtime_runtime_branches.py",
]

PHASE_QNODE_AFFINITY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
    "tools/lean_phase_import.py",
    "tools/run_phase_qnode_affinity_benchmark.py",
    "tests/test_phase_qnode_affinity_benchmark.py",
    "tests/test_lean_phase_import.py",
]

PHASE_QNODE_AFFINITY_COVERAGE_COHORT = [
    "tests/test_phase_qnode_affinity_benchmark.py",
    "tests/test_lean_phase_import.py",
]

PHASE_QNODE_VECTOR_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
    "tests/test_phase_qnode_vector_transforms.py",
    "tests/test_phase_qnode_rust_parity.py",
]

MLIR_LEAF_QUALITY_RATCHET = [
    "src/scpn_quantum_control/compiler/mlir_enzyme_audit.py",
    "src/scpn_quantum_control/compiler/mlir_enzyme_evidence.py",
    "src/scpn_quantum_control/compiler/mlir_phase_qnode_runtime.py",
    "src/scpn_quantum_control/compiler/mlir_transform_plan_assembly.py",
    "src/scpn_quantum_control/compiler/mlir_workload_compilation.py",
    "tests/_mlir_native_compilation_test_helpers.py",
    "tests/test_mlir_enzyme_audit.py",
    "tests/test_mlir_enzyme_evidence_docstrings.py",
    "tests/test_mlir_enzyme_evidence_contracts.py",
    "tests/test_mlir_toolchain_probe_hardening.py",
    "tests/test_mlir_phase_qnode_runtime.py",
    "tests/test_phase_qnode_compiler_lowering.py",
    "tests/test_mlir_transform_plan.py",
    "tests/test_mlir_transform_plan_assembly.py",
    "tests/test_mlir_workload_compilation.py",
    "tests/test_mlir_executable_batching_integration.py",
    "tests/test_mlir_native_compilation_integration.py",
    "tests/test_mlir_scalar_native_compilation_integration.py",
    "tests/test_mlir_vector_native_compilation_integration.py",
    "tests/test_mlir_matrix_native_compilation_integration.py",
    "tests/test_mlir_matrix_2x2_native_compilation_integration.py",
    "tests/test_mlir_symmetric_native_compilation_integration.py",
]

MLIR_LEAF_COVERAGE_COHORT = [
    "tests/test_mlir_enzyme_audit.py",
    "tests/test_mlir_enzyme_evidence_contracts.py",
    "tests/test_mlir_toolchain_probe_hardening.py",
    "tests/test_mlir_phase_qnode_runtime.py",
    "tests/test_phase_qnode_compiler_lowering.py",
    "tests/test_mlir_transform_plan.py",
    "tests/test_mlir_transform_plan_assembly.py",
    "tests/test_mlir_workload_compilation.py",
    "tests/test_mlir_executable_batching_integration.py",
    "tests/test_mlir_native_compilation_integration.py",
    "tests/test_mlir_scalar_native_compilation_integration.py",
    "tests/test_mlir_vector_native_compilation_integration.py",
    "tests/test_mlir_matrix_native_compilation_integration.py",
    "tests/test_mlir_matrix_2x2_native_compilation_integration.py",
    "tests/test_mlir_symmetric_native_compilation_integration.py",
]

PHASE_QNODE_VECTOR_COVERAGE_COHORT = [
    "tests/test_phase_qnode_vector_transforms.py",
]

WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/whole_program_trace_values.py",
    "tests/test_whole_program_trace_values.py",
    "tests/test_whole_program_trace_value_operators.py",
    "tests/test_whole_program_trace_value_selection.py",
    "tests/test_whole_program_trace_value_signal.py",
    "tests/test_whole_program_trace_value_linalg.py",
    "tests/test_whole_program_trace_value_shapes.py",
]

WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT = [
    "tests/test_program_ad_adjoint_generation.py",
    "tests/test_program_ad_adjoint_generation_docstrings.py",
    "tests/test_program_ad_alias_contracts.py",
    "tests/test_program_ad_alias_effects.py",
    "tests/test_program_ad_array_indexing_registry.py",
    "tests/test_program_ad_array_indexing_quality.py",
    "tests/test_program_ad_binary_elementwise_registry.py",
    "tests/test_program_ad_broadcast_assembly.py",
    "tests/test_program_ad_cumulative_primitives.py",
    "tests/test_program_ad_cumulative_primitives_docstrings.py",
    "tests/test_program_ad_effect_ir.py",
    "tests/test_program_ad_elementwise_registry.py",
    "tests/test_program_ad_fail_closed_boundaries.py",
    "tests/test_program_ad_finite_difference_gradient_check.py",
    "tests/test_program_ad_finite_difference_stencils.py",
    "tests/test_program_ad_interpolation.py",
    "tests/test_program_ad_interpolation_primitives_docstrings.py",
    "tests/test_program_ad_like_constructors.py",
    "tests/test_program_ad_linalg_core.py",
    "tests/test_program_ad_linalg_direct_rules.py",
    "tests/test_program_ad_linalg_matrix_ops.py",
    "tests/test_program_ad_linalg_registry.py",
    "tests/test_program_ad_linalg_spectral.py",
    "tests/test_program_ad_product_contractions.py",
    "tests/test_program_ad_reduction_norms.py",
    "tests/test_program_ad_reduction_primitives_docstrings.py",
    "tests/test_program_ad_registry.py",
    "tests/test_program_ad_runtime_registry_dispatch.py",
    "tests/test_program_ad_selection_direct_rules.py",
    "tests/test_program_ad_selection_folds.py",
    "tests/test_program_ad_selection_order_statistics.py",
    "tests/test_program_ad_selection_primitives_docstrings.py",
    "tests/test_program_ad_selection_registry.py",
    "tests/test_program_ad_shape_transforms.py",
    "tests/test_program_ad_signal_primitives.py",
    "tests/test_program_ad_split_assembly.py",
    "tests/test_program_ad_stack_block_assembly.py",
    "tests/test_program_ad_static_array_assembly.py",
    "tests/test_program_ad_stencil_primitives_docstrings.py",
    "tests/test_program_ad_structural_finite_difference_gradient_check.py",
    "tests/test_program_ad_trapezoid.py",
    "tests/test_program_ad_triangular_diagonal_assembly.py",
    "tests/test_program_ad_unary_ufuncs.py",
    "tests/test_program_adjoint_replay.py",
    "tests/test_whole_program_ad_contracts.py",
    "tests/test_whole_program_ad_finite_difference_gradient_check.py",
    "tests/test_whole_program_ad_numpy_structural.py",
    "tests/test_whole_program_ad_runtime.py",
    "tests/test_whole_program_frontend.py",
    "tests/test_whole_program_frontend_contracts.py",
    "tests/test_whole_program_trace_metadata.py",
    "tests/test_whole_program_trace_predicates.py",
    "tests/test_whole_program_trace_runtime.py",
    "tests/test_whole_program_trace_value_linalg.py",
    "tests/test_whole_program_trace_value_operators.py",
    "tests/test_whole_program_trace_value_selection.py",
    "tests/test_whole_program_trace_value_shapes.py",
    "tests/test_whole_program_trace_value_signal.py",
    "tests/test_whole_program_trace_values.py",
]

MLIR_LEAF_COVERAGE_DATA_FILE = ".coverage.mlir-leaf-quality"
MLIR_LEAF_COVERAGE_SOURCE = "src/scpn_quantum_control/compiler"
MLIR_LEAF_COVERAGE_INCLUDE = (
    "*/mlir_enzyme_audit.py,*/mlir_enzyme_evidence.py,*/mlir_phase_qnode_runtime.py,"
    "*/mlir_transform_plan_assembly.py,*/mlir_workload_compilation.py"
)
PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE = ".coverage.phase-qnode-affinity"
PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE = ".coverage.phase-qnode-vector"
WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE = ".coverage.whole-program-trace-values"

_PYTEST_BASE = [
    _PY,
    "-m",
    "pytest",
    "tests/",
    "-x",
    "--tb=short",
    "-q",
    "--ignore=tests/test_hardware_runner.py",
    "--ignore=tests/test_dynamical_lie_algebra.py",  # DLA: 27 min/test, skip for pre-push
]

_PYTEST_COV = _PYTEST_BASE + [
    "--cov=src/scpn_quantum_control",
    "--cov-branch",
    "--cov-fail-under=70",  # temporary combined local smoke guard; CI separately gates lines
]

STATIC_GATES: list[tuple[str, list[str]]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"]),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"]),
    (
        "documentation-surface",
        [
            _PY,
            "tools/audit_documentation_surface.py",
            "--allowlist",
            "tools/documentation_surface_allowlist.json",
            "--fail-on-findings",
        ],
    ),
    (
        "differentiable-promotion-language",
        [_PY, "tools/check_differentiable_promotion_language.py"],
    ),
    (
        "differentiable-competitive-baselines",
        [_PY, "tools/check_differentiable_competitive_baselines.py"],
    ),
    (
        "differentiable-transform-algebra",
        [_PY, "tools/check_differentiable_transform_algebra.py"],
    ),
    (
        "differentiable-support-matrix-page",
        [_PY, "tools/differentiable_support_matrix_page.py", "--check"],
    ),
    (
        "mypy-strict-differentiable-support-matrix-page",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            "tools/differentiable_support_matrix_page.py",
            "tests/test_differentiable_support_matrix_page.py",
        ],
    ),
    (
        "differentiable-reviewer-evidence-page",
        [_PY, "tools/differentiable_reviewer_evidence_page.py", "--check"],
    ),
    (
        "mypy-strict-differentiable-reviewer-evidence-page",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            "tools/differentiable_reviewer_evidence_catalog.py",
            "tools/differentiable_reviewer_evidence_page.py",
            "tests/test_differentiable_reviewer_evidence_page.py",
        ],
    ),
    (
        "ruff D differentiable module-hardening ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *DIFFERENTIABLE_DOCSTRING_RATCHET,
        ],
    ),
    *_decisive_advantage_quality_gates.build_static_quality_gates(_PY),
    *_differentiable_quality_gates.build_static_quality_gates(_PY),
    *_array_indexing_quality_gates.build_static_quality_gates(_PY),
    *_stable_core_product_quality_gates.build_static_quality_gates(_PY),
    *_open_system_completeness_quality_gates.build_static_quality_gates(_PY),
    *_thermo_readiness_product_quality_gates.build_static_quality_gates(_PY),
    *_quantum_sync_oracle_product_quality_gates.build_static_quality_gates(_PY),
    *_custom_derivatives_product_quality_gates.build_static_quality_gates(_PY),
    *_kyma_mechanism_product_quality_gates.build_static_quality_gates(_PY),
    *_kyma_dynamics_quality_gates.build_static_quality_gates(_PY),
    *_kyma_v2_dynamics_quality_gates.build_static_quality_gates(_PY),
    *_differentiable_audit_contracts_quality_gates.build_static_quality_gates(_PY),
    *_campaign_harness_product_quality_gates.build_static_quality_gates(_PY),
    *_chimera_control_quality_gates.build_static_quality_gates(_PY),
    *_whole_program_frontend_contracts_quality_gates.build_static_quality_gates(_PY),
    *_variational_metric_quality_gates.build_static_quality_gates(_PY),
    *_qnode_circuit_contracts_quality_gates.build_static_quality_gates(_PY),
    *_topology_kernel_schema_quality_gates.build_static_quality_gates(_PY),
    *_topology_kernel_evidence_quality_gates.build_static_quality_gates(_PY),
    *_ml_dsa_seal_quality_gates.build_static_quality_gates(_PY),
    *_feedback_loop_quality_gates.build_static_quality_gates(_PY),
    *_hardware_hal_quality_gates.build_static_quality_gates(_PY),
    *_dla_topology_optimizer_quality_gates.build_static_quality_gates(_PY),
    *_dla_topology_schema_quality_gates.build_static_quality_gates(_PY),
    *_fusion_core_frc_bridge_quality_gates.build_static_quality_gates(_PY),
    *_hardware_experiment_vqe_quality_gates.build_static_quality_gates(_PY),
    *_transform_support_matrix_artifact_quality_gates.build_static_quality_gates(_PY),
    *_gradient_plan_explanation_artifact_quality_gates.build_static_quality_gates(_PY),
    *_finite_size_scaling_quality_gates.build_static_quality_gates(_PY),
    *_studio_executive_differentiate_quality_gates.build_static_quality_gates(_PY),
    *_hardware_experiment_control_quality_gates.build_static_quality_gates(_PY),
    *_pulse_shaping_quality_gates.build_static_quality_gates(_PY),
    *_openpulse_control_quality_gates.build_static_quality_gates(_PY),
    *_differentiable_parameter_shift_quality_gates.build_static_quality_gates(_PY),
    *_differentiable_sparse_derivatives_quality_gates.build_static_quality_gates(_PY),
    *_program_ad_adjoint_quality_gates.build_static_quality_gates(_PY),
    *_qpu_compute_types_quality_gates.build_static_quality_gates(_PY),
    *_hls_cosimulation_evidence_quality_gates.build_static_quality_gates(_PY),
    *_dla_topology_objectives_quality_gates.build_static_quality_gates(_PY),
    *_dla_topology_parity_quality_gates.build_static_quality_gates(_PY),
    *_dla_topology_projection_quality_gates.build_static_quality_gates(_PY),
    *_geometric_control_product_quality_gates.build_static_quality_gates(_PY),
    *_tn_mps_baseline_design_quality_gates.build_static_quality_gates(_PY),
    *_error_mitigation_product_quality_gates.build_static_quality_gates(_PY),
    *_cloud_native_deployment_quality_gates.build_static_quality_gates(_PY),
    *_control_stack_compose_quality_gates.build_static_quality_gates(_PY),
    *_public_api_stability_quality_gates.build_static_quality_gates(_PY),
    *_polyglot_parity_certificate_quality_gates.build_static_quality_gates(_PY),
    *_program_ad_fuzz_assurance_quality_gates.build_static_quality_gates(_PY),
    *_multi_hal_quality_gates.build_static_quality_gates(_PY),
    *_hermetic_reproduction_kit_quality_gates.build_static_quality_gates(_PY),
    *_hardware_safe_quality_gates.build_static_quality_gates(_PY),
    *_qpu_compute_product_quality_gates.build_static_quality_gates(_PY),
    *_wirtinger_implicit_product_quality_gates.build_static_quality_gates(_PY),
    *_migration_guides_product_quality_gates.build_static_quality_gates(_PY),
    *_visualisation_dashboard_product_quality_gates.build_static_quality_gates(_PY),
    *_stochastic_estimators_product_quality_gates.build_static_quality_gates(_PY),
    *_differentiable_notebook_curriculum_quality_gates.build_static_quality_gates(_PY),
    *_studio_executive_quality_gates.build_static_quality_gates(_PY),
    *_advanced_witnesses_quality_gates.build_static_quality_gates(_PY),
    *_coverage_frontier_quality_gates.build_static_quality_gates(_PY),
    *_compiler_boundary_product_quality_gates.build_static_quality_gates(_PY),
    *_competitive_baseline_watch_quality_gates.build_static_quality_gates(_PY),
    *_whole_program_ad_product_quality_gates.build_static_quality_gates(_PY),
    *_neural_operator_baseline_product_quality_gates.build_static_quality_gates(_PY),
    *_entanglement_sync_evidence_quality_gates.build_static_quality_gates(_PY),
    *_enaqt_evidence_quality_gates.build_static_quality_gates(_PY),
    *_diff_contract_audit_quality_gates.build_static_quality_gates(_PY),
    *_coupling_recovery_quality_gates.build_static_quality_gates(_PY),
    *_bench_cli_quality_gates.build_static_quality_gates(_PY),
    *_identity_binding_spec_quality_gates.build_static_quality_gates(_PY),
    *_ssgf_geometry_gradient_quality_gates.build_static_quality_gates(_PY),
    *_codesign_components_quality_gates.build_static_quality_gates(_PY),
    *_neural_operator_cost_model_quality_gates.build_static_quality_gates(_PY),
    *_governed_route_matrix_quality_gates.build_static_quality_gates(_PY),
    *_attested_result_pack_quality_gates.build_static_quality_gates(_PY),
    *_open_system_objective_quality_gates.build_static_quality_gates(_PY),
    *_external_validation_quality_gates.build_static_quality_gates(_PY),
    *_gradient_tape_quality_gates.build_static_quality_gates(_PY),
    *_fault_tolerant_resource_product_quality_gates.build_static_quality_gates(_PY),
    *_phase_qnode_product_quality_gates.build_static_quality_gates(_PY),
    *_phase_trainability_quality_gates.build_static_quality_gates(_PY),
    *_kuramoto_layout_cost_quality_gates.build_static_quality_gates(_PY),
    *_kuramoto_layout_relaxation_quality_gates.build_static_quality_gates(_PY),
    *_application_honesty_quality_gates.build_static_quality_gates(_PY),
    *_layout_method_comparison_quality_gates.build_static_quality_gates(_PY),
    *_unsuitable_scenario_registry_quality_gates.build_static_quality_gates(_PY),
    *_scorecard_acceptance_engine_quality_gates.build_static_quality_gates(_PY),
    *_studio_executive_benchmark_quality_gates.build_static_quality_gates(_PY),
    *_synchronisation_witness_quality_gates.build_static_quality_gates(_PY),
    *_experiment_mitigation_quality_gates.build_static_quality_gates(_PY),
    *_research_lane_registry_quality_gates.build_static_quality_gates(_PY),
    *_theory_hook_promotion_quality_gates.build_static_quality_gates(_PY),
    *_resource_budget_gate_quality_gates.build_static_quality_gates(_PY),
    *_advantage_language_protocol_quality_gates.build_static_quality_gates(_PY),
    *_metamorphic_ad_verification_quality_gates.build_static_quality_gates(_PY),
    (
        "mypy-strict-realtime-runtime",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *REALTIME_RUNTIME_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D realtime-runtime quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *REALTIME_RUNTIME_QUALITY_RATCHET,
        ],
    ),
    (
        "mypy-strict-phase-qnode-affinity",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *PHASE_QNODE_AFFINITY_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D phase-qnode-affinity quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *PHASE_QNODE_AFFINITY_QUALITY_RATCHET,
        ],
    ),
    *_program_ad_quality_gates.build_static_quality_gates(_PY),
    (
        "mypy-strict-mlir-leaf-quality",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            *MLIR_LEAF_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D MLIR-leaf quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *MLIR_LEAF_QUALITY_RATCHET,
        ],
    ),
    (
        "mypy-strict-phase-qnode-vector",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *PHASE_QNODE_VECTOR_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D phase-qnode-vector quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *PHASE_QNODE_VECTOR_QUALITY_RATCHET,
        ],
    ),
    *_phase_jax_qnode_quality_gates.build_static_quality_gates(_PY),
    (
        "mypy-strict-whole-program-trace-values",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D whole-program trace-value quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET,
        ],
    ),
    ("test-quality", [_PY, "tools/audit_test_quality.py"]),
    ("module-size-policy", [_PY, "tools/audit_module_size_policy.py"]),
    (
        "mypy-strict-module-size-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_module_size_policy.py"],
    ),
    (
        "descriptive-production-naming",
        [_PY, "tools/audit_descriptive_production_naming.py"],
    ),
    (
        "mypy-strict-descriptive-production-naming",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "tools/audit_descriptive_production_naming.py",
            "tests/test_audit_descriptive_production_naming.py",
        ],
    ),
    ("licence-readiness", [_PY, "tools/audit_license_readiness.py"]),
    (
        "mypy-strict-licence-readiness",
        [_PY, "-m", "mypy", "--strict", "tools/audit_license_readiness.py"],
    ),
    ("test-typing-policy", [_PY, "tools/audit_test_typing_policy.py"]),
    (
        "mypy-strict-test-typing-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_test_typing_policy.py"],
    ),
    (
        "coverage-policy",
        [_PY, "tools/audit_coverage_policy.py", "--validate-policy"],
    ),
    (
        "mypy-strict-coverage-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_policy.py"],
    ),
    ("coverage-debt", [_PY, "tools/audit_coverage_debt.py"]),
    (
        "mypy-strict-coverage-debt",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_debt.py"],
    ),
    (
        "differentiable-external-validation",
        [_PY, "tools/check_differentiable_external_validation.py"],
    ),
    (
        "mypy-strict-differentiable-external-validation",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "tools/check_differentiable_external_validation.py",
        ],
    ),
    (
        "rustfmt",
        [
            _CARGO,
            "fmt",
            "--manifest-path",
            "scpn_quantum_engine/Cargo.toml",
            "--all",
            "--",
            "--check",
        ],
    ),
    ("version-sync", [_PY, "scripts/check_version_consistency.py"]),
    ("rust-pyi", [_PY, "tools/check_rust_pyi_exports.py"]),
    ("mypy", [_PY, "-m", "mypy"]),
    (
        "mypy-strict-differentiable",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "src/scpn_quantum_control/differentiable.py",
            "src/scpn_quantum_control/differentiable_claim_ledger.py",
            "src/scpn_quantum_control/differentiable_architecture_map.py",
            "src/scpn_quantum_control/differentiable_competitive_baselines.py",
            "src/scpn_quantum_control/diff.py",
            "src/scpn/diff.py",
            "src/scpn/__init__.py",
            "src/scpn_quantum_control/differentiable_dependency_environment_evidence.py",
            "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
            "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
            "src/scpn_quantum_control/differentiable_api.py",
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            "src/scpn_quantum_control/differentiable_external_validation.py",
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
            "src/scpn_quantum_control/differentiable_transform_algebra.py",
            "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            "src/scpn_quantum_control/benchmarks/differentiable_evidence.py",
            "src/scpn_quantum_control/phase/differentiable_readiness.py",
            "src/scpn_quantum_control/phase/differentiable_audit.py",
            "src/scpn_quantum_control/phase/gradient_support_matrix.py",
            "src/scpn_quantum_control/phase/provider_gradient.py",
            "src/scpn_quantum_control/phase/hardware_gradient_policy.py",
            "src/scpn_quantum_control/phase/provider_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_publication.py",
            "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
            "src/scpn_quantum_control/phase/gradient_backend.py",
            "src/scpn_quantum_control/phase/gradient_tape.py",
            "src/scpn_quantum_control/phase/natural_gradient.py",
            "src/scpn_quantum_control/phase/gradient_descent.py",
            "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
            "src/scpn_quantum_control/phase/qnode_tape.py",
            "src/scpn_quantum_control/phase/qnode_provider_transforms.py",
            "src/scpn_quantum_control/phase/qnode_transforms.py",
            "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
            "src/scpn_quantum_control/phase/qnode_framework_parity.py",
            "src/scpn_quantum_control/phase/qnode_circuit_builders.py",
            "src/scpn_quantum_control/phase/qnode_circuit.py",
            "src/scpn_quantum_control/phase/qnode_circuit_contracts.py",
            "src/scpn_quantum_control/phase/qnode_circuit_differentiation.py",
            "src/scpn_quantum_control/phase/qnode_circuit_execution.py",
            "src/scpn_quantum_control/phase/qnode_circuit_support.py",
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            "src/scpn_quantum_control/phase/pennylane_provider_plugin.py",
            "src/scpn_quantum_control/phase/jax_bridge.py",
            "src/scpn_quantum_control/phase/jax_bridge_contracts.py",
            "src/scpn_quantum_control/phase/jax_compatibility.py",
            "src/scpn_quantum_control/phase/jax_gradients.py",
            "src/scpn_quantum_control/phase/jax_maturity.py",
            "src/scpn_quantum_control/phase/jax_qnode_transforms.py",
            "src/scpn_quantum_control/phase/torch_bridge.py",
            "src/scpn_quantum_control/phase/torch_bridge_contracts.py",
            "src/scpn_quantum_control/phase/torch_compatibility.py",
            "src/scpn_quantum_control/phase/torch_gradients.py",
            "src/scpn_quantum_control/phase/torch_maturity.py",
            "src/scpn_quantum_control/phase/torch_qnode_transforms.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge_contracts.py",
            "src/scpn_quantum_control/phase/tensorflow_compatibility.py",
            "src/scpn_quantum_control/phase/tensorflow_gradients.py",
            "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            "src/scpn_quantum_control/phase/qiskit_bridge_contracts.py",
            "src/scpn_quantum_control/phase/qiskit_gradients.py",
            "src/scpn_quantum_control/phase/qiskit_runtime.py",
            "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py",
            "src/scpn_quantum_control/phase/transform_nesting.py",
            "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",
            "src/scpn_quantum_control/phase/xy_compiler.py",
            "src/scpn_quantum_control/phase/pennylane_import.py",
            "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py",
            "src/scpn_quantum_control/phase/qnn_training.py",
            "src/scpn_quantum_control/phase/qnn_conformance.py",
            "src/scpn_quantum_control/phase/qnn_finite_shot.py",
            "src/scpn_quantum_control/phase/qnn_convergence.py",
            "src/scpn_quantum_control/phase/qnn_loss_landscape.py",
            "src/scpn_quantum_control/phase/qgnn.py",
            "src/scpn_quantum_control/phase/qnn_framework_agreement.py",
            "src/scpn_quantum_control/phase/model_training_evidence.py",
            "src/scpn_quantum_control/phase/domain_benchmark_datasets.py",
            "src/scpn_quantum_control/phase/objectives.py",
            "src/scpn_quantum_control/phase/objective_planner.py",
            "src/scpn_quantum_control/phase/objective_audit.py",
            "src/scpn_quantum_control/phase/optimizer_audit.py",
            "src/scpn_quantum_control/phase/param_shift.py",
            "src/scpn_quantum_control/phase/general_unitary.py",
            "src/scpn_quantum_control/phase/phase_vqe.py",
            "src/scpn_quantum_control/phase/structured_ansatz.py",
            "src/scpn_quantum_control/phase/xy_kuramoto.py",
            "src/scpn_quantum_control/phase/kuramoto_variants.py",
            "src/scpn_quantum_control/phase/adapt_vqe.py",
            "src/scpn_quantum_control/phase/trotter_error.py",
            "src/scpn_quantum_control/phase/ansatz_methodology.py",
            "src/scpn_quantum_control/phase/results.py",
            "src/scpn_quantum_control/phase/provider_hardware_safety_audit.py",
            "src/scpn_quantum_control/phase/backend_selector.py",
            "src/scpn_quantum_control/phase/ansatz_bench.py",
            "src/scpn_quantum_control/phase/trotter_upde.py",
            "src/scpn_quantum_control/phase/adiabatic_preparation.py",
            "src/scpn_quantum_control/phase/ancilla_lindblad.py",
            "src/scpn_quantum_control/phase/avqds.py",
            "src/scpn_quantum_control/phase/varqite.py",
            "src/scpn_quantum_control/phase/variational_metric.py",
            "src/scpn_quantum_control/phase/coupling_learning.py",
            "src/scpn_quantum_control/phase/contraction_optimiser.py",
            "src/scpn_quantum_control/phase/cross_domain_transfer.py",
            "src/scpn_quantum_control/phase/floquet_kuramoto.py",
        ],
    ),
]

MLIR_LEAF_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "MLIR leaf focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={MLIR_LEAF_COVERAGE_DATA_FILE}",
            "--branch",
            f"--source={MLIR_LEAF_COVERAGE_SOURCE}",
            "-m",
            "pytest",
            "-q",
            *MLIR_LEAF_COVERAGE_COHORT,
        ],
    ),
    (
        "MLIR leaf exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={MLIR_LEAF_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            f"--include={MLIR_LEAF_COVERAGE_INCLUDE}",
        ],
    ),
]

DECISIVE_ADVANTAGE_COVERAGE_GATES = _decisive_advantage_quality_gates.build_coverage_gates(_PY)
DIFFERENTIABLE_QUALITY_COVERAGE_GATES = _differentiable_quality_gates.build_coverage_gates(_PY)
PROGRAM_AD_ARRAY_INDEXING_COVERAGE_GATES = _array_indexing_quality_gates.build_coverage_gates(_PY)
STABLE_CORE_PRODUCT_COVERAGE_GATES = _stable_core_product_quality_gates.build_coverage_gates(_PY)
OPEN_SYSTEM_COMPLETENESS_COVERAGE_GATES = (
    _open_system_completeness_quality_gates.build_coverage_gates(_PY)
)
THERMO_READINESS_PRODUCT_COVERAGE_GATES = (
    _thermo_readiness_product_quality_gates.build_coverage_gates(_PY)
)
QUANTUM_SYNC_ORACLE_COVERAGE_GATES = (
    _quantum_sync_oracle_product_quality_gates.build_coverage_gates(_PY)
)
CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_GATES = (
    _custom_derivatives_product_quality_gates.build_coverage_gates(_PY)
)
KYMA_MECHANISM_PRODUCT_COVERAGE_GATES = _kyma_mechanism_product_quality_gates.build_coverage_gates(
    _PY
)
KYMA_DYNAMICS_COVERAGE_GATES = _kyma_dynamics_quality_gates.build_coverage_gates(_PY)
KYMA_V2_DYNAMICS_COVERAGE_GATES = _kyma_v2_dynamics_quality_gates.build_coverage_gates(_PY)
DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_GATES = (
    _differentiable_audit_contracts_quality_gates.build_coverage_gates(_PY)
)
CAMPAIGN_HARNESS_PRODUCT_COVERAGE_GATES = (
    _campaign_harness_product_quality_gates.build_coverage_gates(_PY)
)
CHIMERA_CONTROL_COVERAGE_GATES = _chimera_control_quality_gates.build_coverage_gates(_PY)
WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_GATES = (
    _whole_program_frontend_contracts_quality_gates.build_coverage_gates(_PY)
)
VARIATIONAL_METRIC_COVERAGE_GATES = _variational_metric_quality_gates.build_coverage_gates(_PY)
QNODE_CIRCUIT_CONTRACTS_COVERAGE_GATES = (
    _qnode_circuit_contracts_quality_gates.build_coverage_gates(_PY)
)
TOPOLOGY_KERNEL_SCHEMA_COVERAGE_GATES = _topology_kernel_schema_quality_gates.build_coverage_gates(
    _PY
)
TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_GATES = (
    _topology_kernel_evidence_quality_gates.build_coverage_gates(_PY)
)
ML_DSA_SEAL_COVERAGE_GATES = _ml_dsa_seal_quality_gates.build_coverage_gates(_PY)
FEEDBACK_LOOP_COVERAGE_GATES = _feedback_loop_quality_gates.build_coverage_gates(_PY)
HARDWARE_HAL_COVERAGE_GATES = _hardware_hal_quality_gates.build_coverage_gates(_PY)
DLA_TOPOLOGY_OPTIMIZER_COVERAGE_GATES = _dla_topology_optimizer_quality_gates.build_coverage_gates(
    _PY
)
DLA_TOPOLOGY_SCHEMA_COVERAGE_GATES = _dla_topology_schema_quality_gates.build_coverage_gates(_PY)
FUSION_CORE_FRC_BRIDGE_COVERAGE_GATES = _fusion_core_frc_bridge_quality_gates.build_coverage_gates(
    _PY
)
HARDWARE_EXPERIMENT_VQE_COVERAGE_GATES = (
    _hardware_experiment_vqe_quality_gates.build_coverage_gates(_PY)
)
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_GATES = (
    _transform_support_matrix_artifact_quality_gates.build_coverage_gates(_PY)
)
GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_GATES = (
    _gradient_plan_explanation_artifact_quality_gates.build_coverage_gates(_PY)
)
FINITE_SIZE_SCALING_COVERAGE_GATES = _finite_size_scaling_quality_gates.build_coverage_gates(_PY)
STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_GATES = (
    _studio_executive_differentiate_quality_gates.build_coverage_gates(_PY)
)
HARDWARE_EXPERIMENT_CONTROL_COVERAGE_GATES = (
    _hardware_experiment_control_quality_gates.build_coverage_gates(_PY)
)
PULSE_SHAPING_COVERAGE_GATES = _pulse_shaping_quality_gates.build_coverage_gates(_PY)
OPENPULSE_CONTROL_COVERAGE_GATES = _openpulse_control_quality_gates.build_coverage_gates(_PY)
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_GATES = (
    _differentiable_parameter_shift_quality_gates.build_coverage_gates(_PY)
)
DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_GATES = (
    _differentiable_sparse_derivatives_quality_gates.build_coverage_gates(_PY)
)
PROGRAM_AD_ADJOINT_COVERAGE_GATES = _program_ad_adjoint_quality_gates.build_coverage_gates(_PY)
QPU_COMPUTE_TYPES_COVERAGE_GATES = _qpu_compute_types_quality_gates.build_coverage_gates(_PY)
HLS_COSIMULATION_EVIDENCE_COVERAGE_GATES = (
    _hls_cosimulation_evidence_quality_gates.build_coverage_gates(_PY)
)
DLA_TOPOLOGY_OBJECTIVES_COVERAGE_GATES = (
    _dla_topology_objectives_quality_gates.build_coverage_gates(_PY)
)
DLA_TOPOLOGY_PARITY_COVERAGE_GATES = _dla_topology_parity_quality_gates.build_coverage_gates(_PY)
DLA_TOPOLOGY_PROJECTION_COVERAGE_GATES = (
    _dla_topology_projection_quality_gates.build_coverage_gates(_PY)
)
GEOMETRIC_CONTROL_PRODUCT_COVERAGE_GATES = (
    _geometric_control_product_quality_gates.build_coverage_gates(_PY)
)
TN_MPS_BASELINE_DESIGN_COVERAGE_GATES = _tn_mps_baseline_design_quality_gates.build_coverage_gates(
    _PY
)
ERROR_MITIGATION_PRODUCT_COVERAGE_GATES = (
    _error_mitigation_product_quality_gates.build_coverage_gates(_PY)
)
CLOUD_NATIVE_DEPLOYMENT_COVERAGE_GATES = (
    _cloud_native_deployment_quality_gates.build_coverage_gates(_PY)
)
CONTROL_STACK_COMPOSE_COVERAGE_GATES = _control_stack_compose_quality_gates.build_coverage_gates(
    _PY
)
PUBLIC_API_STABILITY_COVERAGE_GATES = _public_api_stability_quality_gates.build_coverage_gates(_PY)
POLYGLOT_PARITY_CERTIFICATE_COVERAGE_GATES = (
    _polyglot_parity_certificate_quality_gates.build_coverage_gates(_PY)
)
PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_GATES = (
    _program_ad_fuzz_assurance_quality_gates.build_coverage_gates(_PY)
)
MULTI_HAL_FEDERATION_COVERAGE_GATES = _multi_hal_quality_gates.build_coverage_gates(_PY)
HERMETIC_REPRODUCTION_KIT_COVERAGE_GATES = (
    _hermetic_reproduction_kit_quality_gates.build_coverage_gates(_PY)
)
HARDWARE_SAFE_EXECUTION_COVERAGE_GATES = _hardware_safe_quality_gates.build_coverage_gates(_PY)
QPU_COMPUTE_PRODUCT_COVERAGE_GATES = _qpu_compute_product_quality_gates.build_coverage_gates(_PY)
WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_GATES = (
    _wirtinger_implicit_product_quality_gates.build_coverage_gates(_PY)
)
MIGRATION_GUIDES_PRODUCT_COVERAGE_GATES = (
    _migration_guides_product_quality_gates.build_coverage_gates(_PY)
)
VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_GATES = (
    _visualisation_dashboard_product_quality_gates.build_coverage_gates(_PY)
)
STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_GATES = (
    _stochastic_estimators_product_quality_gates.build_coverage_gates(_PY)
)
DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_GATES = (
    _differentiable_notebook_curriculum_quality_gates.build_coverage_gates(_PY)
)
STUDIO_EXECUTIVE_PRODUCT_COVERAGE_GATES = _studio_executive_quality_gates.build_coverage_gates(_PY)
ADVANCED_WITNESSES_COVERAGE_GATES = _advanced_witnesses_quality_gates.build_coverage_gates(_PY)
COVERAGE_FRONTIER_COVERAGE_GATES = _coverage_frontier_quality_gates.build_coverage_gates(_PY)
COMPILER_BOUNDARY_PRODUCT_COVERAGE_GATES = (
    _compiler_boundary_product_quality_gates.build_coverage_gates(_PY)
)
COMPETITIVE_BASELINE_WATCH_COVERAGE_GATES = (
    _competitive_baseline_watch_quality_gates.build_coverage_gates(_PY)
)
WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_GATES = (
    _whole_program_ad_product_quality_gates.build_coverage_gates(_PY)
)
NEURAL_OPERATOR_COST_MODEL_COVERAGE_GATES = (
    _neural_operator_cost_model_quality_gates.build_coverage_gates(_PY)
)
NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_GATES = (
    _neural_operator_baseline_product_quality_gates.build_coverage_gates(_PY)
)
ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_GATES = (
    _entanglement_sync_evidence_quality_gates.build_coverage_gates(_PY)
)
ENAQT_EVIDENCE_COVERAGE_GATES = _enaqt_evidence_quality_gates.build_coverage_gates(_PY)
DIFF_CONTRACT_AUDIT_COVERAGE_GATES = _diff_contract_audit_quality_gates.build_coverage_gates(_PY)
COUPLING_RECOVERY_COVERAGE_GATES = _coupling_recovery_quality_gates.build_coverage_gates(_PY)
BENCH_CLI_COVERAGE_GATES = _bench_cli_quality_gates.build_coverage_gates(_PY)
IDENTITY_BINDING_SPEC_COVERAGE_GATES = _identity_binding_spec_quality_gates.build_coverage_gates(
    _PY
)
SSGF_GEOMETRY_GRADIENT_COVERAGE_GATES = _ssgf_geometry_gradient_quality_gates.build_coverage_gates(
    _PY
)
CODESIGN_COMPONENTS_COVERAGE_GATES = _codesign_components_quality_gates.build_coverage_gates(_PY)
GOVERNED_ROUTE_MATRIX_COVERAGE_GATES = _governed_route_matrix_quality_gates.build_coverage_gates(
    _PY
)
ATTESTED_RESULT_PACK_COVERAGE_GATES = _attested_result_pack_quality_gates.build_coverage_gates(_PY)
OPEN_SYSTEM_OBJECTIVE_COVERAGE_GATES = _open_system_objective_quality_gates.build_coverage_gates(
    _PY
)
EXTERNAL_VALIDATION_COVERAGE_GATES = _external_validation_quality_gates.build_coverage_gates(_PY)
GRADIENT_TAPE_COVERAGE_GATES = _gradient_tape_quality_gates.build_coverage_gates(_PY)
FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_GATES = (
    _fault_tolerant_resource_product_quality_gates.build_coverage_gates(_PY)
)
PHASE_QNODE_PRODUCT_COVERAGE_GATES = _phase_qnode_product_quality_gates.build_coverage_gates(_PY)
PHASE_TRAINABILITY_COVERAGE_GATES = _phase_trainability_quality_gates.build_coverage_gates(_PY)
KURAMOTO_LAYOUT_COST_COVERAGE_GATES = _kuramoto_layout_cost_quality_gates.build_coverage_gates(_PY)
KURAMOTO_LAYOUT_RELAXATION_COVERAGE_GATES = (
    _kuramoto_layout_relaxation_quality_gates.build_coverage_gates(_PY)
)
APPLICATION_HONESTY_COVERAGE_GATES = _application_honesty_quality_gates.build_coverage_gates(_PY)
LAYOUT_METHOD_COMPARISON_COVERAGE_GATES = (
    _layout_method_comparison_quality_gates.build_coverage_gates(_PY)
)
UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_GATES = (
    _unsuitable_scenario_registry_quality_gates.build_coverage_gates(_PY)
)
SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_GATES = (
    _scorecard_acceptance_engine_quality_gates.build_coverage_gates(_PY)
)
STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_GATES = (
    _studio_executive_benchmark_quality_gates.build_coverage_gates(_PY)
)
SYNCHRONISATION_WITNESS_COVERAGE_GATES = (
    _synchronisation_witness_quality_gates.build_coverage_gates(_PY)
)
EXPERIMENT_MITIGATION_COVERAGE_GATES = _experiment_mitigation_quality_gates.build_coverage_gates(
    _PY
)
RESEARCH_LANE_REGISTRY_COVERAGE_GATES = _research_lane_registry_quality_gates.build_coverage_gates(
    _PY
)
THEORY_HOOK_PROMOTION_COVERAGE_GATES = _theory_hook_promotion_quality_gates.build_coverage_gates(
    _PY
)
RESOURCE_BUDGET_GATE_COVERAGE_GATES = _resource_budget_gate_quality_gates.build_coverage_gates(_PY)
ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_GATES = (
    _advantage_language_protocol_quality_gates.build_coverage_gates(_PY)
)
METAMORPHIC_AD_VERIFICATION_COVERAGE_GATES = (
    _metamorphic_ad_verification_quality_gates.build_coverage_gates(_PY)
)

PHASE_QNODE_AFFINITY_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "phase-qnode affinity focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *PHASE_QNODE_AFFINITY_COVERAGE_COHORT,
        ],
    ),
    (
        "phase-qnode affinity exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/qnode_affinity_benchmark.py",
        ],
    ),
]

STUDIO_PROGRAM_AD_COVERAGE_GATES = _program_ad_quality_gates.build_python_coverage_gates(_PY)
STUDIO_PROGRAM_AD_RUNTIME_GATES = _program_ad_quality_gates.build_runtime_gates(
    _CARGO,
    _PNPM,
)
STUDIO_PROGRAM_AD_BROWSER_TEST_GATE = _program_ad_quality_gates.build_browser_test_gate(_PNPM)
STUDIO_PROGRAM_AD_BROWSER_COVERAGE_GATE = _program_ad_quality_gates.build_browser_coverage_gate(
    _PNPM
)

PHASE_QNODE_VECTOR_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "phase-qnode vector focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *PHASE_QNODE_VECTOR_COVERAGE_COHORT,
        ],
    ),
    (
        "phase-qnode vector exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/qnode_vector_transforms.py",
        ],
    ),
]

PHASE_JAX_QNODE_COVERAGE_GATES = _phase_jax_qnode_quality_gates.build_coverage_gates(_PY)

WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "whole-program trace-value focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT,
        ],
    ),
    (
        "whole-program trace-value exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/whole_program_trace_values.py",
        ],
    ),
    (
        "program AD alias-contract exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/program_ad_alias_contracts.py",
        ],
    ),
    (
        "program AD shape-transform exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/program_ad_shape_transforms.py",
        ],
    ),
]

BANDIT_GATE: tuple[str, list[str]] = (
    "bandit",
    [_PY, "-m", "bandit", "-r", "src/", "-ll", "-q"],
)


def _admit_gate_command(cmd: list[str]) -> list[str]:
    """Return a shell-free command with a verified executable path."""
    if not cmd:
        raise ValueError("gate command is empty")
    executable = Path(cmd[0])
    if not executable.is_absolute():
        raise ValueError(f"gate executable is not absolute: {cmd[0]}")
    try:
        exists = executable.exists()
    except (OSError, ValueError) as exc:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}") from exc
    if not exists:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}")
    if not executable.is_file():
        raise ValueError(f"gate executable is not a file: {executable}")
    if not access(executable, X_OK):
        raise ValueError(f"gate executable is not executable: {executable}")
    return [str(executable), *cmd[1:]]


def _deduplicated_path_entries(entries: Iterable[str]) -> list[str]:
    """Return path entries in first-seen order without empty duplicates."""
    seen: set[str] = set()
    deduplicated: list[str] = []
    for entry in entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        deduplicated.append(entry)
    return deduplicated


def _gate_environment() -> dict[str, str]:
    """Return the subprocess environment for preflight gates.

    Tool scripts execute from ``tools/`` but import the repository packages.
    Prepending the source roots keeps local runtime checks aligned with the
    package layout and the explicit mypy path used for the install-free
    ``oscillatools`` sibling source tree.
    """
    env = dict(environ)
    source_roots = [str(path) for path in _RUNTIME_SOURCE_ROOTS if path.is_dir()]
    existing_pythonpath = env.get("PYTHONPATH", "")
    entries = _deduplicated_path_entries([*source_roots, *existing_pythonpath.split(pathsep)])
    if entries:
        env["PYTHONPATH"] = pathsep.join(entries)
    return env


def run_gate(name: str, cmd: list[str]) -> bool:
    """Run a named preflight command and print a compact result summary."""
    t0 = time.monotonic()
    try:
        admitted_cmd = _admit_gate_command(cmd)
    except ValueError as exc:
        elapsed = time.monotonic() - t0
        print(f"  FAIL  {name} ({elapsed:.1f}s)")
        print(f"        {exc}")
        return False
    result = subprocess.run(  # nosec B603
        admitted_cmd,
        cwd=ROOT,
        capture_output=True,
        env=_gate_environment(),
        text=True,
        shell=False,
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        {line}")
    return False


def _wants_help(args: Iterable[str]) -> bool:
    """Return whether the supplied CLI arguments request usage text."""
    return any(arg in _HELP_FLAGS for arg in args)


def main() -> int:
    """Run the configured preflight gate suite."""
    args = sys.argv[1:]
    if _wants_help(args):
        print((__doc__ or "").strip())
        return 0

    skip_tests = "--no-tests" in args
    no_coverage = "--no-coverage" in args

    gates: list[tuple[str, list[str]]] = list(STATIC_GATES)

    if not skip_tests:
        gates.extend(STUDIO_PROGRAM_AD_RUNTIME_GATES)
        if no_coverage:
            gates.append(STUDIO_PROGRAM_AD_BROWSER_TEST_GATE)
            gates.append(("pytest", _PYTEST_BASE))
        else:
            gates.extend(DECISIVE_ADVANTAGE_COVERAGE_GATES)
            gates.extend(DIFFERENTIABLE_QUALITY_COVERAGE_GATES)
            gates.extend(PROGRAM_AD_ARRAY_INDEXING_COVERAGE_GATES)
            gates.extend(STABLE_CORE_PRODUCT_COVERAGE_GATES)
            gates.extend(OPEN_SYSTEM_COMPLETENESS_COVERAGE_GATES)
            gates.extend(THERMO_READINESS_PRODUCT_COVERAGE_GATES)
            gates.extend(QUANTUM_SYNC_ORACLE_COVERAGE_GATES)
            gates.extend(CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_GATES)
            gates.extend(KYMA_MECHANISM_PRODUCT_COVERAGE_GATES)
            gates.extend(KYMA_DYNAMICS_COVERAGE_GATES)
            gates.extend(KYMA_V2_DYNAMICS_COVERAGE_GATES)
            gates.extend(DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_GATES)
            gates.extend(CAMPAIGN_HARNESS_PRODUCT_COVERAGE_GATES)
            gates.extend(CHIMERA_CONTROL_COVERAGE_GATES)
            gates.extend(WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_GATES)
            gates.extend(VARIATIONAL_METRIC_COVERAGE_GATES)
            gates.extend(QNODE_CIRCUIT_CONTRACTS_COVERAGE_GATES)
            gates.extend(TOPOLOGY_KERNEL_SCHEMA_COVERAGE_GATES)
            gates.extend(TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_GATES)
            gates.extend(ML_DSA_SEAL_COVERAGE_GATES)
            gates.extend(FEEDBACK_LOOP_COVERAGE_GATES)
            gates.extend(HARDWARE_HAL_COVERAGE_GATES)
            gates.extend(DLA_TOPOLOGY_OPTIMIZER_COVERAGE_GATES)
            gates.extend(DLA_TOPOLOGY_SCHEMA_COVERAGE_GATES)
            gates.extend(FUSION_CORE_FRC_BRIDGE_COVERAGE_GATES)
            gates.extend(HARDWARE_EXPERIMENT_VQE_COVERAGE_GATES)
            gates.extend(TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_GATES)
            gates.extend(GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_GATES)
            gates.extend(FINITE_SIZE_SCALING_COVERAGE_GATES)
            gates.extend(STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_GATES)
            gates.extend(HARDWARE_EXPERIMENT_CONTROL_COVERAGE_GATES)
            gates.extend(PULSE_SHAPING_COVERAGE_GATES)
            gates.extend(OPENPULSE_CONTROL_COVERAGE_GATES)
            gates.extend(DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_GATES)
            gates.extend(DIFFERENTIABLE_SPARSE_DERIVATIVES_COVERAGE_GATES)
            gates.extend(PROGRAM_AD_ADJOINT_COVERAGE_GATES)
            gates.extend(QPU_COMPUTE_TYPES_COVERAGE_GATES)
            gates.extend(HLS_COSIMULATION_EVIDENCE_COVERAGE_GATES)
            gates.extend(DLA_TOPOLOGY_OBJECTIVES_COVERAGE_GATES)
            gates.extend(DLA_TOPOLOGY_PARITY_COVERAGE_GATES)
            gates.extend(DLA_TOPOLOGY_PROJECTION_COVERAGE_GATES)
            gates.extend(GEOMETRIC_CONTROL_PRODUCT_COVERAGE_GATES)
            gates.extend(TN_MPS_BASELINE_DESIGN_COVERAGE_GATES)
            gates.extend(ERROR_MITIGATION_PRODUCT_COVERAGE_GATES)
            gates.extend(CLOUD_NATIVE_DEPLOYMENT_COVERAGE_GATES)
            gates.extend(CONTROL_STACK_COMPOSE_COVERAGE_GATES)
            gates.extend(PUBLIC_API_STABILITY_COVERAGE_GATES)
            gates.extend(POLYGLOT_PARITY_CERTIFICATE_COVERAGE_GATES)
            gates.extend(PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_GATES)
            gates.extend(MULTI_HAL_FEDERATION_COVERAGE_GATES)
            gates.extend(HERMETIC_REPRODUCTION_KIT_COVERAGE_GATES)
            gates.extend(HARDWARE_SAFE_EXECUTION_COVERAGE_GATES)
            gates.extend(QPU_COMPUTE_PRODUCT_COVERAGE_GATES)
            gates.extend(WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_GATES)
            gates.extend(MIGRATION_GUIDES_PRODUCT_COVERAGE_GATES)
            gates.extend(VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_GATES)
            gates.extend(STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_GATES)
            gates.extend(DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_GATES)
            gates.extend(STUDIO_EXECUTIVE_PRODUCT_COVERAGE_GATES)
            gates.extend(ADVANCED_WITNESSES_COVERAGE_GATES)
            gates.extend(COVERAGE_FRONTIER_COVERAGE_GATES)
            gates.extend(COMPILER_BOUNDARY_PRODUCT_COVERAGE_GATES)
            gates.extend(COMPETITIVE_BASELINE_WATCH_COVERAGE_GATES)
            gates.extend(WHOLE_PROGRAM_AD_PRODUCT_COVERAGE_GATES)
            gates.extend(NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_GATES)
            gates.extend(ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_GATES)
            gates.extend(ENAQT_EVIDENCE_COVERAGE_GATES)
            gates.extend(DIFF_CONTRACT_AUDIT_COVERAGE_GATES)
            gates.extend(COUPLING_RECOVERY_COVERAGE_GATES)
            gates.extend(BENCH_CLI_COVERAGE_GATES)
            gates.extend(IDENTITY_BINDING_SPEC_COVERAGE_GATES)
            gates.extend(SSGF_GEOMETRY_GRADIENT_COVERAGE_GATES)
            gates.extend(CODESIGN_COMPONENTS_COVERAGE_GATES)
            gates.extend(NEURAL_OPERATOR_COST_MODEL_COVERAGE_GATES)
            gates.extend(GOVERNED_ROUTE_MATRIX_COVERAGE_GATES)
            gates.extend(ATTESTED_RESULT_PACK_COVERAGE_GATES)
            gates.extend(OPEN_SYSTEM_OBJECTIVE_COVERAGE_GATES)
            gates.extend(EXTERNAL_VALIDATION_COVERAGE_GATES)
            gates.extend(GRADIENT_TAPE_COVERAGE_GATES)
            gates.extend(FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_GATES)
            gates.extend(PHASE_QNODE_PRODUCT_COVERAGE_GATES)
            gates.extend(PHASE_TRAINABILITY_COVERAGE_GATES)
            gates.extend(KURAMOTO_LAYOUT_COST_COVERAGE_GATES)
            gates.extend(KURAMOTO_LAYOUT_RELAXATION_COVERAGE_GATES)
            gates.extend(APPLICATION_HONESTY_COVERAGE_GATES)
            gates.extend(LAYOUT_METHOD_COMPARISON_COVERAGE_GATES)
            gates.extend(UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_GATES)
            gates.extend(SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_GATES)
            gates.extend(STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_GATES)
            gates.extend(SYNCHRONISATION_WITNESS_COVERAGE_GATES)
            gates.extend(EXPERIMENT_MITIGATION_COVERAGE_GATES)
            gates.extend(RESEARCH_LANE_REGISTRY_COVERAGE_GATES)
            gates.extend(THEORY_HOOK_PROMOTION_COVERAGE_GATES)
            gates.extend(RESOURCE_BUDGET_GATE_COVERAGE_GATES)
            gates.extend(ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_GATES)
            gates.extend(METAMORPHIC_AD_VERIFICATION_COVERAGE_GATES)
            gates.extend(MLIR_LEAF_COVERAGE_GATES)
            gates.extend(PHASE_QNODE_AFFINITY_COVERAGE_GATES)
            gates.extend(STUDIO_PROGRAM_AD_COVERAGE_GATES)
            gates.extend(PHASE_QNODE_VECTOR_COVERAGE_GATES)
            gates.extend(PHASE_JAX_QNODE_COVERAGE_GATES)
            gates.extend(WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES)
            gates.append(STUDIO_PROGRAM_AD_BROWSER_COVERAGE_GATE)
            gates.append(("pytest + coverage", _PYTEST_COV))

    gates.append(BANDIT_GATE)

    print(f"preflight: {len(gates)} gates")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd in gates:
        if not run_gate(name, cmd):
            failed.append(name)
            break

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
