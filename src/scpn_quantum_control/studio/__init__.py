# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio federation surface
"""The QUANTUM studio's federation surface on the SCPN-STUDIO platform contract.

Exposes the schema-A capability manifest (verbs, evidence schemas, content digest)
that the Hub ingests for federation. See :mod:`scpn_quantum_control.studio.manifest`
and :mod:`scpn_quantum_control.studio.verbs`.
"""

from __future__ import annotations

from .benchmark_databank_bundle import (
    DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH,
    build_benchmark_databank_bundle,
)
from .coupling_invariant import (
    COUPLING_INVARIANT_ARTIFACT_ID,
    COUPLING_INVARIANT_ID,
    CouplingInvariantPayload,
    CouplingInvariantSource,
    build_coupling_invariant_bundle,
    build_coupling_invariant_payload,
    validate_coupling_invariant_payload,
)
from .coverage_frontier import (
    ANSWERED_STATUSES,
    CoverageFrontierReport,
    map_claim_status,
    measure_coverage_frontier,
    measure_coverage_frontier_from_certifications,
    render_coverage_frontier_markdown,
)
from .evidence_bundle import (
    StudioBundleValidation,
    build_claim_ledger_bundle,
    build_claim_ledger_bundles,
    build_hardware_result_pack_bundle,
    build_hardware_result_pack_bundles,
    evidence_axes,
    validate_bundle,
    validate_bundles,
)
from .executive import (
    ActionHandler,
    ActionRegistry,
    ActionStatus,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRecord,
    ExecutiveRequest,
    GeneratedScript,
    ScriptLanguage,
    VerbContract,
    build_generated_script,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from .executive_analyse import (
    ANALYSE_CLAIM_BOUNDARY,
    ANALYSE_VERB,
    AnalyseActionHandler,
)
from .executive_benchmark import (
    BENCHMARK_CLAIM_BOUNDARY,
    BENCHMARK_VERB,
    LIVE_TIMING_CAVEAT,
    BenchmarkActionHandler,
    measure_p50_us,
    native_dense_xy_hamiltonian,
    reference_dense_xy_hamiltonian,
)
from .executive_cli import build_default_registry
from .executive_compile import (
    COMPILE_CLAIM_BOUNDARY,
    COMPILE_VERB,
    CompileActionHandler,
)
from .executive_differentiate import (
    DIFFERENTIATE_CLAIM_BOUNDARY,
    DIFFERENTIATE_VERB,
    DifferentiateActionHandler,
    build_effect_ir,
    default_registry,
)
from .executive_execute import (
    EXECUTE_CLAIM_BOUNDARY,
    EXECUTE_VERB,
    ExecuteActionHandler,
)
from .executive_mitigate import (
    MITIGATE_CLAIM_BOUNDARY,
    MITIGATE_VERB,
    MitigateActionHandler,
)
from .executive_replay import (
    REPLAY_CLAIM_BOUNDARY,
    REPLAY_VERB,
    ReplayActionHandler,
)
from .executive_simulate import (
    SIMULATE_CLAIM_BOUNDARY,
    SIMULATE_VERB,
    SimulateActionHandler,
)
from .executive_validate import (
    VALIDATE_CLAIM_BOUNDARY,
    VALIDATE_VERB,
    ValidateActionHandler,
)
from .federation import (
    build_architecture_map_extension,
    build_federation_document,
    write_federation_document,
)
from .manifest import build_manifest, declared_surface
from .qec_readiness_bundle import (
    DEFAULT_QEC_READINESS_ARTIFACT_PATH,
    QEC_READINESS_ARTIFACT_ID,
    build_qec_readiness_bundle,
)
from .qpu_result_pack import (
    QPU_VERIFIABILITY_MODE,
    QpuResultPackPresentation,
    build_qpu_result_pack_unit,
    present_qpu_result_pack,
    seal_qpu_result_pack,
)
from .readout_mitigation_bundle import (
    DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH,
    READOUT_MITIGATION_ARTIFACT_ID,
    build_readout_mitigation_bundle,
)
from .recompute_kernel import (
    XY_COMPILE_INPUT_VERSION,
    XY_COMPILE_RECOMPUTE_SCHEMA,
    XY_COMPILE_WASM_CRATE,
    XY_COMPILE_WASM_EXPORT,
    DecodedXYCompileInput,
    XYCompileRecomputeUnit,
    build_xy_compile_recompute_unit,
    canonical_xy_compile_input_bytes,
    decode_xy_compile_input_bytes,
    verify_xy_compile_recompute_unit,
    xy_compile_digest_python,
)
from .reference_validation import (
    DEFAULT_REFERENCE_VALIDATION_PATH,
    REFERENCE_VALIDATION_SCHEMA,
    ReferenceValidationCertification,
    ReferenceValidationRegistry,
    ReferenceValidationRegistryValidation,
    load_reference_validation_registry,
)
from .result_pack_seal import (
    build_provider_attestation,
    build_result_pack_unit,
    seal_result_pack,
)
from .scorecard_bundle import (
    DEFAULT_SCORECARD_ARTIFACT_PATH,
    build_scorecard_bundle,
)
from .support_matrix_bundle import (
    DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH,
    build_support_matrix_bundle,
)
from .verbs import QUANTUM_VERBS, STUDIO_ID, evidence_schemas, verb_substrates

__all__ = [
    "ANALYSE_CLAIM_BOUNDARY",
    "ANALYSE_VERB",
    "ANSWERED_STATUSES",
    "ActionHandler",
    "AnalyseActionHandler",
    "ActionRegistry",
    "ActionStatus",
    "BENCHMARK_CLAIM_BOUNDARY",
    "BENCHMARK_VERB",
    "BenchmarkActionHandler",
    "COMPILE_CLAIM_BOUNDARY",
    "COMPILE_VERB",
    "CompileActionHandler",
    "DIFFERENTIATE_CLAIM_BOUNDARY",
    "DIFFERENTIATE_VERB",
    "DifferentiateActionHandler",
    "EXECUTE_CLAIM_BOUNDARY",
    "EXECUTE_VERB",
    "ExecuteActionHandler",
    "ExecutionPlan",
    "ExecutionResult",
    "ExecutiveRecord",
    "ExecutiveRequest",
    "GeneratedScript",
    "LIVE_TIMING_CAVEAT",
    "MITIGATE_CLAIM_BOUNDARY",
    "MITIGATE_VERB",
    "MitigateActionHandler",
    "ScriptLanguage",
    "VerbContract",
    "build_default_registry",
    "build_effect_ir",
    "build_generated_script",
    "default_registry",
    "preview_action",
    "resolve_verb_contract",
    "run_action",
    "CoverageFrontierReport",
    "COUPLING_INVARIANT_ARTIFACT_ID",
    "COUPLING_INVARIANT_ID",
    "CouplingInvariantPayload",
    "CouplingInvariantSource",
    "DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH",
    "DEFAULT_QEC_READINESS_ARTIFACT_PATH",
    "DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH",
    "DEFAULT_REFERENCE_VALIDATION_PATH",
    "DEFAULT_SCORECARD_ARTIFACT_PATH",
    "DEFAULT_SUPPORT_MATRIX_ARTIFACT_PATH",
    "QEC_READINESS_ARTIFACT_ID",
    "QPU_VERIFIABILITY_MODE",
    "QUANTUM_VERBS",
    "QpuResultPackPresentation",
    "READOUT_MITIGATION_ARTIFACT_ID",
    "REFERENCE_VALIDATION_SCHEMA",
    "REPLAY_CLAIM_BOUNDARY",
    "REPLAY_VERB",
    "ReferenceValidationCertification",
    "ReferenceValidationRegistry",
    "ReferenceValidationRegistryValidation",
    "ReplayActionHandler",
    "SIMULATE_CLAIM_BOUNDARY",
    "SIMULATE_VERB",
    "STUDIO_ID",
    "SimulateActionHandler",
    "StudioBundleValidation",
    "VALIDATE_CLAIM_BOUNDARY",
    "VALIDATE_VERB",
    "ValidateActionHandler",
    "XY_COMPILE_INPUT_VERSION",
    "XY_COMPILE_RECOMPUTE_SCHEMA",
    "XY_COMPILE_WASM_CRATE",
    "XY_COMPILE_WASM_EXPORT",
    "DecodedXYCompileInput",
    "XYCompileRecomputeUnit",
    "build_architecture_map_extension",
    "build_benchmark_databank_bundle",
    "build_claim_ledger_bundle",
    "build_claim_ledger_bundles",
    "build_coupling_invariant_bundle",
    "build_coupling_invariant_payload",
    "build_federation_document",
    "build_hardware_result_pack_bundle",
    "build_hardware_result_pack_bundles",
    "build_manifest",
    "build_provider_attestation",
    "build_qec_readiness_bundle",
    "build_qpu_result_pack_unit",
    "build_readout_mitigation_bundle",
    "build_result_pack_unit",
    "build_scorecard_bundle",
    "build_support_matrix_bundle",
    "build_xy_compile_recompute_unit",
    "canonical_xy_compile_input_bytes",
    "decode_xy_compile_input_bytes",
    "declared_surface",
    "evidence_schemas",
    "evidence_axes",
    "map_claim_status",
    "measure_coverage_frontier",
    "measure_coverage_frontier_from_certifications",
    "measure_p50_us",
    "native_dense_xy_hamiltonian",
    "present_qpu_result_pack",
    "reference_dense_xy_hamiltonian",
    "render_coverage_frontier_markdown",
    "seal_qpu_result_pack",
    "seal_result_pack",
    "load_reference_validation_registry",
    "validate_bundle",
    "validate_bundles",
    "validate_coupling_invariant_payload",
    "verify_xy_compile_recompute_unit",
    "verb_substrates",
    "write_federation_document",
    "xy_compile_digest_python",
]
