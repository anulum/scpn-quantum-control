#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Enzyme/MLIR breadth evidence writer
"""Write the Enzyme/MLIR compiler-AD breadth artifact and refreshed audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scpn_quantum_control.compiler import (
    EnzymeMLIRBenchmarkAttachment,
    EnzymeMLIRCompilerADBreadthArtifact,
    EnzymeMLIRCompilerADBreadthCaseEvidence,
    EnzymeNativeExecutionEvidence,
    build_enzyme_mlir_benchmark_attachment,
    build_enzyme_mlir_compiler_ad_breadth_gap_artifact,
    run_enzyme_mlir_maturity_audit,
    write_enzyme_mlir_compiler_ad_breadth_artifact,
)
from scpn_quantum_control.compiler.mlir_enzyme_evidence import (
    ENZYME_MLIR_COMPILER_AD_BREADTH_CASES,
)
from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    validate_phase_qnode_affinity_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "differentiable_phase_qnode"
AUDIT_PATH = DATA_DIR / "enzyme_mlir_maturity_audit_20260616.json"
AUDIT_MARKDOWN_PATH = DATA_DIR / "enzyme_mlir_maturity_audit_20260616.md"
ENZYME_EXECUTION_PATH = DATA_DIR / "enzyme_toolchain_ad_execution_evidence_20260622.json"
PHASE_QNODE_AFFINITY_PATH = (
    DATA_DIR / "local_benchmark_20260616T0955Z" / "phase_qnode_affinity.json"
)
BREADTH_ARTIFACT_ID = "enzyme-mlir-compiler-ad-breadth-artifact-20260706"
BREADTH_CLAIM_BOUNDARY = (
    "Raw Enzyme/MLIR compiler-AD breadth artifact assembled from committed "
    "bounded execution evidence. Passing rows carry derivative-correctness "
    "evidence only; hard-gap rows remain explicit. This artifact does not "
    "promote provider, hardware, GPU, QPU, isolated benchmark, arbitrary "
    "compiler-AD, or performance claims."
)


def main() -> int:
    """Write the breadth artifact plus the refreshed Enzyme/MLIR audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    benchmark_attachment = _benchmark_attachment()
    artifact = build_enzyme_mlir_breadth_artifact(benchmark_attachment)
    files = write_enzyme_mlir_compiler_ad_breadth_artifact(args.output_dir, artifact)
    audit_payload = _refreshed_audit_payload(artifact, benchmark_attachment)
    AUDIT_PATH.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", "utf-8")
    AUDIT_MARKDOWN_PATH.write_text(_render_audit_markdown(audit_payload), "utf-8")
    print(files.json_path)
    print(files.markdown_path)
    print(AUDIT_PATH)
    print(AUDIT_MARKDOWN_PATH)
    return 0


def build_enzyme_mlir_breadth_artifact(
    benchmark_attachment: EnzymeMLIRBenchmarkAttachment,
) -> EnzymeMLIRCompilerADBreadthArtifact:
    """Build the committed Enzyme/MLIR breadth artifact from raw evidence."""

    execution_payload = _json_mapping(ENZYME_EXECUTION_PATH)
    observed_cases = (
        _success_case(
            case_id="scalar_reverse_mode",
            source_case=_case_payload(execution_payload, "scalar_square"),
            transform_modes=("reverse", "vjp"),
            frontend_language="llvm_ir",
            artifact_refs={"raw_case": _artifact_ref(ENZYME_EXECUTION_PATH, "scalar_square")},
        ),
        _success_case(
            case_id="vector_vjp",
            source_case=_case_payload(execution_payload, "vector_sum_squares_4"),
            transform_modes=("reverse", "vjp"),
            frontend_language="llvm_ir",
            artifact_refs={
                "raw_case": _artifact_ref(ENZYME_EXECUTION_PATH, "vector_sum_squares_4")
            },
        ),
        _success_case(
            case_id="matrix_vjp",
            source_case=_case_payload(execution_payload, "matrix_trace_2x2"),
            transform_modes=("reverse", "vjp"),
            frontend_language="llvm_ir",
            artifact_refs={"raw_case": _artifact_ref(ENZYME_EXECUTION_PATH, "matrix_trace_2x2")},
        ),
        _success_case(
            case_id="loop_activity",
            source_case=_case_payload(execution_payload, "vector_weighted_sum_4"),
            transform_modes=("reverse", "vjp"),
            frontend_language="llvm_ir_c_loop",
            artifact_refs={
                "raw_case": _artifact_ref(ENZYME_EXECUTION_PATH, "vector_weighted_sum_4")
            },
        ),
        _aggregate_success_case(
            case_id="llvm_ir_generation",
            execution_payload=execution_payload,
            transform_modes=("reverse", "vjp"),
            frontend_language="llvm_ir",
            artifact_refs={"llvm_ir_generation": _relative(ENZYME_EXECUTION_PATH)},
        ),
        _aggregate_success_case(
            case_id="native_enzyme_execution",
            execution_payload=execution_payload,
            transform_modes=("reverse", "vjp"),
            frontend_language="native_llvm_enzyme",
            artifact_refs={"native_enzyme_execution": _relative(ENZYME_EXECUTION_PATH)},
        ),
        _gap_case(
            case_id="alias_activity",
            transform_modes=("forward", "reverse", "jvp", "vjp"),
            frontend_language="program_ad_alias",
            artifact_refs={
                "program_ad_alias_activity": (
                    "data/differentiable_phase_qnode/"
                    "compiler_alias_activity_evidence_20260706.json"
                )
            },
            failure_class="program_ad_alias_not_enzyme_mlir_raw_case",
            setup_instructions=(
                "Capture Enzyme/MLIR alias-activity raw case evidence before promotion."
            ),
        ),
        _gap_case(
            case_id="matrix_jvp",
            transform_modes=("jvp",),
            frontend_language="llvm_ir",
            artifact_refs={"native_llvm_jit_gate": _llvm_jit_gate_ref()},
            failure_class="matrix_jvp_raw_enzyme_case_missing",
            setup_instructions="Attach raw matrix JVP Enzyme/MLIR case evidence.",
        ),
        _gap_case(
            case_id="mlir_lowering",
            transform_modes=("forward", "reverse", "jvp", "vjp"),
            frontend_language="mlir",
            artifact_refs={"maturity_audit": _relative(AUDIT_PATH)},
            failure_class="mlir_lowering_runtime_row_missing",
            setup_instructions="Attach raw MLIR lowering case evidence with runtime metadata.",
        ),
        _gap_case(
            case_id="scalar_forward_mode",
            transform_modes=("forward",),
            frontend_language="llvm_ir",
            artifact_refs={"native_llvm_jit_gate": _llvm_jit_gate_ref()},
            failure_class="scalar_forward_raw_enzyme_case_missing",
            setup_instructions="Attach raw scalar forward-mode Enzyme/MLIR case evidence.",
        ),
        _gap_case(
            case_id="vector_jvp",
            transform_modes=("jvp",),
            frontend_language="llvm_ir",
            artifact_refs={"native_llvm_jit_gate": _llvm_jit_gate_ref()},
            failure_class="vector_jvp_raw_enzyme_case_missing",
            setup_instructions="Attach raw vector JVP Enzyme/MLIR case evidence.",
        ),
    )
    return build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
        artifact_id=BREADTH_ARTIFACT_ID,
        observed_cases=observed_cases,
        isolated_benchmark_evidence=benchmark_attachment,
        claim_boundary=BREADTH_CLAIM_BOUNDARY,
    )


def _benchmark_attachment() -> EnzymeMLIRBenchmarkAttachment:
    validation = validate_phase_qnode_affinity_artifact(PHASE_QNODE_AFFINITY_PATH)
    return build_enzyme_mlir_benchmark_attachment(
        validation=validation,
        required_breadth_cases=tuple(sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES)),
        claim_boundary=(
            "Enzyme/MLIR compiler-AD breadth benchmark attachment. The current "
            "committed benchmark is functional_non_isolated and therefore blocks "
            "provider-exceedance and performance promotion."
        ),
    )


def _success_case(
    *,
    case_id: str,
    source_case: Mapping[str, Any],
    transform_modes: Sequence[str],
    frontend_language: str,
    artifact_refs: Mapping[str, str],
) -> EnzymeMLIRCompilerADBreadthCaseEvidence:
    return EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id=case_id,
        status="success",
        transform_modes=tuple(transform_modes),
        frontend_language=frontend_language,
        value_error=0.0,
        gradient_error=_float_field(source_case, "gradient_error"),
        runtime_seconds=_float_field(source_case, "runtime_seconds"),
        artifact_refs=artifact_refs,
        failure_class=None,
        setup_instructions=None,
        claim_boundary=BREADTH_CLAIM_BOUNDARY,
    )


def _aggregate_success_case(
    *,
    case_id: str,
    execution_payload: Mapping[str, Any],
    transform_modes: Sequence[str],
    frontend_language: str,
    artifact_refs: Mapping[str, str],
) -> EnzymeMLIRCompilerADBreadthCaseEvidence:
    executed = tuple(
        case for case in _case_payloads(execution_payload) if str(case.get("status")) == "executed"
    )
    runtime_seconds = sum(_float_field(case, "runtime_seconds") for case in executed)
    max_gradient_error = max(_float_field(case, "gradient_error") for case in executed)
    return EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id=case_id,
        status="success",
        transform_modes=tuple(transform_modes),
        frontend_language=frontend_language,
        value_error=0.0,
        gradient_error=max_gradient_error,
        runtime_seconds=runtime_seconds,
        artifact_refs=artifact_refs,
        failure_class=None,
        setup_instructions=None,
        claim_boundary=BREADTH_CLAIM_BOUNDARY,
    )


def _gap_case(
    *,
    case_id: str,
    transform_modes: Sequence[str],
    frontend_language: str,
    artifact_refs: Mapping[str, str],
    failure_class: str,
    setup_instructions: str,
) -> EnzymeMLIRCompilerADBreadthCaseEvidence:
    return EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id=case_id,
        status="hard_gap",
        transform_modes=tuple(transform_modes),
        frontend_language=frontend_language,
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        artifact_refs=artifact_refs,
        failure_class=failure_class,
        setup_instructions=setup_instructions,
        claim_boundary=BREADTH_CLAIM_BOUNDARY,
    )


def _refreshed_audit_payload(
    artifact: EnzymeMLIRCompilerADBreadthArtifact,
    benchmark_attachment: EnzymeMLIRBenchmarkAttachment,
) -> dict[str, object]:
    previous = _json_mapping(AUDIT_PATH)
    native_evidence = _native_evidence(previous)
    result = run_enzyme_mlir_maturity_audit(
        isolated_benchmark_artifact_id=benchmark_attachment.benchmark_artifact_id,
        isolated_benchmark_evidence=benchmark_attachment,
        native_enzyme_execution_evidence=native_evidence,
        mlir_llvm_correctness_artifact_id=str(
            _mapping_field(previous, "mlir_llvm_correctness_evidence")["artifact_id"]
        ),
        compiler_ad_breadth_artifact=artifact,
    )
    payload = dict(result.to_dict())
    payload.update(
        {
            "schema": "scpn_qc_enzyme_mlir_maturity_audit_v2",
            "artifact_id": "enzyme-mlir-maturity-audit-20260616",
            "classification": "hard_gap",
            "promotion_ready": False,
            "spdx_license_identifier": "AGPL-3.0-or-later",
            "commercial_license": "available",
            "copyright": ("© Concepts 1996–2026 Miroslav Šotek. © Code 2020–2026 Miroslav Šotek."),
            "orcid": "0009-0009-3560-0851",
            "contact": "www.anulum.li | protoscience@anulum.li",
            "source_external_comparison_artifact": previous.get(
                "source_external_comparison_artifact",
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "diff-qnode-external-comparison.json",
            ),
            "source_native_probe": previous.get(
                "source_native_probe",
                "LLVM IR square derivative probe generated with opt -passes=enzyme "
                "and clang execution",
            ),
        }
    )
    return payload


def _render_audit_markdown(payload: Mapping[str, object]) -> str:
    native = _mapping_field(payload, "native_enzyme_execution_evidence")
    mlir = _mapping_field(payload, "mlir_llvm_correctness_evidence")
    breadth = _mapping_field(payload, "compiler_ad_breadth_artifact")
    hard_gaps = tuple(str(gap) for gap in _sequence_field(payload, "hard_gaps"))
    toolchain = _mapping_field(payload, "toolchain")
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Enzyme/MLIR maturity audit artefact",
        "-->",
        "",
        "# Enzyme/MLIR Maturity Audit 2026-06-16",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Artefact ID | `{payload['artifact_id']}` |",
        f"| Classification | `{payload['classification']}` |",
        f"| Promotion ready | `{payload['promotion_ready']}` |",
        f"| SCPN MLIR runtime verified | `{payload['scpn_mlir_runtime_verified']}` |",
        f"| Native Enzyme evidence | `{native['artifact_id']}` |",
        f"| Native Enzyme status | `{native['status']}` |",
        f"| Native Enzyme value error | `{native['value_error']}` |",
        f"| Native Enzyme gradient error | `{native['gradient_error']}` |",
        f"| MLIR/LLVM correctness evidence | `{mlir['artifact_id']}` |",
        f"| Compiler AD breadth artifact | `{breadth['artifact_id']}` |",
        "| Compiler AD breadth evidence | `missing` |",
        f"| Ready for provider exceedance | `{payload['ready_for_provider_exceedance']}` |",
        "",
        "## Toolchain Snapshot",
        "",
    ]
    for command, status_value in toolchain.items():
        status = cast(Mapping[str, object], status_value)
        lines.append(f"- `{command}`: `{status['version']}` at `{status['executable']}`")
    lines.extend(
        [
            "",
            "## Breadth Artifact",
            "",
            (
                f"`{breadth['artifact_id']}` records 11 Enzyme/MLIR compiler-AD "
                "breadth rows with explicit hard gaps for missing raw cases."
            ),
            "",
            "## Hard Gaps",
            "",
            *[f"- `{gap}`" for gap in hard_gaps],
            "",
            "## Boundary",
            "",
            str(payload["claim_boundary"]),
            "",
            (
                "The direct LLVM Enzyme probe and raw breadth artifact remain bounded "
                "compiler-AD evidence. This artefact still does not promote "
                "Enzyme/MLIR parity, provider execution, hardware execution, arbitrary "
                "compiler-AD breadth, or performance claims without promotion-ready "
                "isolated benchmark and derived breadth evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _native_evidence(payload: Mapping[str, Any]) -> EnzymeNativeExecutionEvidence:
    native = _mapping_field(payload, "native_enzyme_execution_evidence")
    return EnzymeNativeExecutionEvidence(
        artifact_id=str(native["artifact_id"]),
        status=str(native["status"]),
        failure_class=_optional_str(native.get("failure_class")),
        value_error=_optional_float(native.get("value_error")),
        gradient_error=_optional_float(native.get("gradient_error")),
        runtime_seconds=_optional_float(native.get("runtime_seconds")),
        toolchain=_string_mapping(_mapping_field(native, "toolchain")),
        setup_instructions=_optional_str(native.get("setup_instructions")),
        claim_boundary=str(native["claim_boundary"]),
    )


def _json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], payload)


def _case_payload(payload: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for case in _case_payloads(payload):
        if case.get("case_id") == case_id:
            return case
    raise ValueError(f"missing Enzyme execution case: {case_id}")


def _case_payloads(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Enzyme execution payload must contain a case list")
    rows: list[Mapping[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Enzyme execution case rows must be JSON objects")
        rows.append(cast(Mapping[str, Any], case))
    return tuple(rows)


def _mapping_field(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence_field(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON list")
    return tuple(value)


def _string_mapping(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("toolchain metadata must map strings to strings")
        result[key] = value
    return result


def _float_field(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError("optional float field must be numeric or null")
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string field must be a string or null")
    return value


def _artifact_ref(path: Path, case_id: str) -> str:
    return f"{_relative(path)}#case_id={case_id}"


def _llvm_jit_gate_ref() -> str:
    return "data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json"


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
