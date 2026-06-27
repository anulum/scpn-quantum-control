# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable External Comparison Tests
"""Tests for richer external differentiable framework comparison rows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS,
    ExternalComparisonArtifact,
    ExternalComparisonRow,
    IdenticalCircuitGradientComparisonArtifact,
    IdenticalCircuitGradientComparisonRow,
    external_comparison_failure_mode_rows,
    run_differentiable_external_comparison_suite,
    run_identical_circuit_gradient_comparison_suite,
    write_differentiable_external_comparison,
    write_identical_circuit_gradient_comparison,
)
from scpn_quantum_control.phase.qnode_circuit import (
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)


def test_external_comparison_suite_records_success_rows_and_enzyme_hard_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: True)
    monkeypatch.setattr(comparison, "_enzyme_runner_configured", lambda: False)
    monkeypatch.setattr(comparison, "_catalyst_runner_configured", lambda: False)
    monkeypatch.setattr(
        comparison,
        "_run_jax_reference",
        lambda values: (
            comparison._bounded_phase_objective(values),
            comparison._bounded_phase_gradient(values),
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_run_pytorch_reference",
        lambda values: (
            comparison._bounded_phase_objective(values),
            comparison._bounded_phase_gradient(values),
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_run_tensorflow_reference",
        lambda values: (
            comparison._bounded_phase_objective(values),
            comparison._bounded_phase_gradient(values),
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_run_pennylane_reference",
        lambda values: (
            comparison._bounded_phase_objective(values),
            comparison._bounded_phase_gradient(values),
        ),
    )

    rows = run_differentiable_external_comparison_suite()
    by_backend = {row.backend: row for row in rows if row.case_id == "bounded_phase_objective"}

    assert set(by_backend) == {
        "jax",
        "pytorch",
        "tensorflow",
        "pennylane",
        "enzyme",
        "catalyst",
    }
    for backend in ("jax", "pytorch", "tensorflow", "pennylane"):
        row = by_backend[backend]
        assert row.status == "success"
        assert row.failure_class is None
        assert row.value_error is not None and row.value_error <= 1e-12
        assert row.gradient_error is not None and row.gradient_error <= 1e-12
        assert row.artifact_fields_ready
        assert row.source_of_truth == "scpn_reference"
        assert row.dependency_versions
    assert by_backend["jax"].batching_support == "vmap"
    assert by_backend["pytorch"].batching_support == "torch.func.vmap"
    assert by_backend["tensorflow"].transform_support == "GradientTape"
    assert by_backend["pennylane"].transform_support == "QNode"
    assert by_backend["enzyme"].status == "hard_gap"
    assert by_backend["enzyme"].failure_class == "dependency_missing"
    assert "LLVM/Enzyme" in str(by_backend["enzyme"].setup_instructions)
    assert by_backend["catalyst"].status == "hard_gap"
    assert by_backend["catalyst"].failure_class == "dependency_missing"
    assert "Catalyst" in str(by_backend["catalyst"].setup_instructions)
    failure_classes = {row.failure_class for row in rows if row.status == "hard_gap"}
    assert {
        "unsupported_batching",
        "unsupported_transform",
        "unsupported_dtype",
        "unsupported_device",
    }.issubset(failure_classes)


def test_identical_circuit_gradient_comparison_records_live_backend_boundaries() -> None:
    rows = run_identical_circuit_gradient_comparison_suite()
    by_backend = {row.backend: row for row in rows}

    assert set(by_backend) == {"qiskit", "pennylane"}
    for row in rows:
        assert row.case_id == "single_ry_z_expectation_exact_state"
        assert row.circuit_fingerprint
        assert row.execution_mode == "exact_state"
        assert row.shots is None
        assert row.observable == "Z0"
        assert row.parameter_values == (0.4,)
        assert row.dependency_versions
        assert row.artifact_fields_ready
        assert not row.performance_claim_eligible
        if row.status == "success":
            assert row.failure_class is None
            assert row.value_error is not None and row.value_error <= 1e-12
            assert row.gradient_error is not None and row.gradient_error <= 1e-12
        else:
            assert row.failure_class in {"dependency_missing", "runtime_error"}
            assert row.value_error is None
            assert row.gradient_error is None


def test_identical_circuit_gradient_comparison_success_rows_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQiskitResult:
        value = np.cos(0.4)
        gradient = np.array([-np.sin(0.4)], dtype=np.float64)
        evaluations = 2

    class FakePennyLaneResult:
        pennylane_value = np.cos(0.4)
        pennylane_gradient = np.array([-np.sin(0.4)], dtype=np.float64)
        evaluations = 2

    monkeypatch.setattr(
        comparison,
        "_qiskit_identical_circuit_row",
        lambda **kwargs: comparison._identical_circuit_success_row(
            backend="qiskit",
            operations=kwargs["operations"],
            observable_label=kwargs["observable_label"],
            values=kwargs["values"],
            fingerprint=kwargs["fingerprint"],
            scpn_value=kwargs["scpn_value"],
            backend_value=float(FakeQiskitResult.value),
            scpn_gradient=kwargs["scpn_gradient"],
            backend_gradient=tuple(float(item) for item in FakeQiskitResult.gradient),
            evaluations=FakeQiskitResult.evaluations,
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_pennylane_identical_circuit_row",
        lambda **kwargs: comparison._identical_circuit_success_row(
            backend="pennylane",
            operations=kwargs["operations"],
            observable_label=kwargs["observable_label"],
            values=kwargs["values"],
            fingerprint=kwargs["fingerprint"],
            scpn_value=kwargs["scpn_value"],
            backend_value=float(FakePennyLaneResult.pennylane_value),
            scpn_gradient=kwargs["scpn_gradient"],
            backend_gradient=tuple(float(item) for item in FakePennyLaneResult.pennylane_gradient),
            evaluations=FakePennyLaneResult.evaluations,
        ),
    )

    rows = run_identical_circuit_gradient_comparison_suite()

    assert {row.backend for row in rows} == {"qiskit", "pennylane"}
    for row in rows:
        assert row.status == "success"
        assert row.failure_class is None
        assert row.value_error is not None and row.value_error <= 1e-12
        assert row.gradient_error is not None and row.gradient_error <= 1e-12


def test_identical_circuit_gradient_comparison_writer_marks_ready_not_promoted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "identical-circuit.json"
    circuit, values, operations, observable_label, fingerprint = (
        comparison._identical_circuit_problem()
    )
    scpn_value = execute_phase_qnode_circuit(circuit, values).value
    scpn_gradient = tuple(
        float(item) for item in parameter_shift_phase_qnode_gradient(circuit, values).gradient
    )
    rows = (
        comparison._identical_circuit_success_row(
            backend="qiskit",
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
            scpn_value=float(scpn_value),
            backend_value=float(scpn_value),
            scpn_gradient=scpn_gradient,
            backend_gradient=scpn_gradient,
            evaluations=2,
        ),
        comparison._identical_circuit_success_row(
            backend="pennylane",
            operations=operations,
            observable_label=observable_label,
            values=values,
            fingerprint=fingerprint,
            scpn_value=float(scpn_value),
            backend_value=float(scpn_value),
            scpn_gradient=scpn_gradient,
            backend_gradient=scpn_gradient,
            evaluations=2,
        ),
    )

    artifact = write_identical_circuit_gradient_comparison(output, rows=rows)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert isinstance(artifact, IdenticalCircuitGradientComparisonArtifact)
    assert artifact.identical_circuit_ready
    assert not artifact.promotion_ready
    assert payload["summary"]["success_count"] == 2
    assert payload["summary"]["hard_gap_count"] == 0
    assert payload["identical_circuit_ready"] is True
    assert payload["promotion_ready"] is False
    assert payload["same_circuit_contract"]["shots"] is None


def test_identical_circuit_gradient_comparison_row_requires_success_evidence() -> None:
    try:
        IdenticalCircuitGradientComparisonRow(
            case_id="case",
            backend="qiskit",
            status="success",
            failure_class=None,
            circuit_fingerprint="abc",
            operations=(("ry", (0,), 0),),
            observable="Z0",
            parameter_values=(0.4,),
            execution_mode="exact_state",
            shots=None,
            scpn_value=1.0,
            backend_value=None,
            value_error=0.0,
            scpn_gradient=(0.0,),
            backend_gradient=None,
            gradient_error=None,
            evaluations=3,
            dependency_versions={"qiskit": "test"},
            claim_boundary="bounded comparison",
        )
    except ValueError as exc:
        assert "success rows require numeric" in str(exc)
    else:
        raise AssertionError("success row without backend gradient was accepted")


def test_external_comparison_suite_classifies_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: True)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: False)
    monkeypatch.setattr(comparison, "_enzyme_runner_configured", lambda: False)
    monkeypatch.setattr(comparison, "_catalyst_runner_configured", lambda: False)

    def broken_runner(
        values: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[float, np.ndarray[Any, np.dtype[np.float64]]]:
        raise RuntimeError("framework callback failed")

    monkeypatch.setattr(comparison, "_run_jax_reference", broken_runner)

    row = {
        item.backend: item
        for item in run_differentiable_external_comparison_suite()
        if item.case_id == "bounded_phase_objective"
    }["jax"]

    assert row.status == "hard_gap"
    assert row.failure_class == "runtime_error"
    assert "framework callback failed" in str(row.setup_instructions)


def test_external_comparison_row_requires_hard_gap_fields() -> None:
    row = ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="enzyme",
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support="not_evaluated",
        transform_support="LLVM Enzyme",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions="Install LLVM/Enzyme tooling.",
        claim_boundary="recorded hard gap only",
    )

    assert row.artifact_fields_ready
    assert row.to_dict()["failure_class"] == "dependency_missing"
    assert row.to_dict()["dependency_versions"] is None


def test_external_comparison_row_rejects_success_without_numeric_evidence() -> None:
    try:
        ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="jax",
            status="success",
            failure_class=None,
            value_error=0.0,
            gradient_error=None,
            runtime_seconds=0.0,
            memory_peak_bytes=1024,
            batching_support="vmap",
            transform_support="value_and_grad",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=None,
            claim_boundary="diagnostic comparison only",
        )
    except ValueError as exc:
        assert "success rows require numeric" in str(exc)
    else:
        raise AssertionError("success row without gradient evidence was accepted")


def test_external_comparison_suite_records_dependency_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: False)
    monkeypatch.setattr(comparison, "_enzyme_runner_configured", lambda: False)
    monkeypatch.setattr(comparison, "_catalyst_runner_configured", lambda: False)

    rows = run_differentiable_external_comparison_suite()

    primary_rows = [row for row in rows if row.case_id == "bounded_phase_objective"]
    assert all(row.status == "hard_gap" for row in primary_rows)
    assert all(row.failure_class == "dependency_missing" for row in primary_rows)
    assert np.isfinite(len(rows))


def test_external_comparison_runs_configured_enzyme_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "enzyme_runner.py"
    runner.write_text(
        "\n".join(
            (
                "#!/usr/bin/env python3",
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                "# Commercial license available",
                "# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
                "# © Code 2020-2026 Miroslav Sotek. All rights reserved.",
                "# ORCID: 0009-0009-3560-0851",
                "# Contact: www.anulum.li | protoscience@anulum.li",
                "import json, math, sys",
                "payload = json.load(sys.stdin)",
                "values = payload['values']",
                "print(json.dumps({",
                "    'value': math.cos(values[0]) + 0.25 * math.sin(values[1]),",
                "    'gradient': [-math.sin(values[0]), 0.25 * math.cos(values[1])],",
                "    'toolchain': {'enzyme': 'test-runner', 'llvm': 'test-llvm'},",
                "}))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: True)

    row = comparison._enzyme_row()
    payload = row.to_dict()

    assert row.status == "success"
    assert row.failure_class is None
    assert row.value_error is not None and row.value_error <= 1e-12
    assert row.gradient_error is not None and row.gradient_error <= 1e-12
    assert row.batching_support == "not_supported"
    assert row.transform_support == "LLVM Enzyme runner"
    assert "Enzyme" in row.claim_boundary
    assert payload["toolchain"] == {"enzyme": "test-runner", "llvm": "test-llvm"}
    assert payload["dependency_versions"]


def test_external_comparison_runs_configured_catalyst_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "catalyst_runner.py"
    runner.write_text(
        "\n".join(
            (
                "#!/usr/bin/env python3",
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                "# Commercial license available",
                "# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
                "# © Code 2020-2026 Miroslav Sotek. All rights reserved.",
                "# ORCID: 0009-0009-3560-0851",
                "# Contact: www.anulum.li | protoscience@anulum.li",
                "import json, math, sys",
                "payload = json.load(sys.stdin)",
                "values = payload['values']",
                "print(json.dumps({",
                "    'value': math.cos(values[0]) + 0.25 * math.sin(values[1]),",
                "    'gradient': [-math.sin(values[0]), 0.25 * math.cos(values[1])],",
                "    'toolchain': {'catalyst': 'test-runner', 'mlir': 'test-mlir'},",
                "}))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    row = comparison._catalyst_row()
    payload = row.to_dict()

    assert row.status == "success"
    assert row.failure_class is None
    assert row.value_error is not None and row.value_error <= 1e-12
    assert row.gradient_error is not None and row.gradient_error <= 1e-12
    assert row.batching_support == "not_supported"
    assert row.transform_support == "Catalyst qjit/MLIR/QIR runner"
    assert "Catalyst" in row.claim_boundary
    assert payload["toolchain"] == {"catalyst": "test-runner", "mlir": "test-mlir"}
    assert payload["dependency_versions"]


def test_external_comparison_records_enzyme_jax_tooling_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "enzyme_call.so"
    runner = tmp_path / "enzyme_runner.py"
    plugin.write_bytes(b"native-extension-placeholder")
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setenv("ENZYME_LLVM_PLUGIN", str(plugin))
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    monkeypatch.setattr(
        comparison,
        "_installed_version",
        lambda package: {
            "llvm": "not_installed",
            "enzyme": "not_installed",
            "enzyme_ad": "0.0.6",
        }[package],
    )

    versions = comparison._backend_dependency_versions("enzyme")

    assert versions["enzyme_ad"] == "0.0.6"
    assert versions["enzyme_llvm_plugin"] == f"file:{plugin}"
    assert versions["enzyme_runner"] == f"executable:{runner}"


def test_external_comparison_records_catalyst_tooling_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "catalyst_runner.py"
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))
    monkeypatch.setattr(
        comparison,
        "_installed_version",
        lambda package: {
            "pennylane-catalyst": "0.13.0",
            "catalyst": "importable_unknown_version",
            "mlir": "executable:/usr/bin/mlir-opt",
            "llvm": "executable:/usr/bin/llvm-config",
        }[package],
    )

    versions = comparison._backend_dependency_versions("catalyst")

    assert versions["pennylane-catalyst"] == "0.13.0"
    assert versions["catalyst_runner"] == f"executable:{runner}"


def test_external_comparison_classifies_enzyme_bad_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "bad_enzyme_runner.py"
    runner.write_text("#!/usr/bin/env python3\nprint('not-json')\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: True)

    row = comparison._enzyme_row()

    assert row.status == "hard_gap"
    assert row.failure_class == "runtime_error"
    assert "valid JSON" in str(row.setup_instructions)


def test_external_comparison_classifies_catalyst_bad_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "bad_catalyst_runner.py"
    runner.write_text("#!/usr/bin/env python3\nprint('not-json')\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    row = comparison._catalyst_row()

    assert row.status == "hard_gap"
    assert row.failure_class == "runtime_error"
    assert "valid JSON" in str(row.setup_instructions)


def test_external_comparison_rejects_enzyme_wrong_gradient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "wrong_enzyme_runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'value': 1.0, 'gradient': [0.0, 0.0]}))\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_ENZYME_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_enzyme_tooling_available", lambda: True)

    row = comparison._enzyme_row()

    assert row.status == "hard_gap"
    assert row.failure_class == "correctness_mismatch"
    assert "SCPN reference" in str(row.setup_instructions)


def test_external_comparison_failure_mode_rows_cover_required_taxonomy() -> None:
    rows = external_comparison_failure_mode_rows()
    by_failure = {row.failure_class: row for row in rows}

    assert set(by_failure) == {
        "unsupported_batching",
        "unsupported_transform",
        "unsupported_dtype",
        "unsupported_device",
    }
    assert by_failure["unsupported_batching"].backend == "jax"
    assert by_failure["unsupported_transform"].backend == "pytorch"
    assert by_failure["unsupported_dtype"].dtype == "complex128"
    assert by_failure["unsupported_device"].device == "hardware_qpu"
    for row in rows:
        assert row.status == "hard_gap"
        assert row.value_error is None
        assert row.gradient_error is None
        assert row.dependency_versions is not None
        assert "no hidden success" in row.claim_boundary


def test_external_comparison_row_rejects_empty_dependency_metadata() -> None:
    try:
        ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="jax",
            status="hard_gap",
            failure_class="unsupported_dtype",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="vmap",
            transform_support="value_and_grad",
            dtype="complex128",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions="Use real float64 controls.",
            claim_boundary="unsupported route only",
            dependency_versions={"jax": ""},
        )
    except ValueError as exc:
        assert "dependency version metadata" in str(exc)
    else:
        raise AssertionError("empty dependency metadata was accepted")


def test_external_comparison_dependency_version_falls_back_to_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_module: Any = comparison.__dict__["metadata"]

    def missing_distribution(package: str) -> str:
        raise metadata_module.PackageNotFoundError(package)

    monkeypatch.setattr(metadata_module, "version", missing_distribution)
    monkeypatch.setattr(
        comparison,
        "import_module",
        lambda name: SimpleNamespace(__version__="2.21.0") if name == "tensorflow" else None,
    )

    assert comparison._installed_version("tensorflow") == "2.21.0"


def test_external_comparison_writer_records_non_promotional_artifact(tmp_path: Path) -> None:
    rows = (
        ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="jax",
            status="success",
            failure_class=None,
            value_error=0.0,
            gradient_error=0.0,
            runtime_seconds=0.01,
            memory_peak_bytes=4096,
            batching_support="vmap",
            transform_support="value_and_grad",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=None,
            claim_boundary="bounded CPU comparison only",
            dependency_versions={"jax": "0.0", "jaxlib": "0.0"},
        ),
        ExternalComparisonRow(
            case_id="bounded_phase_objective_unsupported_dtype",
            backend="tensorflow",
            status="hard_gap",
            failure_class="unsupported_dtype",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="vectorized_map",
            transform_support="GradientTape",
            dtype="complex128",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions="Use real float64 controls.",
            claim_boundary="unsupported route only",
            dependency_versions={"tensorflow": "not_installed"},
        ),
    )

    artifact = write_differentiable_external_comparison(
        tmp_path / "external" / "comparison.json",
        rows,
        artifact_id="unit-external-comparison",
    )
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))

    assert isinstance(artifact, ExternalComparisonArtifact)
    assert artifact.artifact_id == "unit-external-comparison"
    assert artifact.to_dict()["artifact_id"] == "unit-external-comparison"
    assert artifact.to_dict()["classification"] == "functional_non_isolated"
    assert payload["schema"] == "scpn_qc_differentiable_external_comparison_v1"
    assert payload["artifact_id"] == "unit-external-comparison"
    assert payload["classification"] == "functional_non_isolated"
    assert payload["production_eligible"] is False
    assert payload["promotion_ready"] is False
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["success_count"] == 1
    assert payload["summary"]["hard_gap_count"] == 1
    assert payload["summary"]["failure_classes"] == ["unsupported_dtype"]
    assert payload["rows"][0]["dependency_versions"] == {"jax": "0.0", "jaxlib": "0.0"}
    assert set(payload["row_schema"]["required_fields"]) == REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS
    for row in payload["rows"]:
        assert set(row) >= REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS
    assert "not isolated benchmark evidence" in payload["claim_boundary"]


def test_external_comparison_writer_rejects_incomplete_row_payload(tmp_path: Path) -> None:
    class IncompleteExternalRow:
        case_id = "bounded_phase_objective"
        backend = "jax"
        status = "success"
        failure_class = None

        @property
        def artifact_fields_ready(self) -> bool:
            return False

        def to_dict(self) -> dict[str, object]:
            return {"case_id": self.case_id, "backend": self.backend}

    try:
        write_differentiable_external_comparison(
            tmp_path / "comparison.json",
            (IncompleteExternalRow(),),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "required artefact fields" in str(exc)
    else:
        raise AssertionError("incomplete external comparison row was accepted")


def test_external_comparison_writer_rejects_invalid_outputs(tmp_path: Path) -> None:
    row = ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="jax",
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support="vmap",
        transform_support="value_and_grad",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions="Install JAX.",
        claim_boundary="dependency gap only",
    )

    try:
        write_differentiable_external_comparison(tmp_path / "comparison.txt", (row,))
    except ValueError as exc:
        assert "must end with .json" in str(exc)
    else:
        raise AssertionError("non-JSON external comparison artifact path was accepted")

    try:
        write_differentiable_external_comparison(tmp_path / "comparison.json", (), artifact_id="")
    except ValueError as exc:
        assert "artifact_id" in str(exc)
    else:
        raise AssertionError("empty external comparison artifact id was accepted")
