# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch bridge validation edges tests
# scpn-quantum-control -- PyTorch bridge validation edge tests
"""Validation and metadata edge tests for the phase PyTorch bridge."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTorchCloudValidationRunSpec,
    PhaseTorchEcosystemMaturityAuditResult,
    PhaseTorchEcosystemMaturityRoute,
    PhaseTorchLiveOverlayEvidence,
    PhaseTorchPhaseQNodeLoweringMatrixResult,
    PhaseTorchPhaseQNodeLoweringRoute,
    is_phase_torch_available,
    plan_torch_cloud_validation_batch,
    run_torch_phase_qnode_lowering_matrix,
    torch_bounded_qnn_value_and_grad,
    torch_parameter_shift_value_and_grad,
)

pytest.importorskip("torch")  # the torch bridge is an optional extra; skip when torch is absent

FloatArray = NDArray[np.float64]


def _features() -> FloatArray:
    """Return a deterministic bounded phase-QNN feature matrix."""

    return np.array([[0.0, 0.4], [np.pi, -0.2]], dtype=np.float64)


def _labels() -> FloatArray:
    """Return deterministic bounded phase-QNN labels."""

    return np.array([0.0, 1.0], dtype=np.float64)


def _params() -> FloatArray:
    """Return deterministic bounded phase-QNN parameters."""

    return np.array([0.2, -0.1], dtype=np.float64)


def test_torch_bridge_reports_unavailable_import_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public availability checks should fail closed when PyTorch import fails."""

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch":
            raise ImportError("blocked for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert is_phase_torch_available() is False
    with pytest.raises(ImportError, match="PyTorch is unavailable"):
        torch_parameter_shift_value_and_grad(lambda values: float(values[0]), [0.1])


@pytest.mark.parametrize(
    ("features", "labels", "params", "tolerance", "match"),
    (
        (np.array([1.0, 2.0]), _labels(), _params(), 1e-6, "features"),
        (np.empty((0, 2)), _labels(), _params(), 1e-6, "features must not be empty"),
        (
            np.array([[np.nan, 0.0], [1.0, 2.0]]),
            _labels(),
            _params(),
            1e-6,
            "features must contain only finite",
        ),
        (_features(), np.array([[0.0], [1.0]]), _params(), 1e-6, "labels"),
        (_features(), np.array([0.0]), _params(), 1e-6, "labels must have shape"),
        (_features(), np.array([0.0, np.nan]), _params(), 1e-6, "labels must contain"),
        (_features(), _labels(), np.array([[0.1, 0.2]]), 1e-6, "values"),
        (_features(), _labels(), np.array([0.1]), 1e-6, "params width"),
        (_features(), _labels(), np.array([0.1, np.nan]), 1e-6, "values"),
        (_features(), _labels(), _params(), -1.0, "tolerance"),
    ),
)
def test_torch_bounded_qnn_value_and_grad_rejects_malformed_inputs(
    features: FloatArray,
    labels: FloatArray,
    params: FloatArray,
    tolerance: float,
    match: str,
) -> None:
    """The public bounded-QNN bridge should reject malformed numeric inputs."""

    with pytest.raises(ValueError, match=match):
        torch_bounded_qnn_value_and_grad(
            features,
            labels,
            params,
            tolerance=tolerance,
        )


def test_torch_phase_lowering_and_ecosystem_metadata_unknown_routes() -> None:
    """Public metadata results should fail closed for unknown route names."""

    lowering = run_torch_phase_qnode_lowering_matrix()
    assert isinstance(lowering, PhaseTorchPhaseQNodeLoweringMatrixResult)
    assert lowering.route_status("bounded_qnn_analytic_tensor") == "passed"
    with pytest.raises(KeyError, match="unknown PyTorch Phase-QNode lowering route"):
        lowering.route_status("missing")
    payload = lowering.to_dict()
    assert payload["ready_for_provider_exceedance"] is False
    open_gaps = payload["open_gaps"]
    assert isinstance(open_gaps, list)
    assert "registered_phase_qnode_hardware_lowering" in open_gaps

    ecosystem = PhaseTorchEcosystemMaturityAuditResult(
        torch_version="test",
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_names=(),
        routes=(
            PhaseTorchEcosystemMaturityRoute(
                name="nn_module_parameter_surface",
                status="passed",
                reason="available",
            ),
            PhaseTorchEcosystemMaturityRoute(
                name="cuda_accelerator_device",
                status="blocked",
                reason="no visible CUDA devices",
                requires=("compatible_cuda_device",),
            ),
        ),
    )
    assert ecosystem.route_status("cuda_accelerator_device") == "blocked"
    with pytest.raises(KeyError, match="unknown PyTorch ecosystem maturity route"):
        ecosystem.route_status("missing")
    assert ecosystem.to_dict()["ready_for_provider_exceedance"] is False


def test_torch_cloud_validation_batch_rejects_bad_inputs_and_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud-validation planning should validate runner and accelerator labels."""

    route = PhaseTorchEcosystemMaturityRoute(
        name="cuda_accelerator_device",
        status="blocked",
        reason="no visible CUDA devices",
        requires=("compatible_cuda_device",),
    )
    monkeypatch.setattr(
        torch_bridge,
        "run_torch_ecosystem_maturity_audit",
        lambda: PhaseTorchEcosystemMaturityAuditResult(
            torch_version="test",
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_names=(),
            routes=(route,),
        ),
    )
    monkeypatch.setattr(
        torch_bridge,
        "run_torch_phase_qnode_lowering_matrix",
        lambda: PhaseTorchPhaseQNodeLoweringMatrixResult(
            routes=(
                PhaseTorchPhaseQNodeLoweringRoute(
                    name="isolated_benchmark_artifact",
                    status="blocked",
                    reason="needs isolated benchmark",
                    requires=("isolated_affinity_benchmark_id",),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="runner"):
        plan_torch_cloud_validation_batch(runner="  ")
    with pytest.raises(ValueError, match="accelerator_backend"):
        plan_torch_cloud_validation_batch(accelerator_backend="metal")

    plan = plan_torch_cloud_validation_batch(runner="ML350", accelerator_backend="rocm")
    assert isinstance(plan, PhaseTorchCloudValidationRunSpec)
    assert plan.ready_for_cloud_dispatch
    payload = cast(dict[str, Any], plan.to_dict())
    assert payload["runner"] == "ML350"
    assert payload["required_environment"]["accelerator_backend"] == "rocm"


def test_torch_live_overlay_loader_rejects_malformed_artifacts(tmp_path: Path) -> None:
    """Live-overlay evidence loading should reject malformed committed artifacts."""

    path = tmp_path / "overlay.json"

    for payload, match in (
        ([], "JSON object"),
        ({"classification": "isolated_affinity", "rows": []}, "functional_non_isolated"),
        ({"classification": "functional_non_isolated"}, "include rows"),
        (
            {"classification": "functional_non_isolated", "rows": []},
            "successful PyTorch row",
        ),
        (
            {
                "classification": "functional_non_isolated",
                "rows": [{"backend": "pytorch", "status": "success"}],
            },
            "dependency_versions",
        ),
        (
            {
                "classification": "functional_non_isolated",
                "rows": [
                    {
                        "backend": "pytorch",
                        "status": "success",
                        "dependency_versions": {"torch": ""},
                    },
                ],
            },
            "torch dependency version",
        ),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            torch_bridge._load_torch_live_overlay_evidence(path)


def test_torch_live_overlay_loader_accepts_valid_artifact(tmp_path: Path) -> None:
    """Live-overlay evidence loading should preserve validated PyTorch row metadata."""

    path = tmp_path / "overlay.json"
    path.write_text(
        json.dumps(
            {
                "artifact_id": "torch-functional-local",
                "classification": "functional_non_isolated",
                "promotion_ready": False,
                "rows": [
                    {
                        "backend": "pytorch",
                        "status": "success",
                        "dependency_versions": {"torch": "2.test"},
                        "value_error": 0.0,
                        "gradient_error": 0.0,
                        "runtime_seconds": 1.25,
                        "memory_peak_bytes": 1024,
                        "batching_support": "bounded",
                        "transform_support": "bounded",
                        "claim_boundary": "functional_non_isolated_only",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    evidence = torch_bridge._load_torch_live_overlay_evidence(path)

    assert isinstance(evidence, PhaseTorchLiveOverlayEvidence)
    assert evidence.torch_version == "2.test"
    assert evidence.to_dict()["artifact_id"] == "torch-functional-local"


def test_torch_result_json_ready_handles_numpy_and_paths() -> None:
    """Maturity evidence JSON conversion should normalize arrays, scalars, and paths."""

    payload = torch_bridge._json_ready(
        {
            "array": np.array([1.0, 2.0], dtype=np.float64),
            "scalar": np.float64(3.0),
            "tuple": (Path("artifact.json"), np.array([4.0], dtype=np.float64)),
        },
    )

    assert payload == {
        "array": [1.0, 2.0],
        "scalar": 3.0,
        "tuple": ["artifact.json", [4.0]],
    }


def test_torch_result_to_dict_uses_object_serializer() -> None:
    """Maturity evidence serialization should use result ``to_dict`` methods."""

    result = SimpleNamespace(to_dict=lambda: {"route": "custom"})

    assert torch_bridge._result_to_dict(result) == {"route": "custom"}
    assert torch_bridge._result_to_dict("plain") == "plain"
