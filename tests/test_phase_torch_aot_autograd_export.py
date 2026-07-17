# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch aot autograd export tests
# scpn-quantum-control -- PyTorch AOTAutograd export tests
"""AOTAutograd graph persistence tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.torch_aot_autograd_export as aot_export
from scpn_quantum_control.phase import (
    TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA,
    PhaseTorchAOTAutogradExportResult,
    run_torch_aot_autograd_export_audit,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torch._functorch.aot_autograd")


def _features() -> NDArray[np.float64]:
    """Return a deterministic two-parameter bounded phase-QNN fixture."""
    return np.array(
        [
            [0.0, 1.0],
            [np.pi / 2.0, -0.4],
            [np.pi, 0.25],
            [3.0 * np.pi / 2.0, 0.75],
        ],
        dtype=np.float64,
    )


def _labels() -> NDArray[np.float64]:
    """Return deterministic labels for the AOTAutograd fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic trainable parameters for the AOTAutograd fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_aot_autograd_export_persists_and_replays_backward_graph(
    tmp_path: Path,
) -> None:
    """The audit should persist AOT forward/backward graphs and replay gradients."""
    artifact_dir = tmp_path / "aot_autograd"

    result = run_torch_aot_autograd_export_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        artifact_dir=artifact_dir,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchAOTAutogradExportResult)
    assert result.passed
    assert result.matrix_schema == TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA
    assert result.artifact_dir == str(artifact_dir)
    assert result.route_status("aot_autograd_forward_backward_capture") == "passed"
    assert result.route_status("aot_autograd_graph_file_round_trip") == "passed"
    assert result.route_status("loaded_backward_gradient_replay") == "passed"
    assert result.route_status("cross_runtime_aot_autograd_execution") == "blocked"
    assert result.route_status("cuda_aot_autograd_execution") == "blocked"
    assert result.route_status("dynamic_shape_aot_autograd_export") == "blocked"
    assert result.forward_graph.kind == "forward"
    assert result.backward_graph.kind == "backward"
    assert result.forward_graph.graph_node_count > 0
    assert result.backward_graph.graph_node_count > result.forward_graph.graph_node_count
    assert "aten.cos.default" in result.forward_graph.operation_names
    assert "aten.sin.default" in result.backward_graph.operation_names
    assert Path(result.forward_graph.artifact_path).is_file()
    assert Path(result.backward_graph.artifact_path).is_file()
    assert result.forward_graph.artifact_size_bytes > 0
    assert result.backward_graph.artifact_size_bytes > 0
    assert len(result.forward_graph.artifact_sha256) == 64
    assert len(result.backward_graph.artifact_sha256) == 64
    assert result.loss_error <= result.tolerance
    assert result.compiled_gradient_error <= result.tolerance
    assert result.loaded_gradient_error <= result.tolerance
    assert result.gradient_shape == (2,)
    assert result.aot_autograd_fx_persistence_claim is True
    assert result.cross_runtime_claim is False
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    forward_graph = cast(dict[str, Any], payload["forward_graph"])
    backward_graph = cast(dict[str, Any], payload["backward_graph"])
    assert routes["loaded_backward_gradient_replay"]["status"] == "passed"
    assert routes["cross_runtime_aot_autograd_execution"]["status"] == "blocked"
    assert forward_graph["artifact_sha256"] == result.forward_graph.artifact_sha256
    assert backward_graph["kind"] == "backward"
    assert "AOTAutograd FX" in str(payload["claim_boundary"])


def test_torch_aot_autograd_export_rejects_file_artifact_dir(tmp_path: Path) -> None:
    """Artifact directory validation should fail closed for regular files."""
    artifact_file = tmp_path / "not_a_directory"
    artifact_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_dir must be a directory"):
        run_torch_aot_autograd_export_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            artifact_dir=artifact_file,
        )


def test_torch_aot_autograd_export_rejects_missing_parent_path(tmp_path: Path) -> None:
    """The audit should not create missing parent paths implicitly."""
    with pytest.raises(ValueError, match="artifact_dir parent"):
        run_torch_aot_autograd_export_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            artifact_dir=tmp_path / "missing" / "aot",
        )


def test_torch_aot_autograd_export_rejects_unknown_route(tmp_path: Path) -> None:
    """Route lookup should fail closed for unknown AOTAutograd rows."""
    result = run_torch_aot_autograd_export_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        artifact_dir=tmp_path / "aot_autograd",
    )

    with pytest.raises(KeyError, match="unknown PyTorch AOTAutograd export route"):
        result.route_status("missing")


def test_torch_aot_autograd_export_rejects_shape_mismatch(tmp_path: Path) -> None:
    """Shape checks should use the production validation path."""
    with pytest.raises(ValueError, match="initial_params"):
        run_torch_aot_autograd_export_audit(
            features=_features(),
            labels=_labels(),
            initial_params=np.array([0.25], dtype=np.float64),
            artifact_dir=tmp_path / "aot_autograd",
        )


def test_torch_aot_autograd_export_helper_failures_are_targeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private fail-closed helpers should emit targeted errors."""
    real_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "torch._functorch.aot_autograd":
            raise ImportError("missing AOTAutograd")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(RuntimeError, match="AOTAutograd is unavailable"):
        aot_export._require_aot_autograd()
    monkeypatch.setattr(builtins, "__import__", real_import)

    with pytest.raises(RuntimeError, match="requires_grad_"):
        aot_export._trainable_tensor(
            _FakeTorch(_NoGradTensor()),
            np.array([0.1], dtype=np.float64),
        )
    with pytest.raises(RuntimeError, match="forward"):
        aot_export._graph_forward(object())
    with pytest.raises(RuntimeError, match="did not capture"):
        aot_export._require_captured_graph({}, "backward")
    with pytest.raises(RuntimeError, match="torch.save"):
        aot_export._torch_save(object(), object(), Path("unused.pt"))
    with pytest.raises(RuntimeError, match="torch.load"):
        aot_export._torch_load_graph(object(), Path("unused.pt"), expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="must be a tuple"):
        aot_export._as_tuple("not-a-tuple", name="loaded AOTAutograd output")


def test_torch_aot_autograd_export_helper_edge_routes() -> None:
    """Helper edge cases should stay fail-closed and JSON-ready."""
    trainable = aot_export._trainable_tensor(
        _FakeTorch(_NoCloneTensor()),
        np.array([0.1], dtype=np.float64),
    )
    assert isinstance(trainable, _NoCloneTensor)
    assert trainable.requires_grad_value is True
    assert aot_export._graph_operation_names(object()) == ()
    assert aot_export._max_abs_error(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    ) == float("inf")
    assert (
        aot_export._max_abs_error(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        == 0.0
    )

    with pytest.raises(RuntimeError, match="forward graph returned no outputs"):
        aot_export._replay_loaded_aot_graphs(
            torch_module=_ReplayTorch(with_ones_like=True),
            forward_graph=_CallableGraph(()),
            backward_graph=_CallableGraph((np.array([0.0], dtype=np.float64),)),
            parameter_values=np.array([0.1], dtype=np.float64),
            feature_matrix=np.array([[0.0]], dtype=np.float64),
            label_vector=np.array([0.0], dtype=np.float64),
        )
    with pytest.raises(RuntimeError, match="torch.ones_like"):
        aot_export._replay_loaded_aot_graphs(
            torch_module=_ReplayTorch(with_ones_like=False),
            forward_graph=_CallableGraph((np.array(1.0, dtype=np.float64), "saved")),
            backward_graph=_CallableGraph((np.array([0.0], dtype=np.float64),)),
            parameter_values=np.array([0.1], dtype=np.float64),
            feature_matrix=np.array([[0.0]], dtype=np.float64),
            label_vector=np.array([0.0], dtype=np.float64),
        )
    with pytest.raises(RuntimeError, match="backward graph returned no outputs"):
        aot_export._replay_loaded_aot_graphs(
            torch_module=_ReplayTorch(with_ones_like=True),
            forward_graph=_CallableGraph((np.array(1.0, dtype=np.float64), "saved")),
            backward_graph=_CallableGraph(()),
            parameter_values=np.array([0.1], dtype=np.float64),
            feature_matrix=np.array([[0.0]], dtype=np.float64),
            label_vector=np.array([0.0], dtype=np.float64),
        )

    routes = aot_export._aot_autograd_routes(
        forward_record=_empty_graph_record("forward"),
        backward_record=_empty_graph_record("backward"),
        compiled_loss_error=1.0,
        loaded_loss_error=1.0,
        compiled_gradient_error=1.0,
        loaded_gradient_error=1.0,
        tolerance=0.0,
    )
    route_statuses = {route.name: route.status for route in routes}
    assert route_statuses["aot_autograd_forward_backward_capture"] == "failed"
    assert route_statuses["aot_autograd_graph_file_round_trip"] == "failed"
    assert route_statuses["loaded_backward_gradient_replay"] == "failed"


class _NoCloneTensor:
    """Minimal tensor without detach/clone methods for branch coverage."""

    def __init__(self) -> None:
        self.requires_grad_value = False

    def requires_grad_(self, value: bool) -> _NoCloneTensor:
        """Record requested gradient tracking and return self."""
        self.requires_grad_value = value
        return self


class _NoGradTensor:
    """Minimal tensor without requires_grad_."""


class _FakeTorch:
    """Minimal torch-like tensor factory for helper tests."""

    float64 = np.float64

    def __init__(self, tensor: object) -> None:
        self._tensor = tensor

    def as_tensor(self, values: object, dtype: object | None = None) -> object:
        """Return the configured tensor while matching the torch signature."""
        del values, dtype
        return self._tensor


class _CallableGraph:
    """Callable graph stub returning a configured output."""

    def __init__(self, output: object) -> None:
        self._output = output

    def __call__(self, *args: object) -> object:
        """Return the configured output."""
        del args
        return self._output


class _ReplayTorch:
    """Minimal torch-like replay helper for loaded AOT graph tests."""

    float64 = np.float64

    def __init__(self, *, with_ones_like: bool) -> None:
        self._with_ones_like = with_ones_like

    def as_tensor(self, values: object, dtype: object | None = None) -> object:
        """Return values unchanged while matching the torch signature."""
        del dtype
        return values

    @property
    def ones_like(self) -> object:
        """Return a torch.ones_like stand-in only when enabled."""
        if not self._with_ones_like:
            return None

        def _ones_like(_value: object) -> NDArray[np.float64]:
            return np.array(1.0, dtype=np.float64)

        return _ones_like


def _empty_graph_record(kind: str) -> aot_export.PhaseTorchAOTAutogradGraphRecord:
    """Return an intentionally failing graph record."""
    return aot_export.PhaseTorchAOTAutogradGraphRecord(
        kind=kind,
        artifact_path=f"{kind}.pt",
        artifact_size_bytes=0,
        artifact_sha256="bad",
        graph_node_count=0,
        graph_module_type="GraphModule",
        graph_code_sha256="bad",
        operation_names=(),
    )


def test_torch_save_returns_the_artifact_sha256(tmp_path: Path) -> None:
    """The save helper returns the digest of the bytes it wrote."""
    import hashlib

    path = tmp_path / "artifact.pt"
    digest = aot_export._torch_save(torch, torch.tensor([1.0, 2.0]), path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_torch_load_graph_verifies_the_digest_gate(tmp_path: Path) -> None:
    """A matching digest loads; the gate never reaches torch.load otherwise."""
    path = tmp_path / "artifact.pt"
    digest = aot_export._torch_save(torch, torch.tensor([3.0]), path)
    loaded = aot_export._torch_load_graph(torch, path, expected_sha256=digest)
    assert float(loaded[0]) == 3.0


def test_torch_load_graph_fails_closed_on_tampered_artifact(tmp_path: Path) -> None:
    """Tampered bytes must raise before any deserialisation happens."""
    path = tmp_path / "artifact.pt"
    digest = aot_export._torch_save(torch, torch.tensor([4.0]), path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 gate"):
        aot_export._torch_load_graph(torch, path, expected_sha256=digest)
