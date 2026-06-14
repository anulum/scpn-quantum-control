# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — external differentiable comparison rows.
"""External framework comparison rows for bounded Phase-QNode claims."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..phase.jax_bridge import is_phase_jax_available
from ..phase.pennylane_bridge import is_phase_pennylane_available
from ..phase.tensorflow_bridge import is_phase_tensorflow_available
from ..phase.torch_bridge import is_phase_torch_available

ComparisonStatus = Literal["success", "hard_gap"]


@dataclass(frozen=True)
class ExternalComparisonRow:
    """One external framework/compiler comparison row."""

    case_id: str
    backend: str
    status: ComparisonStatus
    failure_class: str | None
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    memory_peak_bytes: int | None
    batching_support: str
    transform_support: str
    dtype: str
    device: str
    source_of_truth: str
    setup_instructions: str | None
    claim_boundary: str
    toolchain: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        if self.source_of_truth != "scpn_reference":
            raise ValueError("source_of_truth must be scpn_reference")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.value_error is None
                or self.gradient_error is None
                or self.runtime_seconds is None
                or self.memory_peak_bytes is None
            ):
                raise ValueError("success rows require numeric evidence and no failure class")
            if self.value_error < 0 or self.gradient_error < 0:
                raise ValueError("success row errors must be non-negative")
        if self.status == "hard_gap" and (not self.failure_class or not self.setup_instructions):
            raise ValueError("hard_gap rows require failure_class and setup_instructions")
        if self.toolchain is not None and (
            any(not isinstance(key, str) or not key for key in self.toolchain)
            or any(not isinstance(value, str) or not value for value in self.toolchain.values())
        ):
            raise ValueError("toolchain metadata must map non-empty strings to non-empty strings")

    @property
    def artifact_fields_ready(self) -> bool:
        """Return whether this row is serializable as an evidence artefact."""

        return bool(self.case_id and self.backend and self.status and self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""

        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "memory_peak_bytes": self.memory_peak_bytes,
            "batching_support": self.batching_support,
            "transform_support": self.transform_support,
            "dtype": self.dtype,
            "device": self.device,
            "source_of_truth": self.source_of_truth,
            "setup_instructions": self.setup_instructions,
            "claim_boundary": self.claim_boundary,
            "toolchain": dict(self.toolchain) if self.toolchain is not None else None,
        }


def run_differentiable_external_comparison_suite() -> tuple[ExternalComparisonRow, ...]:
    """Run or classify optional external comparison rows.

    The SCPN analytic parameter-shift reference remains the source of truth.
    Missing optional tooling is recorded as hard-gap evidence instead of being
    silently omitted.
    """

    rows: list[ExternalComparisonRow] = []
    for backend, available, runner, batching, transform, setup in (
        (
            "jax",
            is_phase_jax_available(),
            _run_jax_reference,
            "vmap",
            "value_and_grad",
            "Install the CPU overlay wheel jax[cpu].",
        ),
        (
            "pytorch",
            is_phase_torch_available(),
            _run_pytorch_reference,
            "torch.func.vmap",
            "torch.func.grad/jacrev",
            "Install the CPU overlay wheel torch.",
        ),
        (
            "tensorflow",
            is_phase_tensorflow_available(),
            _run_tensorflow_reference,
            "vectorized_map",
            "GradientTape",
            "Install the CPU overlay wheel tensorflow-cpu.",
        ),
        (
            "pennylane",
            is_phase_pennylane_available(),
            _run_pennylane_reference,
            "not_native",
            "QNode",
            "Install the CPU overlay wheel pennylane.",
        ),
    ):
        rows.append(
            _framework_row(backend, runner, batching, transform, setup)
            if available
            else _dependency_gap_row(backend, batching, transform, setup)
        )
    rows.append(
        _enzyme_row()
        if _enzyme_runner_configured()
        else _dependency_gap_row(
            "enzyme",
            "not_evaluated",
            "LLVM Enzyme",
            "Install LLVM/Enzyme tooling and configure the Enzyme runner.",
        )
    )
    return tuple(rows)


def _framework_row(
    backend: str,
    runner: Any,
    batching_support: str,
    transform_support: str,
    setup: str,
) -> ExternalComparisonRow:
    values = np.array([0.2, -0.4], dtype=np.float64)
    reference_value = _bounded_phase_objective(values)
    reference_gradient = _bounded_phase_gradient(values)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        external_value, external_gradient = runner(values)
    except ImportError:
        tracemalloc.stop()
        return _dependency_gap_row(backend, batching_support, transform_support, setup)
    except Exception as exc:  # pragma: no cover - defensive optional boundary
        tracemalloc.stop()
        return _runtime_gap_row(backend, batching_support, transform_support, str(exc))
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="success",
        failure_class=None,
        value_error=abs(reference_value - external_value),
        gradient_error=float(np.max(np.abs(reference_gradient - external_gradient))),
        runtime_seconds=runtime,
        memory_peak_bytes=max(int(peak), int(values.nbytes + reference_gradient.nbytes)),
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary=(
            "Bounded CPU external comparison against the SCPN reference; no provider, "
            "QPU, GPU, or production performance claim."
        ),
    )


def _enzyme_row() -> ExternalComparisonRow:
    if not _enzyme_runner_configured():
        return _dependency_gap_row(
            "enzyme",
            "not_evaluated",
            "LLVM Enzyme",
            "Configure SCPN_ENZYME_RUNNER to an executable runner that emits value and gradient JSON.",
        )
    values = np.array([0.2, -0.4], dtype=np.float64)
    reference_value = _bounded_phase_objective(values)
    reference_gradient = _bounded_phase_gradient(values)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        external_value, external_gradient, toolchain = _run_enzyme_reference(values)
    except TimeoutError as exc:
        tracemalloc.stop()
        return _runtime_gap_row("enzyme", "not_supported", "LLVM Enzyme runner", str(exc))
    except (RuntimeError, ValueError) as exc:
        tracemalloc.stop()
        return _runtime_gap_row("enzyme", "not_supported", "LLVM Enzyme runner", str(exc))
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    value_error = abs(reference_value - external_value)
    gradient_error = float(np.max(np.abs(reference_gradient - external_gradient)))
    if value_error > 1.0e-8 or gradient_error > 1.0e-8:
        return ExternalComparisonRow(
            case_id="bounded_phase_objective",
            backend="enzyme",
            status="hard_gap",
            failure_class="correctness_mismatch",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            memory_peak_bytes=None,
            batching_support="not_supported",
            transform_support="LLVM Enzyme runner",
            dtype="float64",
            device="cpu",
            source_of_truth="scpn_reference",
            setup_instructions=(
                "Configured Enzyme runner output did not match the SCPN reference "
                f"(value_error={value_error:.3e}, gradient_error={gradient_error:.3e})."
            ),
            claim_boundary="Correctness hard gap only; no hidden success or promoted claim.",
            toolchain=toolchain,
        )
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="enzyme",
        status="success",
        failure_class=None,
        value_error=value_error,
        gradient_error=gradient_error,
        runtime_seconds=runtime,
        memory_peak_bytes=max(int(peak), int(values.nbytes + reference_gradient.nbytes)),
        batching_support="not_supported",
        transform_support="LLVM Enzyme runner",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary=(
            "Bounded CPU LLVM/Enzyme runner comparison against the SCPN reference; "
            "no provider, QPU, GPU, arbitrary-program AD, or production performance claim."
        ),
        toolchain=toolchain,
    )


def _dependency_gap_row(
    backend: str,
    batching_support: str,
    transform_support: str,
    setup: str,
) -> ExternalComparisonRow:
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=setup,
        claim_boundary="Dependency hard gap only; no hidden success or promoted claim.",
    )


def _runtime_gap_row(
    backend: str,
    batching_support: str,
    transform_support: str,
    reason: str,
) -> ExternalComparisonRow:
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="hard_gap",
        failure_class="runtime_error",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support=batching_support,
        transform_support=transform_support,
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=reason,
        claim_boundary="Runtime comparison gap only; no hidden success or promoted claim.",
    )


def _bounded_phase_objective(values: NDArray[np.float64]) -> float:
    return float(math.cos(values[0]) + 0.25 * math.sin(values[1]))


def _bounded_phase_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([-math.sin(values[0]), 0.25 * math.cos(values[1])], dtype=np.float64)


def _run_jax_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def objective(x):
        return jnp.cos(x[0]) + 0.25 * jnp.sin(x[1])

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(values, dtype=jnp.float64))
    return float(value), np.asarray(gradient, dtype=np.float64)


def _run_pytorch_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import torch

    def objective(x):
        return torch.cos(x[0]) + 0.25 * torch.sin(x[1])

    tensor = torch.tensor(values, dtype=torch.float64)
    gradient = torch.func.grad(objective)(tensor)
    return float(objective(tensor).detach().cpu().item()), gradient.detach().cpu().numpy()


def _run_tensorflow_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import tensorflow as tf

    tensor = tf.Variable(values, dtype=tf.float64)
    with tf.GradientTape() as tape:
        value = tf.cos(tensor[0]) + tf.constant(0.25, dtype=tf.float64) * tf.sin(tensor[1])
    gradient = tape.gradient(value, tensor)
    if gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    return float(value.numpy()), np.asarray(gradient.numpy(), dtype=np.float64)


def _run_pennylane_reference(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    import pennylane as qml

    try:
        from pennylane import numpy as pnp
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("PennyLane NumPy interface is unavailable") from exc

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    def objective(x):
        z0, x1 = circuit(x)
        return z0 + 0.25 * x1

    params = pnp.array(values, requires_grad=True)
    value = objective(params)
    gradient = qml.grad(objective)(params)
    return float(value), np.asarray(gradient, dtype=np.float64)


def _run_enzyme_reference(
    values: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], dict[str, str]]:
    runner = os.environ.get("SCPN_ENZYME_RUNNER")
    if not runner:
        raise RuntimeError("SCPN_ENZYME_RUNNER is not configured")
    payload = json.dumps(
        {
            "schema": "scpn_qc_enzyme_runner_request_v1",
            "case_id": "bounded_phase_objective",
            "values": values.tolist(),
            "objective": "cos(x0)+0.25*sin(x1)",
            "gradient_contract": ["-sin(x0)", "0.25*cos(x1)"],
            "dtype": "float64",
        },
        sort_keys=True,
    )
    try:
        completed = subprocess.run(
            [runner],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
            timeout=_enzyme_runner_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Configured Enzyme runner timed out") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr"
        raise RuntimeError(f"Configured Enzyme runner failed: {stderr}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Configured Enzyme runner did not emit valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Configured Enzyme runner JSON must be an object")
    value = _as_finite_scalar("Enzyme runner value", result.get("value"))
    gradient = _as_gradient_vector("Enzyme runner gradient", result.get("gradient"), values.size)
    toolchain = _as_toolchain_metadata(result.get("toolchain"))
    return value, gradient, toolchain


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_gradient_vector(name: str, value: object, width: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must contain finite real numeric values")
    gradient = np.asarray(value, dtype=np.float64)
    if gradient.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {gradient.shape}")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return gradient.astype(np.float64, copy=True)


def _as_toolchain_metadata(value: object) -> dict[str, str]:
    if value is None:
        return {"enzyme": "configured-runner", "llvm": "configured-runner"}
    if not isinstance(value, dict):
        raise ValueError("Enzyme runner toolchain metadata must be an object")
    metadata = {str(key): str(item) for key, item in value.items()}
    if any(not key or not item for key, item in metadata.items()):
        raise ValueError("Enzyme runner toolchain metadata cannot contain empty keys or values")
    return metadata


def _enzyme_runner_timeout_seconds() -> float:
    raw = os.environ.get("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS", "10")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS must be numeric") from exc
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS must be positive")
    return timeout


def _enzyme_tooling_available() -> bool:
    configured_plugin = os.environ.get("ENZYME_LLVM_PLUGIN")
    return shutil.which("enzyme") is not None or bool(
        configured_plugin and os.path.exists(configured_plugin)
    )


def _enzyme_runner_configured() -> bool:
    runner = os.environ.get("SCPN_ENZYME_RUNNER")
    return bool(runner and os.path.exists(runner) and _enzyme_tooling_available())


__all__ = [
    "ExternalComparisonRow",
    "run_differentiable_external_comparison_suite",
]
