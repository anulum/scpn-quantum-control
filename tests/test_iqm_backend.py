# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM backend adapter tests
"""Tests for the IQM Qiskit backend adapter."""

from __future__ import annotations

import types
from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from scpn_quantum_control.hardware import backends as be
from scpn_quantum_control.hardware.iqm_backend import (
    IQMBackendConfig,
    IQMQuantumBackend,
    IQMRunResult,
    IQMTargetCompilationError,
    _backend_name,
    _extract_counts,
    _job_id,
    is_iqm_available,
)


class _FakeResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeJob:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def job_id(self) -> str:
        return "fake-iqm-job-1"

    def result(self, timeout: float | None = None) -> _FakeResult:
        assert timeout == 12.5
        return _FakeResult(self._counts)


def _fake_iqm_backend() -> Any:
    """Return a real transpilation target whose ``run`` records the shots.

    A genuine ``GenericBackendV2`` instance is used rather than a subclass, so
    the adapter compiles against an actual coupling map and basis-gate set —
    which is what a real IQM device provides, and what a bare object cannot
    stand in for. Only ``run`` is replaced, because the stub job it returns is
    the one thing a local test cannot get from the real target.

    Returns
    -------
    Any
        The backend. Qiskit ships no type information, so a precise annotation
        here would be a fiction rather than a check.

    """
    backend: Any = GenericBackendV2(num_qubits=5, seed=7)
    backend.name = "fake_garnet"
    backend.received_shots = None

    def _run(circuits: list[QuantumCircuit], *, shots: int) -> _FakeJob:
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 5
        backend.received_shots = shots
        return _FakeJob({"00": 31, "11": 33})

    backend.run = _run
    return backend


class _FakeIQMProvider:
    def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
        self.url = url
        self.quantum_computer = quantum_computer

    def get_backend(self) -> Any:
        return _fake_iqm_backend()


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_is_iqm_available_returns_bool() -> None:
    """The package-level dependency probe always returns a Boolean."""
    assert isinstance(is_iqm_available(), bool)


def test_iqm_config_rejects_unsafe_values() -> None:
    """Execution configuration rejects unsafe budgets and implicit remotes."""
    with pytest.raises(ValueError, match="shots must be positive"):
        IQMBackendConfig(shots=0)
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        IQMBackendConfig(timeout_s=0.0)
    with pytest.raises(ValueError, match="server_url is required"):
        IQMBackendConfig(mode="remote")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shots", True),
        ("shots", 1.5),
        ("shots", "8"),
        ("timeout_s", True),
        ("timeout_s", float("nan")),
        ("timeout_s", float("inf")),
        ("timeout_s", "10"),
        ("optimisation_level", True),
        ("optimisation_level", 1.0),
    ],
)
def test_iqm_config_rejects_coercible_or_nonfinite_budgets(field: str, value: Any) -> None:
    """Resource admission rejects malformed counts and time limits deterministically."""
    with pytest.raises(ValueError, match=field):
        IQMBackendConfig(**{field: value})


@pytest.mark.parametrize("value", [True, -1, 1.5, "64", float("nan")])
@pytest.mark.parametrize("shape", ["method", "raw"])
def test_iqm_counts_reject_lossy_values(value: Any, shape: str) -> None:
    """Neither provider result shape may coerce measurement counts."""
    raw = {"00": value}
    result = (
        _FakeResult(raw)
        if shape == "method"
        else types.SimpleNamespace(
            results=[types.SimpleNamespace(data=types.SimpleNamespace(counts=raw))]
        )
    )
    with pytest.raises(ValueError, match="count"):
        _extract_counts(result)


@pytest.mark.parametrize("raw", [{1: 2}, {"": 2}, object()])
def test_iqm_counts_reject_malformed_maps(raw: Any) -> None:
    """Reject non-mappings and labels that would change through string coercion."""
    result = types.SimpleNamespace(get_counts=lambda: raw)
    with pytest.raises(ValueError, match="count"):
        _extract_counts(result)


def test_iqm_run_uses_real_target_and_local_simulator() -> None:
    """Targeted Bell preparation survives actual compilation and local execution."""
    target = GenericBackendV2(
        num_qubits=3,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        noise_info=False,
        seed=7,
    )
    resolutions = 0

    class Adapter(IQMQuantumBackend):
        def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
            nonlocal resolutions
            resolutions += 1
            return target

    result = Adapter().run_counts(_bell_circuit(), IQMBackendConfig(shots=64, timeout_s=12.5))
    assert resolutions == 1
    assert set(result.counts) <= {"00", "11"}
    assert sum(result.counts.values()) == 64
    assert result.metadata["compiled_for"] == target.name
    assert result.metadata["n_qubits"] == 3


def test_failed_target_compilation_never_submits() -> None:
    """Run entry point refuses an unmappable target before crossing its run boundary."""
    calls: list[str] = []

    class InvalidBackend:
        name = "invalid-target"

        def run(self, *args: Any, **kwargs: Any) -> None:
            calls.append("run")

    class Adapter(IQMQuantumBackend):
        def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
            calls.append("resolve")
            return InvalidBackend()

    with pytest.raises(IQMTargetCompilationError):
        Adapter().run_counts(_bell_circuit())
    assert calls == ["resolve"]


@pytest.mark.parametrize("count", [True, 1.5])
def test_run_counts_refuses_invalid_provider_measurement(count: Any) -> None:
    """A malformed provider measurement cannot become a successful public result."""
    target = _fake_iqm_backend()
    submissions = 0

    def run(circuits: list[QuantumCircuit], *, shots: int) -> _FakeJob:
        nonlocal submissions
        submissions += 1
        assert circuits[0].layout is not None
        assert shots == 64
        return _FakeJob({"00": count})

    target.run = run

    class Adapter(IQMQuantumBackend):
        def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
            return target

    with pytest.raises(ValueError, match="counts"):
        Adapter().run_counts(_bell_circuit(), IQMBackendConfig(shots=64, timeout_s=12.5))
    assert submissions == 1


def test_iqm_descriptor_is_registered_and_approval_gated() -> None:
    """The registry exposes IQM submission only behind explicit approval."""
    descriptor = be.describe_backend("iqm")
    assert descriptor.name == "iqm"
    assert descriptor.provider == "iqm"
    assert descriptor.can_submit is True
    assert descriptor.submit_requires_approval is True
    assert descriptor.supports_shots is True
    assert "kuramoto_xy" in descriptor.workloads
    assert "superconducting_qpu" in descriptor.capabilities


def test_iqm_backend_requires_dependency_for_backend_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend resolution fails clearly without the optional IQM client."""

    def missing_module(name: str) -> Any:
        raise ModuleNotFoundError(name)

    backend = IQMQuantumBackend(import_module=missing_module)
    assert backend.is_available() is False
    with pytest.raises(ImportError, match=r"iqm-client\[qiskit\]"):
        backend.resolve_backend(IQMBackendConfig(mode="fake", fake_backend="garnet"))


def test_iqm_fake_backend_resolution_and_count_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local fake backend resolves and returns bounded circuit counts."""
    fake_module = types.SimpleNamespace(IQMFakeGarnet=_fake_iqm_backend)

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.fake_backends.fake_garnet":
            return fake_module
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=_FakeIQMProvider)
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=import_module)
    result = adapter.run_counts(
        _bell_circuit(),
        IQMBackendConfig(mode="fake", fake_backend="garnet", shots=64, timeout_s=12.5),
    )

    assert isinstance(result, IQMRunResult)
    assert result.job_id == "fake-iqm-job-1"
    assert result.backend_name == "fake_garnet"
    assert result.counts == {"00": 31, "11": 33}
    assert result.metadata["mode"] == "fake"
    assert result.metadata["shots"] == 64


def test_iqm_remote_backend_uses_provider_url_and_quantum_computer() -> None:
    """Remote resolution forwards only the explicit URL and computer name."""
    captured: dict[str, str | None] = {}

    class Provider(_FakeIQMProvider):
        def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
            captured["url"] = url
            captured["quantum_computer"] = quantum_computer
            super().__init__(url, quantum_computer=quantum_computer)

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=Provider)
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=import_module)
    backend = adapter.resolve_backend(
        IQMBackendConfig(
            mode="remote",
            server_url="https://example.iqm.invalid",
            quantum_computer="garnet",
        )
    )

    assert backend.name == "fake_garnet"
    assert captured == {
        "url": "https://example.iqm.invalid",
        "quantum_computer": "garnet",
    }


def test_iqm_config_rejects_bad_mode_and_optimisation_level() -> None:
    """Mode and optimisation level are validated at construction."""
    with pytest.raises(ValueError, match="mode must be"):
        IQMBackendConfig(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="optimisation_level must be"):
        IQMBackendConfig(optimisation_level=9)


def test_iqm_config_rejects_unknown_fake_backend() -> None:
    """An unknown fake backend name is rejected with the known set listed."""
    with pytest.raises(ValueError, match="unknown IQM fake backend"):
        IQMBackendConfig(mode="fake", fake_backend="nonsuch")


def test_is_iqm_available_true_when_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful provider import reports availability."""
    monkeypatch.setattr(
        "scpn_quantum_control.hardware.iqm_backend.importlib.import_module",
        lambda _name: object(),
    )
    assert is_iqm_available() is True


def test_is_available_true_when_import_succeeds() -> None:
    """The adapter reports availability when the provider import succeeds."""
    adapter = IQMQuantumBackend(import_module=lambda _name: object())
    assert adapter.is_available() is True


def test_resolve_fake_backend_rejects_unknown_key() -> None:
    """Resolving an unknown fake-backend key fails closed."""
    adapter = IQMQuantumBackend(import_module=lambda _name: object())
    with pytest.raises(ValueError, match="unknown IQM fake backend"):
        adapter._resolve_fake_backend("nonsuch")


def test_resolve_remote_backend_requires_dependency() -> None:
    """Remote resolution without the IQM client raises a clear ImportError."""

    def missing_module(name: str) -> Any:
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=missing_module)
    with pytest.raises(ImportError, match=r"iqm-client\[qiskit\]"):
        adapter.resolve_backend(
            IQMBackendConfig(mode="remote", server_url="https://example.iqm.invalid")
        )


def test_resolve_remote_backend_uses_backend_method_fallback() -> None:
    """A provider without get_backend falls back to backend()."""

    class ProviderWithoutGetBackend:
        def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
            self.url = url

        def backend(self) -> Any:
            return _fake_iqm_backend()

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=ProviderWithoutGetBackend)
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=import_module)
    backend = adapter.resolve_backend(
        IQMBackendConfig(mode="remote", server_url="https://example.iqm.invalid")
    )
    assert backend.name == "fake_garnet"


class _ArgCountsResult:
    """A result whose get_counts requires the legacy experiment index argument."""

    def get_counts(self, index: int) -> dict[str, int]:
        assert index == 0
        return {"01": 5}


class _ListCountsResult:
    """A result returning a list of count maps (multi-experiment shape)."""

    def __init__(self, maps: list[dict[str, int]]) -> None:
        self._maps = maps

    def get_counts(self) -> list[dict[str, int]]:
        return list(self._maps)


def test_extract_counts_handles_indexed_get_counts() -> None:
    """A get_counts requiring an index is retried with experiment 0."""
    assert _extract_counts(_ArgCountsResult()) == {"01": 5}


def test_extract_counts_unwraps_single_element_list() -> None:
    """A single-element list of count maps is unwrapped."""
    assert _extract_counts(_ListCountsResult([{"00": 7}])) == {"00": 7}


def test_extract_counts_rejects_multiple_count_maps() -> None:
    """Multiple count maps for a single-circuit run fail closed."""
    with pytest.raises(RuntimeError, match="multiple count maps"):
        _extract_counts(_ListCountsResult([{"0": 1}, {"1": 1}]))


def test_extract_counts_reads_results_data_counts() -> None:
    """A result exposing results[0].data.counts is decoded."""
    data = types.SimpleNamespace(counts={"11": 9})
    result = types.SimpleNamespace(results=[types.SimpleNamespace(data=data)])
    assert _extract_counts(result) == {"11": 9}


def test_extract_counts_rejects_non_mapping_results_counts() -> None:
    """Reject a results-data counts field that is not a mapping."""
    data = types.SimpleNamespace(counts=[("11", 9)])
    result = types.SimpleNamespace(results=[types.SimpleNamespace(data=data)])

    with pytest.raises(RuntimeError, match="Could not extract IQM counts"):
        _extract_counts(result)


def test_extract_counts_fails_when_unreadable() -> None:
    """A result with no recognisable counts shape fails closed."""
    with pytest.raises(RuntimeError, match="Could not extract IQM counts"):
        _extract_counts(types.SimpleNamespace())


def test_backend_name_prefers_callable_then_falls_back_to_type() -> None:
    """Backend name resolution handles callable, attribute, and type fallback."""
    callable_named = types.SimpleNamespace(name=lambda: "garnet-rt")
    assert _backend_name(callable_named) == "garnet-rt"

    class Nameless:
        name = None

    assert _backend_name(Nameless()) == "Nameless"


def test_job_id_handles_attribute_and_missing() -> None:
    """Job-id resolution handles a string attribute and a missing id."""
    assert _job_id(types.SimpleNamespace(job_id="raw-id")) == "raw-id"
    assert _job_id(types.SimpleNamespace(job_id=None)) == "iqm_job_id_unavailable"


def test_transpilation_targets_the_backend_and_refuses_a_non_target() -> None:
    """A backend that cannot serve as a target is rejected, not compiled around."""

    class _NotABackend:
        name = "not_a_target"

    class _Adapter(IQMQuantumBackend):
        def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
            return _NotABackend()

    with pytest.raises(IQMTargetCompilationError, match="not_a_target"):
        _Adapter().transpile_circuit(_bell_circuit(), IQMBackendConfig(mode="fake"))


def test_run_counts_compiles_for_the_backend_it_submits_to() -> None:
    """Compile and submit must share one resolution, not two."""
    resolutions: list[Any] = []

    class _Adapter(IQMQuantumBackend):
        def resolve_backend(self, config: IQMBackendConfig | None = None) -> Any:
            backend = _fake_iqm_backend()
            resolutions.append(backend)
            return backend

    adapter = _Adapter()
    result = adapter.run_counts(
        _bell_circuit(),
        IQMBackendConfig(mode="fake", fake_backend="garnet", shots=64, timeout_s=12.5),
    )

    assert len(resolutions) == 1, "compile and submit resolved the backend separately"
    assert resolutions[0].received_shots == 64
    assert result.metadata["compiled_for"] == "fake_garnet"
    assert result.backend_name == "fake_garnet"


def test_compiled_circuit_is_mapped_onto_the_target_device() -> None:
    """The submitted circuit carries the target's layout, not a logical one."""
    adapter = IQMQuantumBackend()
    backend = _fake_iqm_backend()
    compiled = adapter.transpile_circuit(
        _bell_circuit(), IQMBackendConfig(mode="fake"), backend=backend
    )

    assert compiled.layout is not None
    assert compiled.num_qubits == backend.num_qubits


@pytest.mark.parametrize("mode", ["fake", "remote"])
def test_missing_provider_package_fails_closed_on_every_resolution_path(mode: str) -> None:
    """Both resolution paths refuse with installation guidance, not a raw import error.

    The optional dependency is deliberately not installed alongside the main
    Qiskit floor, so the adapter must translate the absence into an actionable
    ``ImportError`` on every path that needs it rather than letting a bare
    ``ModuleNotFoundError`` escape.
    """

    def import_module(name: str) -> Any:
        raise ModuleNotFoundError(name)

    adapter = IQMQuantumBackend(import_module=import_module)
    assert adapter.is_available() is False

    config = (
        IQMBackendConfig(mode="fake", fake_backend="garnet")
        if mode == "fake"
        else IQMBackendConfig(mode="remote", server_url="https://example.iqm.invalid")
    )
    with pytest.raises(ImportError, match=r"iqm-client\[qiskit\] is required") as raised:
        adapter.resolve_backend(config)

    assert ".venv-iqm" in str(raised.value)
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_target_compilation_error_is_importable_from_the_package() -> None:
    """Callers must be able to catch the refusal without a private import."""
    from scpn_quantum_control.hardware import IQMTargetCompilationError as exported

    assert exported is IQMTargetCompilationError
    assert issubclass(exported, RuntimeError)
