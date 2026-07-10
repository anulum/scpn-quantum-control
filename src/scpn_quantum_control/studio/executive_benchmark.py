# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive benchmark handler
"""The ``benchmark`` executive action handler — native construction speedup.

The simulated ``benchmark`` verb measures the wall-clock construction time of
the dense XY Hamiltonian for a bounded ``K_nm``/``omega`` network on the
requested backend (the native Rust PyO3 kernel or the pure-numpy reference),
parity-checks the native operator against the numpy reference, and summarises
the committed tier-benchmark databank
(:mod:`scpn_quantum_control.studio.benchmark_databank_bundle`).

The claim boundary is deliberately narrow. Wall-clock timings on this host are
*opportunistic local measurements on a shared workstation* — they swing with CPU
pinning, cache state, and host load — so every sealed record carries
``production_claim_allowed: False`` and its timing caveat verbatim. What *is*
reproducible is the parity verdict (the native operator matches the numpy
reference to tolerance) and the committed databank summary; the reproduction
script asserts exactly those and re-prints fresh timings without asserting them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from oscillatools.accel.rust_import import optional_rust_engine

from .benchmark_databank_bundle import build_benchmark_databank_bundle
from .evidence_bundle import validate_bundle
from .executive import (
    ActionHandler,
    ExecutionPlan,
    ExecutionResult,
    ExecutiveRequest,
    GeneratedScript,
    VerbContract,
    build_generated_script,
)
from .verbs import BENCHMARK_DATABANK_SCHEMA, NATIVE_SPEEDUP_SCHEMA

BENCHMARK_VERB: Final[str] = "benchmark"
_DEFAULT_BACKEND: Final[str] = "rust"
_MAX_NODES: Final[int] = 10
_MAX_REPEATS: Final[int] = 32
_MAX_WARMUP: Final[int] = 8
_PARITY_TOLERANCE: Final[float] = 1e-9

BENCHMARK_CLAIM_BOUNDARY: Final[str] = (
    "local wall-clock construction timing of the dense XY Hamiltonian for the "
    "given bounded network, parity-checked against the numpy reference, plus a "
    "summary of the committed benchmark databank; environment-dependent "
    "regression evidence, never a published performance claim, a physical K_nm "
    "claim, or QPU execution"
)

LIVE_TIMING_CAVEAT: Final[str] = (
    "Opportunistic local timing on a shared workstation; the wall-clock numbers "
    "swing with CPU pinning, cache state, and host load. Regression evidence "
    "only, not a published performance claim."
)

_PAULI_X: Final[NDArray[np.complex128]] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y: Final[NDArray[np.complex128]] = np.array(
    [[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128
)
_PAULI_Z: Final[NDArray[np.complex128]] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_bounded_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _as_row(row: object) -> Sequence[Any]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise ValueError("each K_nm row must be a sequence")
    return row


def _as_coupling_matrix(value: object) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("K_nm must be a square list of rows")
    rows = [[_as_float("K_nm row entry", entry) for entry in _as_row(row)] for row in value]
    size = len(rows)
    if not 2 <= size <= _MAX_NODES:
        raise ValueError(f"K_nm must have between 2 and {_MAX_NODES} nodes")
    if any(len(row) != size for row in rows):
        raise ValueError("K_nm must be square")
    for left in range(size):
        if rows[left][left] != 0.0:
            raise ValueError("K_nm diagonal must be zero")
        for right in range(left + 1, size):
            if rows[left][right] != rows[right][left]:
                raise ValueError("K_nm must be symmetric")
    return rows


def _normalise_benchmark(parameters: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(parameters) - {"K_nm", "omega", "repeats", "warmup"}
    if unknown:
        raise ValueError(f"unknown benchmark parameters: {sorted(unknown)}")
    k_nm = _as_coupling_matrix(parameters.get("K_nm"))
    size = len(k_nm)
    raw_omega = parameters.get("omega")
    if not isinstance(raw_omega, Sequence) or isinstance(raw_omega, (str, bytes)):
        raise ValueError("omega must be a sequence")
    if len(raw_omega) != size:
        raise ValueError("omega length must match the number of nodes")
    omega = [_as_float("omega entry", entry) for entry in raw_omega]
    repeats = _as_bounded_int(
        "repeats", parameters.get("repeats", 5), minimum=1, maximum=_MAX_REPEATS
    )
    warmup = _as_bounded_int("warmup", parameters.get("warmup", 1), minimum=0, maximum=_MAX_WARMUP)
    return {"K_nm": k_nm, "omega": omega, "repeats": repeats, "warmup": warmup}


def _pauli_chain(
    operators: Mapping[int, NDArray[np.complex128]], size: int
) -> NDArray[np.complex128]:
    """Return the little-endian Kronecker chain placing ``operators`` by qubit."""
    identity = np.eye(2, dtype=np.complex128)
    matrix: NDArray[np.complex128] = np.eye(1, dtype=np.complex128)
    for qubit in range(size - 1, -1, -1):
        matrix = cast("NDArray[np.complex128]", np.kron(matrix, operators.get(qubit, identity)))
    return matrix


def reference_dense_xy_hamiltonian(
    coupling: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build the dense XY Hamiltonian through the pure-numpy reference path.

    Implements ``H = -sum_i omega_i Z_i - sum_{i<j} K[i,j] (X_i X_j + Y_i Y_j)``
    in Qiskit little-endian qubit ordering (qubit ``0`` is the rightmost
    Kronecker factor), matching the convention of
    :func:`scpn_quantum_control.bridge.knm_hamiltonian.knm_to_hamiltonian` and
    the native Rust kernel. The XY Hamiltonian is real-symmetric, so the real
    part is returned as ``float64``.

    Parameters
    ----------
    coupling : NDArray[np.float64]
        Symmetric zero-diagonal coupling matrix of shape ``(n, n)``.
    frequencies : NDArray[np.float64]
        Natural frequencies of length ``n``.

    Returns
    -------
    NDArray[np.float64]
        The dense XY Hamiltonian of shape ``(2**n, 2**n)``.
    """
    size = len(frequencies)
    if coupling.shape != (size, size):
        raise ValueError("coupling must be square and match the frequency count")
    dimension = 2**size
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for node in range(size):
        hamiltonian -= frequencies[node] * _pauli_chain({node: _PAULI_Z}, size)
    for left in range(size):
        for right in range(left + 1, size):
            strength = coupling[left, right]
            if strength == 0.0:
                continue
            hamiltonian -= strength * _pauli_chain({left: _PAULI_X, right: _PAULI_X}, size)
            hamiltonian -= strength * _pauli_chain({left: _PAULI_Y, right: _PAULI_Y}, size)
    return np.real(hamiltonian)


def native_dense_xy_hamiltonian(
    coupling: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build the dense XY Hamiltonian through the native Rust PyO3 kernel.

    Parameters
    ----------
    coupling : NDArray[np.float64]
        Symmetric zero-diagonal coupling matrix of shape ``(n, n)``.
    frequencies : NDArray[np.float64]
        Natural frequencies of length ``n``.

    Returns
    -------
    NDArray[np.float64]
        The dense XY Hamiltonian of shape ``(2**n, 2**n)``.

    Raises
    ------
    RuntimeError
        If the ``scpn_quantum_engine`` native kernel is not importable.
    """
    engine = optional_rust_engine()
    if engine is None or not hasattr(engine, "build_xy_hamiltonian_dense"):
        raise RuntimeError("the scpn_quantum_engine native kernel is not available")
    size = len(frequencies)
    flat = np.asarray(
        engine.build_xy_hamiltonian_dense(
            coupling.ravel().astype(np.float64), frequencies.astype(np.float64), size
        ),
        dtype=np.float64,
    )
    return flat.reshape(2**size, 2**size)


def measure_p50_us(fn: Callable[[], object], *, warmup: int, repeats: int) -> float:
    """Return the P50 wall-clock latency of ``fn`` in microseconds.

    Parameters
    ----------
    fn : Callable[[], object]
        The zero-argument construction to time.
    warmup : int
        Discarded warm-up invocations run before sampling.
    repeats : int
        Timed invocations; the median sample is returned.

    Returns
    -------
    float
        The median (P50) latency in microseconds.
    """
    for _ in range(warmup):
        fn()
    samples_us: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples_us.append((time.perf_counter_ns() - start) / 1_000.0)
    return float(np.median(samples_us))


def _arrays(
    benchmark_spec: Mapping[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k_nm = np.asarray(benchmark_spec["K_nm"], dtype=np.float64)
    omega = np.asarray(benchmark_spec["omega"], dtype=np.float64)
    return k_nm, omega


def _summarise_databank() -> dict[str, Any]:
    """Summarise the committed benchmark databank through its validated bundle."""
    validated = validate_bundle(build_benchmark_databank_bundle())
    status_counts: dict[str, int] = {}
    for case in validated.bundle.cases:
        status_counts[case.status] = status_counts.get(case.status, 0) + 1
    return {
        "databank_admitted": validated.verdict.admitted,
        "databank_row_count": len(validated.bundle.cases),
        "databank_status_counts": status_counts,
        "databank_timing_caveat": validated.bundle.claim_boundary.validity_domain.note,
    }


class BenchmarkActionHandler(ActionHandler):
    """Executive handler for the simulated ``benchmark`` verb."""

    @property
    def verb(self) -> str:
        """Return ``"benchmark"``."""
        return BENCHMARK_VERB

    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Validate the network and resolve a construction-benchmark plan.

        Parameters
        ----------
        request : ExecutiveRequest
            The benchmark request; ``parameters`` must describe a bounded
            network (``K_nm``, ``omega``) and may bound the timing loop
            (``repeats``, ``warmup``).
        contract : VerbContract
            The resolved ``benchmark`` contract.

        Returns
        -------
        ExecutionPlan
            The normalised, inspectable plan.
        """
        backend = request.backend or _DEFAULT_BACKEND
        if backend not in contract.backends:
            raise ValueError(f"backend {backend!r} is not declared for the benchmark verb")
        benchmark_spec = _normalise_benchmark(request.parameters)
        steps = (
            f"validate the {len(benchmark_spec['K_nm'])}-node K_nm/omega network",
            (
                f"time the {backend} dense XY-Hamiltonian construction "
                f"(warmup {benchmark_spec['warmup']}, repeats {benchmark_spec['repeats']})"
            ),
            "parity-check the native operator against the numpy reference"
            if backend == "rust"
            else "time the numpy reference construction only",
            "summarise the committed benchmark databank bundle",
            "write a standalone reproduction script",
        )
        return ExecutionPlan(
            verb=self.verb,
            action_id=request.action_id,
            backend=backend,
            contract=contract,
            claim_boundary=BENCHMARK_CLAIM_BOUNDARY,
            steps=steps,
            parameters=benchmark_spec,
        )

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Time the dense construction and summarise the committed databank.

        Parameters
        ----------
        plan : ExecutionPlan
            The planned benchmark.

        Returns
        -------
        ExecutionResult
            A succeeded result carrying the P50 timings, the parity verdict,
            the (never asserted) speedup, and the databank summary.

        Raises
        ------
        RuntimeError
            If the ``rust`` backend is requested but the native kernel is not
            importable — the spine seals this as a failed record.
        """
        benchmark_spec: dict[str, Any] = dict(plan.parameters)
        k_nm, omega = _arrays(benchmark_spec)
        warmup = int(benchmark_spec["warmup"])
        repeats = int(benchmark_spec["repeats"])
        size = len(omega)

        reference = reference_dense_xy_hamiltonian(k_nm, omega)
        reference_p50_us = measure_p50_us(
            lambda: reference_dense_xy_hamiltonian(k_nm, omega), warmup=warmup, repeats=repeats
        )

        native_p50_us: float | None = None
        speedup_p50: float | None = None
        parity: bool | None = None
        if plan.backend == "rust":
            native = native_dense_xy_hamiltonian(k_nm, omega)
            parity = bool(np.allclose(native, reference, atol=_PARITY_TOLERANCE))
            native_p50_us = measure_p50_us(
                lambda: native_dense_xy_hamiltonian(k_nm, omega), warmup=warmup, repeats=repeats
            )
            speedup_p50 = reference_p50_us / native_p50_us if native_p50_us > 0.0 else None

        outputs = {
            "backend": plan.backend,
            "speedup_schema": NATIVE_SPEEDUP_SCHEMA,
            "databank_schema": BENCHMARK_DATABANK_SCHEMA,
            "n_nodes": size,
            "hilbert_dim": 2**size,
            "warmup": warmup,
            "repeats": repeats,
            "reference_p50_us": reference_p50_us,
            "native_p50_us": native_p50_us,
            "speedup_p50": speedup_p50,
            "parity": parity,
            "production_claim_allowed": False,
            "timing_caveat": LIVE_TIMING_CAVEAT,
            **_summarise_databank(),
        }
        return ExecutionResult(status="succeeded", outputs=outputs)

    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Write a standalone script that reproduces the deterministic verdicts.

        The script re-asserts only what is reproducible — the operator shape,
        the parity verdict, and the committed databank row count — and re-prints
        fresh timings without asserting them, because wall-clock numbers are
        environment-dependent.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded benchmark result.

        Returns
        -------
        GeneratedScript
            The reproduction script, digest attached.
        """
        benchmark_spec: dict[str, Any] = dict(plan.parameters)
        source = _render_script(
            action_id=plan.action_id,
            benchmark_spec=benchmark_spec,
            backend=plan.backend,
            parity=result.outputs["parity"],
            hilbert_dim=int(result.outputs["hilbert_dim"]),
            databank_row_count=int(result.outputs["databank_row_count"]),
            sealed_reference_p50_us=float(result.outputs["reference_p50_us"]),
        )
        slug = _safe_slug(plan.action_id)
        return build_generated_script(
            filename=f"benchmark_{slug}.py",
            entrypoint=f"python benchmark_{slug}.py",
            source=source,
        )


def _safe_slug(action_id: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in action_id).strip("_")
    return slug or "action"


def _render_script(
    *,
    action_id: str,
    benchmark_spec: Mapping[str, Any],
    backend: str,
    parity: bool | None,
    hilbert_dim: int,
    databank_row_count: int,
    sealed_reference_p50_us: float,
) -> str:
    return (
        '"""Standalone reproduction of a SCPN-QUANTUM-CONTROL studio benchmark action.\n'
        "\n"
        f"Action id: {action_id}\n"
        "Rebuilds the dense XY Hamiltonian for the sealed network, re-asserts the\n"
        "deterministic verdicts (operator shape, native/reference parity, committed\n"
        "databank row count), and re-prints fresh wall-clock timings. Timings are\n"
        "environment-dependent and are informative only, never asserted.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "from scpn_quantum_control.studio.executive_benchmark import (\n"
        "    measure_p50_us,\n"
        "    native_dense_xy_hamiltonian,\n"
        "    reference_dense_xy_hamiltonian,\n"
        ")\n"
        "from scpn_quantum_control.studio.benchmark_databank_bundle import (\n"
        "    build_benchmark_databank_bundle,\n"
        ")\n\n"
        f"K_NM = {benchmark_spec['K_nm']!r}\n"
        f"OMEGA = {benchmark_spec['omega']!r}\n"
        f"WARMUP = {benchmark_spec['warmup']!r}\n"
        f"REPEATS = {benchmark_spec['repeats']!r}\n"
        f"BACKEND = {backend!r}\n"
        f"EXPECTED_PARITY = {parity!r}\n"
        f"EXPECTED_HILBERT_DIM = {hilbert_dim!r}\n"
        f"EXPECTED_DATABANK_ROW_COUNT = {databank_row_count!r}\n"
        f"SEALED_REFERENCE_P50_US = {sealed_reference_p50_us!r}  # informative only\n\n\n"
        "def main() -> int:\n"
        '    """Re-assert the deterministic verdicts and re-print fresh timings."""\n'
        "    k_nm = np.asarray(K_NM, dtype=np.float64)\n"
        "    omega = np.asarray(OMEGA, dtype=np.float64)\n"
        "    reference = reference_dense_xy_hamiltonian(k_nm, omega)\n"
        "    assert reference.shape == (EXPECTED_HILBERT_DIM, EXPECTED_HILBERT_DIM)\n"
        "    reference_p50_us = measure_p50_us(\n"
        "        lambda: reference_dense_xy_hamiltonian(k_nm, omega),\n"
        "        warmup=WARMUP,\n"
        "        repeats=REPEATS,\n"
        "    )\n"
        '    if BACKEND == "rust":\n'
        "        native = native_dense_xy_hamiltonian(k_nm, omega)\n"
        "        parity = bool(np.allclose(native, reference, atol=1e-9))\n"
        "        assert parity == EXPECTED_PARITY, parity\n"
        "    bundle = build_benchmark_databank_bundle()\n"
        "    assert len(bundle.cases) == EXPECTED_DATABANK_ROW_COUNT, len(bundle.cases)\n"
        "    print(\n"
        '        f"parity_verified backend={BACKEND} "\n'
        '        f"fresh_reference_p50_us={reference_p50_us:.1f} "\n'
        '        f"sealed_reference_p50_us={SEALED_REFERENCE_P50_US:.1f} (not asserted)"\n'
        "    )\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


__all__ = [
    "BENCHMARK_CLAIM_BOUNDARY",
    "BENCHMARK_VERB",
    "LIVE_TIMING_CAVEAT",
    "BenchmarkActionHandler",
    "measure_p50_us",
    "native_dense_xy_hamiltonian",
    "reference_dense_xy_hamiltonian",
]
