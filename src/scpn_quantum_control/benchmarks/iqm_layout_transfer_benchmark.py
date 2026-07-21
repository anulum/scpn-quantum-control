# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM square-lattice layout-transfer benchmark harness
"""Benchmark harness for the IQM Garnet layout-transfer preregistration.

Implements the frozen design of
``docs/campaigns/iqm_layout_transfer_square_lattice_prereg_2026-07-21.md``:
for each preregistered chain size the same two-edge-colour Kuramoto-XY
Trotter circuit (:mod:`scpn_quantum_control.analysis.two_colour_schedule`)
is transpiled onto the square lattice under three qubit placements —

* ``optimised`` — chain region from
  :func:`~scpn_quantum_control.hardware.iqm_lattice_calibration.best_chain_region`
  polished by
  :func:`~scpn_quantum_control.hardware.kuramoto_layout_optimiser.optimise_kuramoto_layout`
  over the calibration-fed cost model;
* ``default`` — the transpiler's automatic placement;
* ``naive`` — the amended preregistered baseline: the lexicographically
  smallest connected chain (calibration-blind, topology only; Amendment 1
  of the campaign document).

The primary observable is the absolute mean Z-magnetisation order-parameter
proxy (the counts-supported observable of
:class:`~scpn_quantum_control.analysis.sync_order_parameter.SyncOrderParameter`).
Because it is a sum of single-qubit marginals, the two preregistered
readout-calibration states per size (all-zeros / all-ones over the union of
the arms' measured qubits) support an *exact* tensored per-qubit readout
correction of the endpoint — no full-matrix claim is made or needed.

Everything here is classical and hermetic: the lattice enters only as a
:class:`~scpn_quantum_control.hardware.iqm_lattice_calibration.LatticeCalibration`,
so unit tests run on synthetic grids and the ``IQMFakeGarnet`` dry run plugs
the same functions in unchanged. QPU submission stays owner-gated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from ..analysis.dla_parity_exact_baseline import T_STEP, coupling_matrix
from ..analysis.two_colour_schedule import build_two_colour_circuit, two_qubit_depth
from ..hardware.iqm_lattice_calibration import LatticeCalibration, best_chain_region
from ..hardware.kuramoto_layout_optimiser import LayoutSearchConfig, optimise_kuramoto_layout

__all__ = [
    "ARM_NAMES",
    "CHAIN_SIZES",
    "DEPTH_PARITY_TOLERANCE",
    "IQM_BASIS_GATES",
    "MAIN_SHOTS",
    "READOUT_SHOTS",
    "TRANSPILER_SEED",
    "TROTTER_DEPTH",
    "ArmPlan",
    "DepthParityResult",
    "LayoutTransferPlan",
    "SizeBlockPlan",
    "build_layout_transfer_plan",
    "chain_swap_depth_provider",
    "corrected_order_parameter",
    "coupling_map_from_calibration",
    "depth_parity_gate",
    "exact_order_parameter",
    "initial_bitstring",
    "measured_physical_qubits",
    "naive_chain_layout",
    "optimised_initial_layout",
    "per_qubit_one_probabilities",
    "per_qubit_readout_errors",
]

#: Preregistered chain sizes (frozen in the campaign document).
CHAIN_SIZES: tuple[int, ...] = (8, 12, 16)
#: Preregistered Trotter step count, identical across sizes and arms — the
#: two-colour schedule keeps the per-step two-qubit depth constant in ``n``,
#: matching the committed classical evidence (5 steps = 20 two-qubit layers).
TROTTER_DEPTH: int = 5
#: Preregistered shot budgets.
MAIN_SHOTS: int = 2048
READOUT_SHOTS: int = 1024
#: Depth-parity validity gate: max/min transpiled two-qubit depth across the
#: three arms of one size must stay within this relative tolerance.
DEPTH_PARITY_TOLERANCE: float = 0.10
#: Frozen transpiler seed — Qiskit routing is stochastic when unseeded.
TRANSPILER_SEED: int = 20260721
#: IQM native basis (phased-RX and CZ).
IQM_BASIS_GATES: tuple[str, ...] = ("r", "cz")
#: The three preregistered arms, in submission order.
ARM_NAMES: tuple[str, ...] = ("optimised", "default", "naive")

#: Two-qubit layers of one two-colour Trotter step (colour A + colour B, each
#: an rxx·ryy pair), and layers added per routed SWAP in the deterministic
#: chain depth model below.
_LAYERS_PER_STEP: int = 4
_LAYERS_PER_SWAP: int = 3


def initial_bitstring(n: int) -> str:
    """Preregistered quarter-filling initial state ``'1000' · (n/4)``.

    Quarter filling fixes the conserved mean Z-magnetisation at ``0.5`` for
    every size — far from the folding point of the absolute-value observable
    and sensitive to both amplitude damping and readout error.
    """
    if n < 4 or n % 4 != 0:
        raise ValueError("chain size must be a positive multiple of four")
    return "1000" * (n // 4)


def exact_order_parameter(
    n: int, bitstring: str | None = None, depth: int = TROTTER_DEPTH, t_step: float = T_STEP
) -> float:
    """Exact statevector value of the absolute mean Z-magnetisation proxy.

    The XY-Trotter dynamics conserve total excitation number, so this equals
    the initial-state magnetisation at every depth; the statevector
    computation asserts nothing and simply measures it.
    """
    initial = initial_bitstring(n) if bitstring is None else bitstring
    probabilities = Statevector(
        build_two_colour_circuit(n, initial, depth, t_step)
    ).probabilities()
    magnetisation = 0.0
    for index, probability in enumerate(probabilities):
        magnetisation += float(probability) * (n - 2 * int(index).bit_count())
    return abs(magnetisation) / n


def coupling_map_from_calibration(calibration: LatticeCalibration) -> CouplingMap:
    """Symmetric Qiskit :class:`~qiskit.transpiler.CouplingMap` of the lattice."""
    pairs = [[a, b] for a, b in calibration.edges] + [[b, a] for a, b in calibration.edges]
    return CouplingMap(pairs)


def chain_swap_depth_provider(calibration: LatticeCalibration) -> Any:
    """Deterministic two-qubit depth model of the two-colour chain circuit.

    Each Trotter step costs the constant two-colour layer count plus
    ``_LAYERS_PER_SWAP`` layers for every consecutive logical pair whose
    physical qubits are not lattice neighbours (one SWAP each). Pure and
    transpiler-free, so the preregistered layout search is exactly
    reproducible — no stochastic router inside the cost loop.
    """
    adjacency = set(calibration.edges)

    def provider(
        layout: tuple[int, ...],
        K: Any,  # noqa: N803 — matches the DepthProvider protocol
        omega: Any,
        coupling_map: Any,
        *,
        t: float,
        reps: int,
    ) -> int:
        misses = sum(
            1
            for i in range(len(layout) - 1)
            if (min(layout[i], layout[i + 1]), max(layout[i], layout[i + 1])) not in adjacency
        )
        return int(reps) * (_LAYERS_PER_STEP + _LAYERS_PER_SWAP * misses)

    return provider


def naive_chain_layout(calibration: LatticeCalibration, n: int) -> tuple[int, ...]:
    """Lexicographically smallest connected simple chain of length ``n``.

    The amended naive arm (prereg Amendment 1): a deterministic,
    calibration-blind baseline computed from the coupling graph alone —
    depth-first search from the lowest qubit index, expanding neighbours in
    ascending order, returning the first complete chain (which greedy
    lexicographic DFS makes the smallest one). Keeps the naive arm SWAP-free
    so the depth-parity validity gate compares placements, not routing.
    """
    if n < 2:
        raise ValueError("a chain needs at least two qubits")

    def extend(path: list[int]) -> tuple[int, ...] | None:
        if len(path) == n:
            return tuple(path)
        for nxt in calibration.neighbours(path[-1]):
            if nxt not in path:
                path.append(nxt)
                found = extend(path)
                if found is not None:
                    return found
                path.pop()
        return None

    for start in range(calibration.num_qubits):
        found = extend([start])
        if found is not None:
            return found
    raise ValueError(f"no connected chain of length {n} on the lattice")


def optimised_initial_layout(
    calibration: LatticeCalibration,
    n: int,
    *,
    depth: int = TROTTER_DEPTH,
    seed: int = 0,
    depth_provider: Any = None,
) -> tuple[int, ...]:
    """Calibration-aware layout: best chain region polished by the optimiser.

    The adapter selects the highest-fidelity chain region; the discrete
    Kuramoto optimiser then searches placements within that region under the
    campaign cost model (``t = depth · T_STEP`` so the cost circuit matches
    the preregistered Trotterisation). The optimiser is seeded with the
    region path, so the result is never worse than the adapter's choice.
    """
    region = best_chain_region(calibration, n)
    provider = chain_swap_depth_provider(calibration) if depth_provider is None else depth_provider
    result = optimise_kuramoto_layout(
        coupling_matrix(n),
        np.linspace(0.8, 1.2, n, dtype=np.float64),
        coupling_map_from_calibration(calibration),
        region.physical_qubits,
        mean_gate_fidelity=region.mean_gate_fidelity,
        config=LayoutSearchConfig(t=depth * T_STEP, reps=depth, seed=seed),
        initial_layout=region.physical_qubits,
        depth_provider=provider,
    )
    return result.best_layout


def measured_physical_qubits(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Physical qubit measured into each classical bit, in clbit order.

    Fails closed when a classical bit is unmeasured or measured twice — the
    readout correction below is only meaningful for a one-to-one mapping.
    """
    mapping: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        clbit = circuit.find_bit(instruction.clbits[0]).index
        if clbit in mapping:
            raise ValueError(f"classical bit {clbit} is measured more than once")
        mapping[clbit] = circuit.find_bit(instruction.qubits[0]).index
    if sorted(mapping) != list(range(circuit.num_clbits)):
        raise ValueError("every classical bit must be measured exactly once")
    return tuple(mapping[c] for c in range(circuit.num_clbits))


def per_qubit_one_probabilities(counts: Mapping[str, int], n_clbits: int) -> NDArray[np.float64]:
    """Per-clbit probability of reading ``1`` (Qiskit key order: clbit 0 last)."""
    totals = np.zeros(n_clbits, dtype=np.float64)
    shots = 0
    for key, value in counts.items():
        clean = key.replace(" ", "")
        if len(clean) != n_clbits:
            raise ValueError(f"count key {key!r} does not have {n_clbits} bits")
        count = int(value)
        if count < 0:
            raise ValueError("counts must be non-negative")
        shots += count
        for clbit in range(n_clbits):
            if clean[n_clbits - 1 - clbit] == "1":
                totals[clbit] += count
    if shots <= 0:
        raise ValueError("empty count dictionary")
    return totals / shots


def per_qubit_readout_errors(
    counts_all_zeros: Mapping[str, int],
    counts_all_ones: Mapping[str, int],
    physical_qubits: tuple[int, ...],
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-qubit readout error rates from the two calibration circuits.

    Returns ``(e01, e10)`` keyed by physical qubit: ``e01[q] = P(read 1 |
    prepared 0)`` and ``e10[q] = P(read 0 | prepared 1)``.
    """
    p1_zeros = per_qubit_one_probabilities(counts_all_zeros, len(physical_qubits))
    p1_ones = per_qubit_one_probabilities(counts_all_ones, len(physical_qubits))
    e01 = {q: float(p1_zeros[i]) for i, q in enumerate(physical_qubits)}
    e10 = {q: float(1.0 - p1_ones[i]) for i, q in enumerate(physical_qubits)}
    return e01, e10


def corrected_order_parameter(
    counts: Mapping[str, int],
    physical_qubits: tuple[int, ...],
    e01: Mapping[int, float],
    e10: Mapping[int, float],
) -> float:
    """Readout-corrected absolute mean Z-magnetisation of one arm's counts.

    Applies the exact tensored two-state correction to each single-qubit
    marginal (the observable is a sum of marginals, so this is not an
    approximation to a full-matrix inversion for this endpoint). Corrected
    marginals are clipped into ``[0, 1]``; a non-positive correction
    denominator (readout worse than a coin flip) fails closed.
    """
    p1 = per_qubit_one_probabilities(counts, len(physical_qubits))
    spins = np.zeros(len(physical_qubits), dtype=np.float64)
    for i, q in enumerate(physical_qubits):
        denominator = 1.0 - float(e01[q]) - float(e10[q])
        if denominator <= 0.0:
            raise ValueError(f"readout correction denominator non-positive for qubit {q}")
        p_true = float(np.clip((p1[i] - float(e01[q])) / denominator, 0.0, 1.0))
        spins[i] = 1.0 - 2.0 * p_true
    return float(abs(spins.mean()))


@dataclass(frozen=True)
class DepthParityResult:
    """Depth-parity validity gate outcome for one size block."""

    two_qubit_depths: dict[str, int]
    max_over_min: float
    tolerance: float
    passes: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the gate outcome."""
        return {
            "two_qubit_depths": dict(self.two_qubit_depths),
            "max_over_min": self.max_over_min,
            "tolerance": self.tolerance,
            "passes": self.passes,
        }


def depth_parity_gate(
    two_qubit_depths: Mapping[str, int], tolerance: float = DEPTH_PARITY_TOLERANCE
) -> DepthParityResult:
    """Evaluate the preregistered depth-parity validity gate across arms."""
    if not two_qubit_depths:
        raise ValueError("depth-parity gate needs at least one arm depth")
    values = [int(v) for v in two_qubit_depths.values()]
    if min(values) <= 0:
        raise ValueError("transpiled two-qubit depth must be positive for every arm")
    ratio = max(values) / min(values)
    return DepthParityResult(
        two_qubit_depths={k: int(v) for k, v in two_qubit_depths.items()},
        max_over_min=float(ratio),
        tolerance=float(tolerance),
        # Multiplicative form keeps the boundary inclusive (44 vs 40 at 10 %).
        passes=bool(max(values) <= min(values) * (1.0 + tolerance)),
    )


@dataclass(frozen=True)
class ArmPlan:
    """One transpiled arm of one size block."""

    arm: str
    requested_initial_layout: tuple[int, ...] | None
    measured_qubits: tuple[int, ...]
    two_qubit_depth: int
    two_qubit_gate_count: int
    circuit: QuantumCircuit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (the circuit itself is omitted)."""
        return {
            "arm": self.arm,
            "requested_initial_layout": (
                None
                if self.requested_initial_layout is None
                else list(self.requested_initial_layout)
            ),
            "measured_qubits": list(self.measured_qubits),
            "two_qubit_depth": self.two_qubit_depth,
            "two_qubit_gate_count": self.two_qubit_gate_count,
        }


@dataclass(frozen=True)
class SizeBlockPlan:
    """All circuits and readiness evidence for one preregistered chain size."""

    n: int
    depth: int
    initial_state: str
    exact_reference: float
    arms: tuple[ArmPlan, ...]
    depth_parity: DepthParityResult
    readout_qubits: tuple[int, ...]
    readout_circuits: tuple[QuantumCircuit, QuantumCircuit]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (circuits omitted)."""
        return {
            "n": self.n,
            "depth": self.depth,
            "initial_state": self.initial_state,
            "exact_reference": self.exact_reference,
            "arms": [arm.to_dict() for arm in self.arms],
            "depth_parity": self.depth_parity.to_dict(),
            "readout_qubits": list(self.readout_qubits),
        }


@dataclass(frozen=True)
class LayoutTransferPlan:
    """The full preregistered circuit matrix with readiness evidence."""

    blocks: tuple[SizeBlockPlan, ...]
    main_shots: int
    readout_shots: int
    transpiler_seed: int
    basis_gates: tuple[str, ...]

    @property
    def circuit_count(self) -> int:
        """Total circuits in the matrix (mains plus readout calibrations)."""
        return sum(len(block.arms) + len(block.readout_circuits) for block in self.blocks)

    @property
    def all_gates_pass(self) -> bool:
        """True when every size block passes the depth-parity validity gate."""
        return all(block.depth_parity.passes for block in self.blocks)

    def circuit_manifest(self) -> tuple[tuple[str, QuantumCircuit], ...]:
        """Deterministic ``(label, circuit)`` submission order."""
        entries: list[tuple[str, QuantumCircuit]] = []
        for block in self.blocks:
            for arm in block.arms:
                entries.append((f"main_n{block.n}_{arm.arm}", arm.circuit))
            entries.append((f"readout_n{block.n}_zeros", block.readout_circuits[0]))
            entries.append((f"readout_n{block.n}_ones", block.readout_circuits[1]))
        return tuple(entries)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable plan artefact payload."""
        return {
            "campaign": "iqm_layout_transfer_square_lattice_prereg_2026-07-21",
            "blocks": [block.to_dict() for block in self.blocks],
            "main_shots": self.main_shots,
            "readout_shots": self.readout_shots,
            "transpiler_seed": self.transpiler_seed,
            "basis_gates": list(self.basis_gates),
            "circuit_count": self.circuit_count,
            "all_gates_pass": self.all_gates_pass,
        }


def _two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)


def _transpile_arm(
    logical: QuantumCircuit,
    calibration: LatticeCalibration,
    layout: tuple[int, ...] | None,
    *,
    seed: int,
) -> QuantumCircuit:
    return transpile(
        logical,
        coupling_map=coupling_map_from_calibration(calibration),
        initial_layout=None if layout is None else list(layout),
        basis_gates=list(IQM_BASIS_GATES),
        optimization_level=1,
        seed_transpiler=seed,
    )


def _readout_circuits(
    calibration: LatticeCalibration, qubits: tuple[int, ...], *, seed: int
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """All-zeros / all-ones calibration circuits over ``qubits`` (in order)."""
    circuits: list[QuantumCircuit] = []
    for excite in (False, True):
        qc = QuantumCircuit(calibration.num_qubits, len(qubits))
        for clbit, qubit in enumerate(qubits):
            if excite:
                qc.x(qubit)
            qc.measure(qubit, clbit)
        circuits.append(
            transpile(
                qc,
                initial_layout=list(range(calibration.num_qubits)),
                basis_gates=list(IQM_BASIS_GATES),
                optimization_level=0,
                seed_transpiler=seed,
            )
        )
    return circuits[0], circuits[1]


def build_layout_transfer_plan(
    calibration: LatticeCalibration,
    *,
    sizes: tuple[int, ...] = CHAIN_SIZES,
    depth: int = TROTTER_DEPTH,
    seed: int = TRANSPILER_SEED,
    depth_provider: Any = None,
) -> LayoutTransferPlan:
    """Assemble the full preregistered layout-transfer circuit matrix.

    For each size the same logical two-colour circuit is transpiled under the
    three preregistered placements, the depth-parity validity gate is
    evaluated, the exact statevector reference is computed, and the two
    readout-calibration circuits are built over the union of the arms'
    measured qubits. The returned plan is submission-ready evidence — it
    performs no I/O and never talks to a backend.
    """
    blocks: list[SizeBlockPlan] = []
    for n in sizes:
        initial = initial_bitstring(n)
        logical = build_two_colour_circuit(n, initial, depth)
        logical.measure_all()

        layouts: dict[str, tuple[int, ...] | None] = {
            "optimised": optimised_initial_layout(
                calibration, n, depth=depth, depth_provider=depth_provider
            ),
            "default": None,
            "naive": naive_chain_layout(calibration, n),
        }
        arms: list[ArmPlan] = []
        for arm_name in ARM_NAMES:
            routed = _transpile_arm(logical, calibration, layouts[arm_name], seed=seed)
            arms.append(
                ArmPlan(
                    arm=arm_name,
                    requested_initial_layout=layouts[arm_name],
                    measured_qubits=measured_physical_qubits(routed),
                    two_qubit_depth=two_qubit_depth(routed),
                    two_qubit_gate_count=_two_qubit_gate_count(routed),
                    circuit=routed,
                )
            )

        readout_qubits = tuple(sorted({q for arm in arms for q in arm.measured_qubits}))
        blocks.append(
            SizeBlockPlan(
                n=n,
                depth=depth,
                initial_state=initial,
                exact_reference=exact_order_parameter(n, initial, depth),
                arms=tuple(arms),
                depth_parity=depth_parity_gate({arm.arm: arm.two_qubit_depth for arm in arms}),
                readout_qubits=readout_qubits,
                readout_circuits=_readout_circuits(calibration, readout_qubits, seed=seed),
            )
        )
    return LayoutTransferPlan(
        blocks=tuple(blocks),
        main_shots=MAIN_SHOTS,
        readout_shots=READOUT_SHOTS,
        transpiler_seed=seed,
        basis_gates=IQM_BASIS_GATES,
    )
