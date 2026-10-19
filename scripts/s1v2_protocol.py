# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 information-matched feedback protocol
"""Frozen circuits and analysis of the S1v2 information-matched feedback campaign.

Implements ``docs/campaigns/s1v2_information_matched_feedback_prereg_2026-10-19.md``.
Provider-neutral: the IBM and IQM submitters import this module, so both
platforms run the same logical circuits and the same analysis.

Design (fixed before any hardware data):

* three system qubits on the equator (``RY(pi/2)``), detuned by
  ``omega = linspace(-delta, delta, 3)`` and XY-coupled with strength ``K``;
* each round: Trotter step, a weak ``Y`` readout of every system qubit onto
  its own monitor qubit, all monitors and coins measured together, then per
  qubit the rotations ``RZ(-sign*phi)`` unconditionally and ``RZ(+2*sign*phi)``
  when the control bit is 1, and finally monitor and coin resets. The rotations
  are written with ``rx``/``ry`` only (``_rz_from_rx_ry`` realises
  ``RZ(-theta)``), because IQM can condition only ``x, y, rx, ry, r``;
* arms: ``CL`` (control bit = monitor), ``OL`` (control bit = an independent
  coin whose per-slot ``P(1)`` equals the *ideal* closed-loop monitor rate, see
  ``COIN_PROBABILITIES``) and ``NONE`` (no correction). Every arm prepares,
  measures and resets the same coins and runs on one common layout, so the
  arms use identical qubits, measurements and resets. The primary contrast is
  therefore a comparison of two correction policies with equal resources and
  ideal-matched correction marginals; on hardware the closed-loop marginals may
  differ from the ideal ones, which the measured monitor rates disclose;
* observables from direct ``X`` and ``Y`` readout of the system qubits: the
  Kuramoto order parameter ``R = |mean(<X_i> + i <Y_i>)|`` and the reference
  alignment ``A = mean(<X_i>)``.

The pilot is an operational coherence screen, not a latency measurement and
not a measurement of the main circuits' coherence: one system qubit on the
equator undergoes ``k`` mid-circuit monitor measurements, with or without a
never-firing conditional gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

EXPERIMENT_ID = "s1v2_information_matched_feedback_2026-10-19"
PREREGISTRATION_PATH = "docs/campaigns/s1v2_information_matched_feedback_prereg_2026-10-19.md"

N_SYSTEM = 3
#: Per-slot coin probabilities of the information-matched open-loop arm
#: (round-major, qubit-minor). They equal the ideal closed-loop monitor rates of
#: polarity +1, estimated once with Qiskit Aer (statevector, 200,000 shots,
#: seed_simulator=2, seed_transpiler=1; standard error about 0.001), so in the
#: ideal model the open loop receives the corrections that the predictable phase
#: drift implies, with the closed loop's per-slot frequencies.
COIN_PROBABILITIES: dict[str, tuple[float, ...]] = {
    "c0": (0.782, 0.497, 0.218, 0.607, 0.499, 0.398),
    "c2": (0.810, 0.498, 0.189, 0.657, 0.499, 0.344, 0.397, 0.498, 0.606),
}
MONITOR_ANGLE = math.pi / 3
CORRECTION_ANGLE = 0.5

#: Frozen candidate configurations; c0 is primary, c2 is the uncoupled reference.
CANDIDATES: dict[str, dict[str, float | int]] = {
    "c0": {"K": 0.15, "delta": 0.4, "dt": 1.0, "steps": 1, "rounds": 2},
    "c2": {"K": 0.0, "delta": 0.4, "dt": 1.0, "steps": 1, "rounds": 3},
}

#: (candidate, arm, polarity) in the frozen main matrix.
MAIN_ARMS: tuple[tuple[str, str, int], ...] = (
    ("c0", "CL", 1),
    ("c0", "CL", -1),
    ("c0", "OL", 1),
    ("c0", "NONE", 1),
    ("c2", "CL", 1),
    ("c2", "OL", 1),
    ("c2", "NONE", 1),
)
BASES = ("X", "Y")
PILOT_EVENTS = (0, 1, 2, 4, 8)
PILOT_VARIANTS = ("conditional", "measure_only")

MAIN_SHOTS = 4096
PILOT_SHOTS = 2048
CALIBRATION_SHOTS = 4096

#: Decision events in one c0 circuit: two rounds of (monitor decision + reset).
C0_DECISION_EVENTS = 4
#: Pilot screen: the 99 % lower bound of the zero-event coherence must reach the
#: baseline floor, and the 99 % lower bound of C(4)/C(0) must reach the ratio floor.
PILOT_BASELINE_FLOOR = 0.5
PILOT_COHERENCE_FLOOR = 0.15
PRIMARY_MIN_EFFECT = 0.05
ONE_SIDED_ALPHA = 0.01
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261019


def xy_hamiltonian(coupling: float, omega: Sequence[float]) -> SparsePauliOp:
    """Return ``H = -K sum_{i<j}(X_i X_j + Y_i Y_j) - sum_i omega_i Z_i``.

    Parameters
    ----------
    coupling
        Uniform XY coupling ``K``.
    omega
        Natural frequencies, one per qubit.

    Returns
    -------
    SparsePauliOp
        The Hamiltonian in Qiskit little-endian order.

    """
    n = len(omega)
    terms: list[tuple[str, float]] = []
    if coupling:
        for i in range(n):
            for j in range(i + 1, n):
                for pauli in ("X", "Y"):
                    label = ["I"] * n
                    label[i] = pauli
                    label[j] = pauli
                    terms.append(("".join(reversed(label)), -coupling))
    for i, value in enumerate(omega):
        label = ["I"] * n
        label[i] = "Z"
        terms.append(("".join(reversed(label)), -float(value)))
    return SparsePauliOp.from_list(terms)


def _rz_from_rx_ry(circuit: QuantumCircuit, theta: float, qubit: object) -> None:
    """Apply ``RX(pi/2)``, ``RY(theta)``, ``RX(-pi/2)`` in that order, which equals ``RZ(-theta)``.

    The sign is fixed by the frozen design; tests pin the identity exactly.
    """
    circuit.rx(math.pi / 2, qubit)
    circuit.ry(theta, qubit)
    circuit.rx(-math.pi / 2, qubit)


def arm_label(candidate: str, arm: str, polarity: int, basis: str) -> str:
    """Return the frozen circuit label of one main-matrix entry."""
    sign = "p" if polarity > 0 else "m"
    return f"s1v2_{candidate}_{arm}_{sign}_{basis}"


def build_arm(
    candidate: str, arm: str, polarity: int, basis: str, *, model_latency_ns: int = 0
) -> QuantumCircuit:
    """Build one frozen main-matrix circuit.

    Parameters
    ----------
    candidate
        ``"c0"`` or ``"c2"``.
    arm
        ``"CL"``, ``"OL"`` or ``"NONE"``.
    polarity
        ``+1`` or ``-1``; the sign of the phase correction.
    basis
        ``"X"`` or ``"Y"``; final readout basis of the system qubits.
    model_latency_ns
        Simulation only: an idle ``delay`` on every qubit after each monitor
        decision and after each reset, standing for classical decision time.
        Hardware circuits always use 0, so the device supplies its own latency.

    Returns
    -------
    QuantumCircuit
        Registers ``s`` (system), ``m`` (monitors) and ``c`` (coins), and
        classical registers ``mb`` (monitor bits), ``cb`` (coin bits) and ``ro``
        (system readout). Every arm prepares, measures and resets the same coins,
        so CL, OL and NONE use the same qubits, measurements and resets; only the
        control bit of the correction (monitor, coin, none) differs.

    Raises
    ------
    ValueError
        If an argument is outside the frozen design.

    """
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    if arm not in ("CL", "OL", "NONE"):
        raise ValueError(f"unknown arm {arm!r}")
    if polarity not in (1, -1):
        raise ValueError("polarity must be +1 or -1")
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}")
    if model_latency_ns < 0:
        raise ValueError("model_latency_ns must be non-negative")
    cfg = CANDIDATES[candidate]
    rounds = int(cfg["rounds"])
    omega = np.linspace(-float(cfg["delta"]), float(cfg["delta"]), N_SYSTEM)
    system = QuantumRegister(N_SYSTEM, "s")
    monitors = QuantumRegister(N_SYSTEM, "m")
    registers: list[QuantumRegister | ClassicalRegister] = [system, monitors]
    coins = QuantumRegister(N_SYSTEM, "c")
    registers.append(coins)
    monitor_bits = ClassicalRegister(N_SYSTEM * rounds, "mb")
    registers.append(monitor_bits)
    coin_bits = ClassicalRegister(N_SYSTEM * rounds, "cb")
    registers.append(coin_bits)
    readout = ClassicalRegister(N_SYSTEM, "ro")
    registers.append(readout)
    circuit = QuantumCircuit(*registers, name=arm_label(candidate, arm, polarity, basis))
    for qubit in system:
        circuit.ry(math.pi / 2, qubit)
    evolution = PauliEvolutionGate(
        xy_hamiltonian(float(cfg["K"]), omega.tolist()),
        time=float(cfg["dt"]),
        synthesis=LieTrotter(reps=int(cfg["steps"])),
    )
    for round_index in range(rounds):
        circuit.append(evolution, list(system))
        for i in range(N_SYSTEM):
            circuit.ry(math.pi / 2, monitors[i])
            circuit.sdg(system[i])
            circuit.h(system[i])
            circuit.cx(system[i], monitors[i])
            circuit.ry(MONITOR_ANGLE, monitors[i])
            circuit.cx(system[i], monitors[i])
            circuit.h(system[i])
            circuit.s(system[i])
        for i in range(N_SYSTEM):
            circuit.measure(monitors[i], monitor_bits[round_index * N_SYSTEM + i])
            probability = COIN_PROBABILITIES[candidate][round_index * N_SYSTEM + i]
            circuit.ry(2 * math.asin(math.sqrt(probability)), coins[i])
            circuit.measure(coins[i], coin_bits[round_index * N_SYSTEM + i])
        if model_latency_ns:
            circuit.delay(model_latency_ns, list(circuit.qubits), unit="ns")
        if arm != "NONE":
            for i in range(N_SYSTEM):
                _rz_from_rx_ry(circuit, polarity * CORRECTION_ANGLE, system[i])
                bits = coin_bits if arm == "OL" else monitor_bits
                control = bits[round_index * N_SYSTEM + i]
                with circuit.if_test((control, 1)):
                    _rz_from_rx_ry(circuit, -2 * polarity * CORRECTION_ANGLE, system[i])
        for i in range(N_SYSTEM):
            circuit.reset(monitors[i])
            circuit.reset(coins[i])
        if model_latency_ns:
            circuit.delay(model_latency_ns, list(circuit.qubits), unit="ns")
    if basis == "Y":
        for qubit in system:
            circuit.sdg(qubit)
    for qubit in system:
        circuit.h(qubit)
    circuit.measure(system, readout)
    return circuit


def pilot_label(events: int, variant: str, basis: str) -> str:
    """Return the frozen label of one latency-pilot circuit."""
    return f"s1v2_pilot_{variant}_k{events}_{basis}"


def build_pilot(events: int, variant: str, basis: str) -> QuantumCircuit:
    """Build one latency-pilot circuit on one system and one monitor qubit.

    Parameters
    ----------
    events
        Number of monitor measurement (+ decision) events.
    variant
        ``"conditional"`` adds an ``if_test`` that never fires (monitor stays
        in ``|0>``); ``"measure_only"`` omits it.
    basis
        Final readout basis ``"X"`` or ``"Y"`` of the system qubit.

    Returns
    -------
    QuantumCircuit
        Registers ``s`` and ``m``; classical ``mb`` (``max(events, 1)`` bits)
        and ``ro`` (one bit).

    Raises
    ------
    ValueError
        If an argument is outside the frozen pilot design.

    """
    if events not in PILOT_EVENTS:
        raise ValueError(f"events must be one of {PILOT_EVENTS}")
    if variant not in PILOT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}")
    system = QuantumRegister(1, "s")
    monitor = QuantumRegister(1, "m")
    monitor_bits = ClassicalRegister(max(events, 1), "mb")
    readout = ClassicalRegister(1, "ro")
    circuit = QuantumCircuit(
        system, monitor, monitor_bits, readout, name=pilot_label(events, variant, basis)
    )
    circuit.ry(math.pi / 2, system[0])
    for event in range(events):
        circuit.measure(monitor[0], monitor_bits[event])
        if variant == "conditional":
            with circuit.if_test((monitor_bits[event], 1)):
                circuit.rx(math.pi, system[0])
    if basis == "Y":
        circuit.sdg(system[0])
    circuit.h(system[0])
    circuit.measure(system[0], readout[0])
    return circuit


def build_calibration() -> list[QuantumCircuit]:
    """Return the two system-readout calibration circuits (all 0, all 1)."""
    circuits = []
    for value in (0, 1):
        system = QuantumRegister(N_SYSTEM, "s")
        readout = ClassicalRegister(N_SYSTEM, "ro")
        circuit = QuantumCircuit(system, readout, name=f"s1v2_cal_{value}{value}{value}")
        if value:
            circuit.x(system)
        circuit.measure(system, readout)
        circuits.append(circuit)
    return circuits


def main_circuits() -> list[QuantumCircuit]:
    """Return the frozen main matrix (7 arms x 2 bases) plus 2 calibration circuits."""
    circuits = [
        build_arm(candidate, arm, polarity, basis)
        for candidate, arm, polarity in MAIN_ARMS
        for basis in BASES
    ]
    return circuits + build_calibration()


def pilot_circuits() -> list[QuantumCircuit]:
    """Return the frozen latency pilot (5 event counts x 2 variants x 2 bases)."""
    return [
        build_pilot(events, variant, basis)
        for events in PILOT_EVENTS
        for variant in PILOT_VARIANTS
        for basis in BASES
    ]


def logical_digest(circuits: Sequence[QuantumCircuit]) -> str:
    """Return the SHA-256 of the circuits' OpenQASM 3 text, in order."""
    from qiskit import qasm3

    digest = hashlib.sha256()
    for circuit in circuits:
        digest.update(circuit.name.encode("utf-8") + b"\n")
        digest.update(qasm3.dumps(circuit).encode("utf-8") + b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------- analysis


def register_marginal(counts: Mapping[str, int], register_index: int) -> dict[str, int]:
    """Marginalise space-separated Qiskit counts onto one register.

    Parameters
    ----------
    counts
        Counts keyed by space-separated register bitstrings (last register first).
    register_index
        Index into ``key.split()``.

    Returns
    -------
    dict
        Counts of that register's bitstrings.

    """
    out: dict[str, int] = {}
    for key, value in counts.items():
        part = key.split()[register_index]
        out[part] = out.get(part, 0) + int(value)
    return out


def readout_marginal(counts: Mapping[str, int], width: int) -> dict[str, int]:
    """Return the counts of the last-declared register (``ro``) from provider counts.

    Qiskit-style keys separate registers by spaces with the last-declared
    register first; keys without spaces carry the last register in the
    leftmost ``width`` characters.

    Raises
    ------
    ValueError
        If a key is too short for the requested register width.

    """
    out: dict[str, int] = {}
    for key, value in counts.items():
        part = key.split()[0] if " " in key else key[:width]
        if len(part) != width:
            raise ValueError(f"count key {key!r} has no {width}-bit readout register")
        out[part] = out.get(part, 0) + int(value)
    return out


def bit_probabilities(counts: Mapping[str, int], width: int) -> np.ndarray:
    """Return ``P(bit_i = 1)`` for every bit of fixed-width counts (bit 0 rightmost)."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts are empty")
    ones = np.zeros(width)
    for key, value in counts.items():
        if len(key) != width:
            raise ValueError(f"count key {key!r} is not {width} bits wide")
        for i in range(width):
            ones[i] += value * int(key[width - 1 - i])
    return ones / total


def expectations(counts: Mapping[str, int], width: int) -> np.ndarray:
    """Return ``<Z_i> = 1 - 2 P(bit_i = 1)`` from fixed-width counts."""
    return 1.0 - 2.0 * bit_probabilities(counts, width)


def order_parameter(ex: np.ndarray, ey: np.ndarray) -> float:
    """Return ``R = |mean(<X_i> + i <Y_i>)|``."""
    return float(abs(np.mean(np.asarray(ex) + 1j * np.asarray(ey))))


def readout_scale(
    cal_zero: Mapping[str, int], cal_one: Mapping[str, int], width: int
) -> np.ndarray:
    """Return the per-qubit symmetric-readout contraction ``1 - p(1|0) - p(0|1)``.

    Raises
    ------
    ValueError
        If any qubit's readout is uninformative (contraction not positive).

    """
    p_one_given_zero = bit_probabilities(cal_zero, width)
    p_zero_given_one = 1.0 - bit_probabilities(cal_one, width)
    scale = 1.0 - p_one_given_zero - p_zero_given_one
    if np.any(scale <= 0.0):
        raise ValueError("readout calibration is uninformative")
    return scale


def bootstrap_expectations(
    counts: Mapping[str, int], width: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``<Z_i>`` for ``draws`` multinomial resamples of the counts (draws x width).

    Parameters
    ----------
    counts
        Fixed-width counts (bit 0 rightmost).
    width
        Register width.
    draws
        Number of resamples.
    rng
        Random generator.

    Returns
    -------
    numpy.ndarray
        One row of single-qubit expectations per resample.

    """
    keys = list(counts)
    values = np.array([counts[key] for key in keys], dtype=np.int64)
    total = int(values.sum())
    signs = np.array(
        [[1 - 2 * int(key[width - 1 - i]) for i in range(width)] for key in keys], dtype=np.float64
    )
    samples = rng.multinomial(total, values / total, size=draws)
    result: np.ndarray = samples @ signs / total
    return result


@dataclass(frozen=True)
class ArmEstimate:
    """Point estimates and bootstrap draws of R and A for one arm."""

    r: float
    a: float
    r_draws: np.ndarray
    a_draws: np.ndarray


def estimate_arm(
    x_counts: Mapping[str, int],
    y_counts: Mapping[str, int],
    *,
    scale: np.ndarray | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    rng: np.random.Generator | None = None,
) -> ArmEstimate:
    """Estimate R and A for one arm, with a multinomial bootstrap of both bases.

    Parameters
    ----------
    x_counts, y_counts
        System-readout counts of the X- and Y-basis circuits.
    scale
        Optional per-qubit readout contraction; expectations are divided by it.
    draws
        Bootstrap draws.
    rng
        Random generator (seeded by the caller for reproducibility).

    Returns
    -------
    ArmEstimate
        Point estimates and bootstrap draws.

    """
    generator = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    divisor = np.ones(N_SYSTEM) if scale is None else np.asarray(scale)

    ex = expectations(x_counts, N_SYSTEM) / divisor
    ey = expectations(y_counts, N_SYSTEM) / divisor
    ex_draws = bootstrap_expectations(x_counts, N_SYSTEM, draws, generator) / divisor
    ey_draws = bootstrap_expectations(y_counts, N_SYSTEM, draws, generator) / divisor
    return ArmEstimate(
        order_parameter(ex, ey),
        float(np.mean(ex)),
        np.abs(np.mean(ex_draws + 1j * ey_draws, axis=1)),
        np.mean(ex_draws, axis=1),
    )


def one_sided_lower(draws: np.ndarray, alpha: float = ONE_SIDED_ALPHA) -> float:
    """Return the one-sided lower ``(1 - alpha)`` bootstrap bound."""
    return float(np.quantile(draws, alpha))


def one_sided_upper(draws: np.ndarray, alpha: float = ONE_SIDED_ALPHA) -> float:
    """Return the one-sided upper ``(1 - alpha)`` bootstrap bound."""
    return float(np.quantile(draws, 1.0 - alpha))


def analyse_main(
    counts_by_label: Mapping[str, Mapping[str, int]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Apply the preregistered analysis to one backend's main-matrix counts.

    Parameters
    ----------
    counts_by_label
        System-readout counts (register ``ro`` only, 3-bit keys) per frozen label,
        including the two calibration labels.
    draws, seed
        Bootstrap settings.

    Returns
    -------
    dict
        Point estimates, bounds, the primary and polarity decisions, and the
        readout-mitigated secondary estimates.

    """
    rng = np.random.default_rng(seed)
    scale = readout_scale(
        counts_by_label["s1v2_cal_000"], counts_by_label["s1v2_cal_111"], N_SYSTEM
    )
    raw: dict[str, ArmEstimate] = {}
    mitigated: dict[str, ArmEstimate] = {}
    for candidate, arm, polarity in MAIN_ARMS:
        key = f"{candidate}_{arm}_{'p' if polarity > 0 else 'm'}"
        x_counts = counts_by_label[arm_label(candidate, arm, polarity, "X")]
        y_counts = counts_by_label[arm_label(candidate, arm, polarity, "Y")]
        raw[key] = estimate_arm(x_counts, y_counts, draws=draws, rng=rng)
        mitigated[key] = estimate_arm(x_counts, y_counts, scale=scale, draws=draws, rng=rng)

    def contrast(table: Mapping[str, ArmEstimate], first: str, second: str) -> dict[str, float]:
        diff = table[first].r_draws - table[second].r_draws
        return {
            "point": table[first].r - table[second].r,
            "lower_99": one_sided_lower(diff),
        }

    primary = contrast(raw, "c0_CL_p", "c0_OL_p")
    primary_pass = primary["lower_99"] > 0.0 and primary["point"] >= PRIMARY_MIN_EFFECT
    polarity_check = {
        "a_plus": raw["c0_CL_p"].a,
        "a_plus_upper_99": one_sided_upper(raw["c0_CL_p"].a_draws),
        "a_minus": raw["c0_CL_m"].a,
        "a_minus_lower_99": one_sided_lower(raw["c0_CL_m"].a_draws),
    }
    polarity_pass = polarity_check["a_plus_upper_99"] < 0.0 < polarity_check["a_minus_lower_99"]
    return {
        "readout_scale": scale.tolist(),
        "arms": {key: {"R": est.r, "A": est.a} for key, est in raw.items()},
        "arms_readout_mitigated": {
            key: {"R": est.r, "A": est.a} for key, est in mitigated.items()
        },
        "primary_c0_cl_minus_ol": primary,
        "primary_pass": bool(primary_pass),
        "polarity": polarity_check,
        "polarity_pass": bool(polarity_pass),
        "secondary": {
            "c0_cl_minus_none": contrast(raw, "c0_CL_p", "c0_NONE_p"),
            "c2_cl_minus_ol": contrast(raw, "c2_CL_p", "c2_OL_p"),
            "c0_cl_minus_ol_readout_mitigated": contrast(mitigated, "c0_CL_p", "c0_OL_p"),
        },
    }


def monitor_rates(monitor_counts: Mapping[str, int], rounds: int) -> list[float]:
    """Return per-slot ``P(1)`` of the ``mb`` register (round-major, qubit-minor)."""
    return [float(p) for p in bit_probabilities(monitor_counts, N_SYSTEM * rounds)]


def pilot_coherence(x_counts: Mapping[str, int], y_counts: Mapping[str, int]) -> float:
    """Return ``|<X> + i <Y>|`` of the pilot system qubit from 1-bit counts."""
    ex = expectations(x_counts, 1)[0]
    ey = expectations(y_counts, 1)[0]
    return float(abs(ex + 1j * ey))


def analyse_pilot(
    counts_by_label: Mapping[str, Mapping[str, int]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Apply the preregistered pilot screen to one backend's pilot counts.

    The screen passes only if the zero-event coherence is clearly present (its
    one-sided 99 % lower bound reaches ``PILOT_BASELINE_FLOOR``) and the retained
    ratio ``C(4)/C(0)`` of the conditional variant has a one-sided 99 % lower
    bound of at least ``PILOT_COHERENCE_FLOOR``. Both bounds come from a
    multinomial bootstrap of the X and Y circuits. A noise-dominated baseline
    therefore cannot pass through a ratio of two small numbers.

    Parameters
    ----------
    counts_by_label
        System-readout counts (register ``ro``, 1-bit keys) per pilot label.
    draws, seed
        Bootstrap settings.

    Returns
    -------
    dict
        Coherence per event count and variant, the baseline and ratio with their
        lower bounds, and the go/stop decision for the main block.

    """
    rng = np.random.default_rng(seed)
    coherence: dict[str, dict[int, float]] = {variant: {} for variant in PILOT_VARIANTS}
    for variant in PILOT_VARIANTS:
        for events in PILOT_EVENTS:
            coherence[variant][events] = pilot_coherence(
                counts_by_label[pilot_label(events, variant, "X")],
                counts_by_label[pilot_label(events, variant, "Y")],
            )

    def coherence_draws(events: int) -> np.ndarray:
        x_counts = counts_by_label[pilot_label(events, "conditional", "X")]
        y_counts = counts_by_label[pilot_label(events, "conditional", "Y")]
        ex = bootstrap_expectations(x_counts, 1, draws, rng)[:, 0]
        ey = bootstrap_expectations(y_counts, 1, draws, rng)[:, 0]
        coherence_values: np.ndarray = np.abs(ex + 1j * ey)
        return coherence_values

    base_draws = coherence_draws(0)
    event_draws = coherence_draws(C0_DECISION_EVENTS)
    ratio_draws = np.where(base_draws > 0, event_draws / np.maximum(base_draws, 1e-12), 0.0)
    base = coherence["conditional"][0]
    ratio = coherence["conditional"][C0_DECISION_EVENTS] / base if base > 0 else 0.0
    baseline_lower = one_sided_lower(base_draws)
    ratio_lower = one_sided_lower(ratio_draws)
    return {
        "coherence": {variant: dict(values) for variant, values in coherence.items()},
        "baseline_coherence": base,
        "baseline_lower_99": baseline_lower,
        "baseline_floor": PILOT_BASELINE_FLOOR,
        "retained_ratio_at_c0_events": ratio,
        "retained_ratio_lower_99": ratio_lower,
        "coherence_floor": PILOT_COHERENCE_FLOOR,
        "main_block_go": bool(
            baseline_lower >= PILOT_BASELINE_FLOOR and ratio_lower >= PILOT_COHERENCE_FLOOR
        ),
    }
