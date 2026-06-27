# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Petri Nets
"""Quantum Petri nets with superposition-token dynamics for control systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector

from .._constants import WEIGHT_SPARSITY_EPS
from ..bridge.sc_to_quantum import probability_to_angle

_qpetri_sample_marking_rust: Any = None
_qpetri_state_metrics_rust: Any = None
_qpetri_transition_activity_rust: Any = None
_qpetri_campaign_aggregate_rust: Any = None
_qpetri_rust_import_error: ImportError | None = None

try:
    from scpn_quantum_engine import (
        qpetri_campaign_aggregate as _imported_qpetri_campaign_aggregate,
    )
    from scpn_quantum_engine import (
        qpetri_sample_marking as _imported_qpetri_sample_marking,
    )
    from scpn_quantum_engine import qpetri_state_metrics as _imported_qpetri_state_metrics
    from scpn_quantum_engine import (
        qpetri_transition_activity as _imported_qpetri_transition_activity,
    )
except ImportError as exc:
    _qpetri_rust_import_error = exc
else:
    _qpetri_sample_marking_rust = _imported_qpetri_sample_marking
    _qpetri_state_metrics_rust = _imported_qpetri_state_metrics
    _qpetri_transition_activity_rust = _imported_qpetri_transition_activity
    _qpetri_campaign_aggregate_rust = _imported_qpetri_campaign_aggregate


@dataclass(frozen=True)
class QuantumPetriStepReport:
    """Structured report for one superposition-token transition sweep."""

    input_marking: NDArray[np.float64]
    output_marking: NDArray[np.float64]
    transition_activity: NDArray[np.float64]
    statevector_purity: float
    statevector_entropy_bits: float

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of one Petri-net step."""
        return {
            "input_marking": self.input_marking.astype(float).tolist(),
            "output_marking": self.output_marking.astype(float).tolist(),
            "transition_activity": self.transition_activity.astype(float).tolist(),
            "statevector_purity": float(self.statevector_purity),
            "statevector_entropy_bits": float(self.statevector_entropy_bits),
        }


@dataclass(frozen=True)
class QuantumPetriCampaignReport:
    """Aggregate campaign report over many markings."""

    steps: list[QuantumPetriStepReport]
    mean_output_marking: NDArray[np.float64]
    mean_transition_activity: NDArray[np.float64]
    mean_statevector_entropy_bits: float
    mean_statevector_purity: float

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of a Petri campaign."""
        return {
            "steps": [step.to_payload() for step in self.steps],
            "mean_output_marking": self.mean_output_marking.astype(float).tolist(),
            "mean_transition_activity": self.mean_transition_activity.astype(float).tolist(),
            "mean_statevector_entropy_bits": float(self.mean_statevector_entropy_bits),
            "mean_statevector_purity": float(self.mean_statevector_purity),
            "n_steps": len(self.steps),
        }


class QuantumPetriNet:
    """Quantum Petri net with amplitude-encoded token state."""

    def __init__(
        self,
        n_places: int,
        n_transitions: int,
        W_in: NDArray[np.float64],
        W_out: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ):
        if n_places <= 0 or n_transitions <= 0:
            raise ValueError(
                f"n_places ({n_places}) and n_transitions ({n_transitions}) must be positive"
            )
        W_in_arr = np.asarray(W_in, dtype=np.float64)
        W_out_arr = np.asarray(W_out, dtype=np.float64)
        thresholds_arr = np.asarray(thresholds, dtype=np.float64)
        if W_in_arr.shape != (n_transitions, n_places):
            raise ValueError(
                f"W_in shape {W_in_arr.shape} != expected ({n_transitions}, {n_places})"
            )
        if W_out_arr.shape != (n_places, n_transitions):
            raise ValueError(
                f"W_out shape {W_out_arr.shape} != expected ({n_places}, {n_transitions})"
            )
        if len(thresholds_arr) != n_transitions:
            raise ValueError(
                f"thresholds length {len(thresholds_arr)} != n_transitions {n_transitions}"
            )
        self.n_places = n_places
        self.n_transitions = n_transitions
        self.W_in = W_in_arr
        self.W_out = W_out_arr
        self.thresholds = thresholds_arr

    def encode_marking(self, marking: NDArray[np.float64]) -> QuantumCircuit:
        """Amplitude-encode token state: p_i -> Ry(theta_i)|0>."""
        marking_arr = np.asarray(marking, dtype=np.float64)
        if marking_arr.ndim != 1 or marking_arr.shape[0] != self.n_places:
            raise ValueError(
                f"marking must be one-dimensional with length {self.n_places}, got {marking_arr.shape}"
            )
        qc = QuantumCircuit(self.n_places)
        for i, m in enumerate(marking_arr):
            theta = probability_to_angle(float(np.clip(m, 0.0, 1.0)))
            qc.ry(theta, i)
        return qc

    def apply_transition(self, circuit: QuantumCircuit, t_idx: int) -> None:
        """Apply transition t_idx as controlled rotations."""
        input_places = [
            p for p in range(self.n_places) if abs(self.W_in[t_idx, p]) > WEIGHT_SPARSITY_EPS
        ]

        for p in input_places:
            theta = probability_to_angle(float(abs(self.W_in[t_idx, p]))) * self.thresholds[t_idx]
            circuit.ry(-theta, p)

        for p_out in range(self.n_places):
            w_out = self.W_out[p_out, t_idx]
            if abs(w_out) < WEIGHT_SPARSITY_EPS:
                continue
            theta = probability_to_angle(float(abs(w_out)))
            controls = [p for p in input_places if p != p_out]
            if len(controls) == 0:
                circuit.ry(theta, p_out)
            elif len(controls) == 1:
                circuit.cry(theta, controls[0], p_out)
            else:
                gate = RYGate(theta).control(len(controls))
                circuit.append(gate, controls + [p_out])

    def _statevector_marking(self, sv: Statevector) -> NDArray[np.float64]:
        out = np.zeros(self.n_places, dtype=np.float64)
        for p in range(self.n_places):
            probs = sv.probabilities([p])
            out[p] = probs[1]
        return out

    def _transition_activity(self, marking: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(np.asarray(marking, dtype=np.float64), 0.0, 1.0)
        if _qpetri_transition_activity_rust is not None:
            return np.asarray(
                _qpetri_transition_activity_rust(
                    self.W_in.reshape(-1),
                    clipped,
                    self.thresholds,
                    self.n_transitions,
                    self.n_places,
                    WEIGHT_SPARSITY_EPS,
                ),
                dtype=np.float64,
            )
        activity = np.zeros(self.n_transitions, dtype=np.float64)
        for t in range(self.n_transitions):
            incoming = np.clip(np.abs(self.W_in[t]), 0.0, 1.0)
            if float(np.sum(incoming)) <= WEIGHT_SPARSITY_EPS:
                activity[t] = 0.0
                continue
            weighted_token = float(np.dot(incoming, clipped))
            normaliser = float(np.sum(incoming))
            activity[t] = float(
                np.clip((weighted_token / normaliser) * self.thresholds[t], 0.0, 1.0)
            )
        return activity

    @staticmethod
    def _campaign_aggregate_numpy(
        output_stack: NDArray[np.float64],
        activity_stack: NDArray[np.float64],
        entropies: NDArray[np.float64],
        purities: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        return (
            np.mean(output_stack, axis=0).astype(np.float64),
            np.mean(activity_stack, axis=0).astype(np.float64),
            float(np.mean(entropies)),
            float(np.mean(purities)),
        )

    def step_report(self, marking: NDArray[np.float64]) -> QuantumPetriStepReport:
        """Run one transition sweep and return superposition observables."""
        qc = self.encode_marking(marking)
        for t in range(self.n_transitions):
            self.apply_transition(qc, t)
        sv = Statevector.from_instruction(qc)
        output = self._statevector_marking(sv)
        full_probs = np.asarray(sv.probabilities(), dtype=np.float64)
        if _qpetri_state_metrics_rust is not None:
            entropy_bits, purity = _qpetri_state_metrics_rust(full_probs)
            entropy_bits = float(entropy_bits)
            purity = float(purity)
        else:
            non_zero_probs = full_probs[full_probs > 0.0]
            entropy_bits = float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))
            purity = float(np.sum(np.square(full_probs)))
        return QuantumPetriStepReport(
            input_marking=np.asarray(marking, dtype=np.float64),
            output_marking=output,
            transition_activity=self._transition_activity(marking),
            statevector_purity=purity,
            statevector_entropy_bits=entropy_bits,
        )

    def step(self, marking: NDArray[np.float64], shots: int | None = None) -> NDArray[np.float64]:
        """Return output marking after one sweep.

        If ``shots`` is supplied, output is a sampled estimate from Bernoulli
        marginals, matching hardware-like finite-shot observation.
        """
        report = self.step_report(marking)
        if shots is None:
            return report.output_marking
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots!r}")
        if _qpetri_sample_marking_rust is not None:
            return np.asarray(
                _qpetri_sample_marking_rust(
                    np.asarray(report.output_marking, dtype=np.float64),
                    shots,
                    123456789,
                ),
                dtype=np.float64,
            )
        rng = np.random.default_rng(123456789)
        return np.array(
            [rng.binomial(shots, p) / shots for p in report.output_marking], dtype=np.float64
        )

    def run_campaign(self, markings: NDArray[np.float64]) -> QuantumPetriCampaignReport:
        """Execute many markings and return aggregate campaign metrics."""
        matrix = np.asarray(markings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_places:
            raise ValueError(
                "markings must be a 2D array with shape (n_samples, n_places). "
                f"Expected second dimension {self.n_places}, got {matrix.shape}."
            )
        steps = [self.step_report(matrix[idx]) for idx in range(matrix.shape[0])]
        output_stack = np.vstack([s.output_marking for s in steps])
        activity_stack = np.vstack([s.transition_activity for s in steps])
        entropies = np.array([s.statevector_entropy_bits for s in steps], dtype=np.float64)
        purities = np.array([s.statevector_purity for s in steps], dtype=np.float64)

        if _qpetri_campaign_aggregate_rust is not None:
            mean_output, mean_activity, mean_entropy, mean_purity = (
                _qpetri_campaign_aggregate_rust(
                    output_stack.reshape(-1),
                    activity_stack.reshape(-1),
                    entropies,
                    purities,
                    matrix.shape[0],
                    self.n_places,
                    self.n_transitions,
                )
            )
            mean_output_marking = np.asarray(mean_output, dtype=np.float64)
            mean_transition_activity = np.asarray(mean_activity, dtype=np.float64)
            mean_statevector_entropy_bits = float(mean_entropy)
            mean_statevector_purity = float(mean_purity)
        else:
            (
                mean_output_marking,
                mean_transition_activity,
                mean_statevector_entropy_bits,
                mean_statevector_purity,
            ) = self._campaign_aggregate_numpy(output_stack, activity_stack, entropies, purities)

        return QuantumPetriCampaignReport(
            steps=steps,
            mean_output_marking=mean_output_marking,
            mean_transition_activity=mean_transition_activity,
            mean_statevector_entropy_bits=mean_statevector_entropy_bits,
            mean_statevector_purity=mean_statevector_purity,
        )

    @classmethod
    def from_matrices(
        cls,
        W_in: NDArray[np.float64],
        W_out: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ) -> QuantumPetriNet:
        """Construct from arc-weight matrices, inferring shape."""
        n_t, n_p = W_in.shape
        return cls(n_p, n_t, W_in, W_out, thresholds)


__all__ = [
    "QuantumPetriNet",
    "QuantumPetriStepReport",
    "QuantumPetriCampaignReport",
]
