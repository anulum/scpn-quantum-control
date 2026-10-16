# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-bound MCWF sampling evidence
"""Attach source and sampling uncertainty to the original MCWF ensemble."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import DenseAllocationError
from ..execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from ..execution_reservations import reserve_execution_memory
from .open_system_objectives import BoundedOpenSystemObjectiveCase
from .tensor_jump import mcwf_ensemble


@dataclass(frozen=True, slots=True)
class MCWFEnsembleEvidence:
    """Separate sampling evidence returned by the source-bound builder.

    Attributes
    ----------
    source_case_json
        Snapshot of the original validated Python case, not a workspace codec.
    trajectory_seeds
        Exact child seeds in original ensemble order.
    effective_gamma_amp, effective_gamma_deph
        Original source rates after the case's declared damping scale, in
        inverse model-time units. These are simulation parameters.
    times
        Read-only original time grid in model-time units.
    trajectory_values
        Read-only dimensionless R histories, one original trajectory per row.
    mean
        Read-only original mean of trajectory magnitudes; this is not the
        magnitude evaluated on an ensemble density matrix.
    sample_variance
        Read-only n-minus-one variance of R, or None for a single trajectory.
    standard_error
        Read-only square root of sample_variance/n, or None when unavailable.
    total_jumps
        Original sum of quantum jumps over the ensemble.

    """

    source_case_json: str
    trajectory_seeds: tuple[int, ...]
    effective_gamma_amp: float
    effective_gamma_deph: float
    times: NDArray[np.float64]
    trajectory_values: NDArray[np.float64]
    mean: NDArray[np.float64]
    sample_variance: NDArray[np.float64] | None
    standard_error: NDArray[np.float64] | None
    total_jumps: int

    def to_dict(self) -> dict[str, object]:
        """Export detached source and sampling values without a fidelity verdict.

        Returns
        -------
        dict
            JSON-ready companion with exact seed custody and explicit unknown
            discretisation error and confidence interval. The source-case
            digest identifies its Python JSON snapshot, not a numerical plan.

        """
        return {
            "schema": "mcwf_ensemble_sampling_evidence.v1",
            "source_case": json.loads(self.source_case_json),
            "source_case_digest": hashlib.sha256(
                b"mcwf_source_case.v1\n" + self.source_case_json.encode("utf-8")
            ).hexdigest(),
            "model_domain": "cp_markovian_simulation",
            "effective_rates": {
                "gamma_amp": self.effective_gamma_amp,
                "gamma_deph": self.effective_gamma_deph,
            },
            "trajectory_seeds": list(self.trajectory_seeds),
            "rng": "numpy.default_rng: integers(0, 2**31)",
            "numpy_version": np.__version__,
            "solver": "scpn_quantum_control.phase.tensor_jump.mcwf_ensemble",
            "estimand": "mean of dimensionless per-trajectory synchronisation magnitudes R",
            "times": self.times.tolist(),
            "trajectory_values": self.trajectory_values.tolist(),
            "mean": self.mean.tolist(),
            "sample_variance": (
                None if self.sample_variance is None else self.sample_variance.tolist()
            ),
            "standard_error": (
                None if self.standard_error is None else self.standard_error.tolist()
            ),
            "variance_denominator": len(self.trajectory_seeds) - 1,
            "total_jumps": self.total_jumps,
            "time_discretisation_error": None,
            "confidence_interval": None,
            "claim_boundary": (
                "Sampling evidence for the original CP Markovian simulation only; "
                "no calibrated interval, discretisation bound, non-Markovian process, "
                "hardware noise fidelity or gradient qualification."
            ),
        }


def materialise_mcwf_ensemble_evidence(
    case: BoundedOpenSystemObjectiveCase,
    *,
    model_domain: str = "cp_markovian_simulation",
    max_dense_gib: float | None = None,
) -> MCWFEnsembleEvidence:
    """Run the original ensemble with source custody and separate sampling error.

    Parameters
    ----------
    case
        Existing validated, explicitly seeded case. Its original parameter
        scales apply to coupling and simulation rates. Source arrays are
        copied without mutating the caller or its global NumPy RNG.
    model_domain
        Only ``cp_markovian_simulation`` is supported; this cannot admit an
        unqualified non-CP, non-Markovian or physical-device model.
    max_dense_gib
        Optional allowance for declared state, full history, copied inputs and
        statistics in GiB. Third-party and Python-object overhead is separate.

    Returns
    -------
    MCWFEnsembleEvidence
        Original raw values plus their immutable source snapshot, child seeds,
        sample variance and standard error. No confidence or bias bound is
        inferred from same-seed replay or from a single trajectory.

    Raises
    ------
    ValueError
        If the model domain or delegated original physical input is refused.
    DenseAllocationError
        If the grid or declared live buffers exceed addressability or budget.

    """
    if model_domain != "cp_markovian_simulation":
        raise ValueError("Only CP Markovian simulation sampling evidence is supported.")
    intervals = case.t_max / case.dt
    if not math.isfinite(intervals) or intervals > sys.maxsize - 2:
        raise DenseAllocationError("MCWF evidence history exceeds native addressability.")
    samples = 1 if case.t_max == 0 else max(1, math.ceil(intervals)) + 1
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer.hilbert("trajectory_state", "forward", case.n_oscillators, count=4),
            ExecutionBuffer(
                "trajectory_histories",
                "dense_output",
                (case.n_trajectories, samples),
                "float64",
                3,
            ),
            ExecutionBuffer("statistics", "dense_output", (samples,), "float64", 10),
            ExecutionBuffer("seeds", "intermediate", (case.n_trajectories,), "int64"),
            ExecutionBuffer(
                "coupling",
                "intermediate",
                (case.n_oscillators, case.n_oscillators),
                "float64",
                3,
            ),
            ExecutionBuffer("frequencies", "intermediate", (case.n_oscillators,), "float64", 2),
        )
    )
    with reserve_execution_memory(plan, max_gib=max_dense_gib):
        snapshot = replace(case)
        source_json = json.dumps(
            snapshot.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        coupling, amplitude, dephasing = snapshot.scaled_inputs(snapshot.initial_params)
        raw = mcwf_ensemble(
            coupling,
            snapshot.omega,
            amplitude,
            dephasing,
            snapshot.t_max,
            snapshot.dt,
            snapshot.n_trajectories,
            snapshot.seed,
            max_dense_gib=max_dense_gib,
        )
        seeds = tuple(
            int(seed)
            for seed in np.random.default_rng(snapshot.seed).integers(
                0, 2**31, size=snapshot.n_trajectories
            )
        )
        times = np.asarray(raw["times"], dtype=np.float64).copy()
        values = np.asarray(raw["R_trajectories"], dtype=np.float64).copy()
        mean = np.asarray(raw["R_mean"], dtype=np.float64).copy()
        variance: NDArray[np.float64] | None = None
        error: NDArray[np.float64] | None = None
        if snapshot.n_trajectories > 1:
            variance = np.var(values, axis=0, ddof=1)
            error = np.sqrt(variance / snapshot.n_trajectories)
            variance.flags.writeable = False
            error.flags.writeable = False
        for array in (times, values, mean):
            array.flags.writeable = False
        return MCWFEnsembleEvidence(
            source_json,
            seeds,
            amplitude,
            dephasing,
            times,
            values,
            mean,
            variance,
            error,
            int(raw["total_jumps"]),
        )
