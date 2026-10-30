# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — select numerical methods by measured error and cost
"""Choose a numerical method only from measured evidence, never from a heuristic.

The planner ranks candidates that were actually run on the requested problem
and size, at the requested arithmetic precision, with every required error
component measured. It never runs a solver: candidates come from committed
evidence rows. Its rules:

* an exact request never selects an approximate method (compressed, trajectory,
  Trotter, ODE or variational); approximate options are listed as needing an
  approximate request and explicit consent;
* an approximate method is selected only with ``allow_approximation=True`` and
  only when its errors meet both the request tolerance and its own declared
  acceptance tolerance;
* a missing error component makes the answer ``unknown``, never a guess;
* a problem or size without measured rows is ``refused``;
* among feasible candidates the frozen objective picks the lowest measured wall
  time, then the lowest memory (unknown memory last), then the method name.

Each selection carries a canonical plan digest over the request, the candidate
evidence and the objective version, so a cache keyed by it can never reuse a
choice made for another precision or tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

NUMERICAL_METHOD_ADMISSION_SCHEMA: Final[str] = "scpn_qc_numerical_method_admission_v1"
#: Version of the frozen ranking rule; a change here changes every plan digest.
METHOD_SELECTION_OBJECTIVE: Final[str] = (
    "scpn_qc_method_objective_v1: feasible = same problem and size, same precision, "
    "every required error component measured and within the request tolerance and the "
    "candidate's acceptance tolerance, approximation allowed when not exact; then the "
    "lowest measured wall time, the lowest memory (unknown last), the method name"
)
NUMERICAL_METHOD_ADMISSION_CLAIM_BOUNDARY: Final[str] = (
    "selection among committed measured evidence rows for the stated problem, size and "
    "precision; no solver is run, no unmeasured size is extrapolated and no method is "
    "claimed optimal outside the measured candidates"
)

Exactness = Literal["exact", "approximate"]
Precision = Literal["float64", "float32"]
Family = Literal["exact", "compressed", "trajectory", "trotter", "ode", "variational"]
SelectionStatus = Literal["selected", "unknown", "consent_required", "refused"]

_PRECISIONS: Final[frozenset[str]] = frozenset({"float64", "float32"})
_FAMILIES: Final[frozenset[str]] = frozenset(
    {"exact", "compressed", "trajectory", "trotter", "ode", "variational"}
)


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be an explicit boolean")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(name: str, value: object, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


@dataclass(frozen=True)
class EvidenceReference:
    """Where a candidate's measurements come from.

    Parameters
    ----------
    path
        Repository-relative path of the committed evidence file.
    sha256
        SHA-256 of the file's exact bytes.
    schema
        Schema the file declares.

    """

    path: str
    sha256: str
    schema: str


@dataclass(frozen=True)
class MeasuredMethodCandidate:
    """One method that was actually run on one problem, with its measurements.

    Parameters
    ----------
    method
        Method name, unique within its evidence.
    backend
        Backend that ran it.
    family
        ``exact`` or the kind of approximation it makes.
    problem
        Problem identifier the run belongs to.
    n_qubits
        Problem size.
    precision
        Arithmetic precision of the run, as stated for the evidence.
    error_components
        Measured errors against the reference; ``None`` marks an unmeasured one.
    wall_time_ms
        Measured wall time.
    memory_bytes
        Measured peak memory, or ``None`` when not measured.
    acceptance_tolerance
        Error the method declares it must stay within, or ``None`` for exact.
    evidence
        Source of the measurements.
    assumptions
        Statements the evidence does not carry itself.

    """

    method: str
    backend: str
    family: Family
    problem: str
    n_qubits: int
    precision: Precision
    error_components: Mapping[str, float | None]
    wall_time_ms: float
    memory_bytes: int | None
    acceptance_tolerance: float | None
    evidence: EvidenceReference
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse malformed measurements before any selection uses them."""
        for name in ("method", "backend", "problem"):
            _require_text(name, getattr(self, name))
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
        if self.precision not in _PRECISIONS:
            raise ValueError("precision must be float64 or float32")
        if (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or self.n_qubits < 1
        ):
            raise ValueError("n_qubits must be a positive integer")
        _require_finite("wall_time_ms", self.wall_time_ms)
        if self.memory_bytes is not None:
            _require_finite("memory_bytes", self.memory_bytes)
        if self.acceptance_tolerance is not None:
            _require_finite("acceptance_tolerance", self.acceptance_tolerance, positive=True)
        for name, value in self.error_components.items():
            _require_text("error component name", name)
            if value is not None:
                _require_finite(f"error component {name}", value)


@dataclass(frozen=True)
class NumericalMethodRequest:
    """What the caller needs computed and what it accepts.

    Parameters
    ----------
    problem
        Problem identifier.
    n_qubits
        Problem size.
    exactness
        ``exact`` forbids every approximate method.
    precision
        Required arithmetic precision.
    tolerance
        Largest acceptable value of every required error component.
    required_error_components
        Error components that must be measured and within tolerance.
    allow_approximation
        Explicit consent to an approximate method; must be a boolean.

    """

    problem: str
    n_qubits: int
    exactness: Exactness
    precision: Precision
    tolerance: float
    required_error_components: tuple[str, ...]
    allow_approximation: bool = False

    def __post_init__(self) -> None:
        """Refuse a malformed request before any candidate is considered."""
        _require_text("problem", self.problem)
        if (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or self.n_qubits < 1
        ):
            raise ValueError("n_qubits must be a positive integer")
        if self.exactness not in ("exact", "approximate"):
            raise ValueError("exactness must be exact or approximate")
        if self.precision not in _PRECISIONS:
            raise ValueError("precision must be float64 or float32")
        _require_finite("tolerance", self.tolerance, positive=True)
        if not self.required_error_components:
            raise ValueError("at least one required error component must be named")
        for name in self.required_error_components:
            _require_text("required error component", name)
        _require_bool("allow_approximation", self.allow_approximation)


@dataclass(frozen=True)
class CandidateVerdict:
    """Why one candidate was or was not chosen.

    Parameters
    ----------
    method
        Candidate method.
    feasible
        Whether it meets every rule.
    reasons
        Rules it fails, empty when feasible.
    unknown
        Required error components it did not measure.

    """

    method: str
    feasible: bool
    reasons: tuple[str, ...]
    unknown: tuple[str, ...]


@dataclass(frozen=True)
class NumericalMethodSelection:
    """The planner's answer, with everything it rests on.

    Parameters
    ----------
    status
        ``selected``, ``unknown``, ``consent_required`` or ``refused``.
    selected
        Chosen candidate, only when ``status`` is ``selected``.
    verdicts
        One verdict per candidate in the requested domain, in input order.
    unknown
        Required error components no in-domain candidate measured.
    assumptions
        Statements the evidence does not carry itself.
    plan_digest
        Canonical digest of the request, candidates and objective.
    reason
        Why the status is what it is.

    """

    status: SelectionStatus
    selected: MeasuredMethodCandidate | None
    verdicts: tuple[CandidateVerdict, ...]
    unknown: tuple[str, ...]
    assumptions: tuple[str, ...]
    plan_digest: str
    reason: str
    objective: str = field(default=METHOD_SELECTION_OBJECTIVE)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record of the selection.

        Returns
        -------
        dict[str, Any]
            Schema, status, selection, verdicts, unknowns, assumptions, digest,
            objective and claim boundary.

        """
        return {
            "schema": NUMERICAL_METHOD_ADMISSION_SCHEMA,
            "status": self.status,
            "selected": None
            if self.selected is None
            else {
                "method": self.selected.method,
                "backend": self.selected.backend,
                "family": self.selected.family,
                "wall_time_ms": self.selected.wall_time_ms,
                "evidence": self.selected.evidence.path,
            },
            "verdicts": [
                {
                    "method": verdict.method,
                    "feasible": verdict.feasible,
                    "reasons": list(verdict.reasons),
                    "unknown": list(verdict.unknown),
                }
                for verdict in self.verdicts
            ],
            "unknown": list(self.unknown),
            "assumptions": list(self.assumptions),
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "objective": self.objective,
            "claim_boundary": NUMERICAL_METHOD_ADMISSION_CLAIM_BOUNDARY,
        }


def _canonical(value: object) -> object:
    """Typed canonical tree: floats as IEEE-754 binary64 hex, as the frozen design hashes them."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return {"f64": struct.pack(">d", value).hex()}
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(f"cannot canonicalise {type(value).__name__}")


def plan_digest(
    request: NumericalMethodRequest, candidates: Sequence[MeasuredMethodCandidate]
) -> str:
    """Canonical SHA-256 of a request, its candidates and the frozen objective.

    Parameters
    ----------
    request
        The request; every field, including precision and tolerance, enters.
    candidates
        Candidates in input order.

    Returns
    -------
    str
        ``sha256:`` followed by the hex digest.

    """
    tree = {
        "objective": METHOD_SELECTION_OBJECTIVE,
        "request": {
            "problem": request.problem,
            "n_qubits": request.n_qubits,
            "exactness": request.exactness,
            "precision": request.precision,
            "tolerance": float(request.tolerance),
            "required_error_components": list(request.required_error_components),
            "allow_approximation": request.allow_approximation,
        },
        "candidates": [
            {
                "method": candidate.method,
                "backend": candidate.backend,
                "family": candidate.family,
                "problem": candidate.problem,
                "n_qubits": candidate.n_qubits,
                "precision": candidate.precision,
                "error_components": {
                    name: None if value is None else float(value)
                    for name, value in candidate.error_components.items()
                },
                "wall_time_ms": float(candidate.wall_time_ms),
                "memory_bytes": candidate.memory_bytes,
                "acceptance_tolerance": None
                if candidate.acceptance_tolerance is None
                else float(candidate.acceptance_tolerance),
                "evidence": [candidate.evidence.path, candidate.evidence.sha256],
            }
            for candidate in candidates
        ],
    }
    payload = (
        NUMERICAL_METHOD_ADMISSION_SCHEMA
        + "\n"
        + json.dumps(_canonical(tree), sort_keys=True, separators=(",", ":"))
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verdict(
    request: NumericalMethodRequest, candidate: MeasuredMethodCandidate
) -> CandidateVerdict:
    reasons: list[str] = []
    if candidate.precision != request.precision:
        reasons.append(f"measured at {candidate.precision}, the request needs {request.precision}")
    approximate = candidate.family != "exact"
    if approximate and request.exactness == "exact":
        reasons.append(
            f"{candidate.family} approximation; an exact request never selects it "
            "(ask for an approximate result and give consent)"
        )
    elif approximate and not request.allow_approximation:
        reasons.append(f"{candidate.family} approximation needs explicit consent")
    unknown = tuple(
        name
        for name in request.required_error_components
        if candidate.error_components.get(name) is None
    )
    limit = request.tolerance
    if candidate.acceptance_tolerance is not None:
        limit = min(limit, candidate.acceptance_tolerance)
    for name in request.required_error_components:
        value = candidate.error_components.get(name)
        if value is not None and value > limit:
            reasons.append(f"{name} {value:.3e} exceeds {limit:.3e}")
    return CandidateVerdict(candidate.method, not reasons and not unknown, tuple(reasons), unknown)


def select_numerical_method(
    request: NumericalMethodRequest, candidates: Iterable[MeasuredMethodCandidate]
) -> NumericalMethodSelection:
    """Select a method from measured candidates under the frozen objective.

    Parameters
    ----------
    request
        What is needed and what is accepted.
    candidates
        Measured evidence rows; rows for other problems or sizes are ignored.

    Returns
    -------
    NumericalMethodSelection
        ``selected`` with the chosen candidate, ``unknown`` when a required
        error component is unmeasured, ``consent_required`` when only an
        approximation could serve, or ``refused``.

    """
    pool = tuple(candidates)
    digest = plan_digest(request, pool)
    domain = [
        candidate
        for candidate in pool
        if candidate.problem == request.problem and candidate.n_qubits == request.n_qubits
    ]
    assumptions = tuple(
        dict.fromkeys(item for candidate in domain for item in candidate.assumptions)
    )
    if not domain:
        return NumericalMethodSelection(
            "refused",
            None,
            (),
            (),
            (),
            digest,
            f"no measured method for {request.problem} at {request.n_qubits} qubits; "
            "an unmeasured size is not extrapolated",
        )
    verdicts = tuple(_verdict(request, candidate) for candidate in domain)
    feasible = [
        candidate for candidate, verdict in zip(domain, verdicts, strict=True) if verdict.feasible
    ]
    if feasible:
        chosen = min(
            feasible,
            key=lambda candidate: (
                candidate.wall_time_ms,
                math.inf if candidate.memory_bytes is None else candidate.memory_bytes,
                candidate.method,
            ),
        )
        return NumericalMethodSelection(
            "selected",
            chosen,
            verdicts,
            (),
            assumptions,
            digest,
            "lowest measured wall time among feasible candidates",
        )
    unknown = tuple(
        dict.fromkeys(
            name for verdict in verdicts if not verdict.reasons for name in verdict.unknown
        )
    )
    if unknown:
        return NumericalMethodSelection(
            "unknown",
            None,
            verdicts,
            unknown,
            assumptions,
            digest,
            "required error components were not measured: " + ", ".join(unknown),
        )
    needs_consent = any(
        candidate.family != "exact"
        and _verdict(
            NumericalMethodRequest(
                request.problem,
                request.n_qubits,
                "approximate",
                request.precision,
                request.tolerance,
                request.required_error_components,
                True,
            ),
            candidate,
        ).feasible
        for candidate in domain
    )
    if needs_consent:
        return NumericalMethodSelection(
            "consent_required",
            None,
            verdicts,
            (),
            assumptions,
            digest,
            "only an approximate method meets the tolerance; request an approximate "
            "result with explicit consent to use it",
        )
    return NumericalMethodSelection(
        "refused",
        None,
        verdicts,
        (),
        assumptions,
        digest,
        "no measured candidate meets the request",
    )


class NumericalMethodSelectionCache:
    """Selections keyed by their plan digest.

    A changed precision, tolerance, exactness, consent or candidate set gives a
    new digest, so a cached choice is never reused for a different request.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._entries: dict[str, NumericalMethodSelection] = {}

    def __len__(self) -> int:
        """Return the number of cached selections."""
        return len(self._entries)

    def select(
        self, request: NumericalMethodRequest, candidates: Sequence[MeasuredMethodCandidate]
    ) -> tuple[NumericalMethodSelection, bool]:
        """Return the selection for a request, computing it on a cache miss.

        Parameters
        ----------
        request
            What is needed.
        candidates
            Measured evidence rows.

        Returns
        -------
        tuple[NumericalMethodSelection, bool]
            The selection and whether it came from the cache.

        """
        key = plan_digest(request, candidates)
        cached = self._entries.get(key)
        if cached is not None:
            return cached, True
        selection = select_numerical_method(request, candidates)
        self._entries[key] = selection
        return selection, False


def _evidence(
    path: str, raw: bytes, payload: Mapping[str, Any], schema: str | None
) -> EvidenceReference:
    return EvidenceReference(
        path,
        hashlib.sha256(raw).hexdigest(),
        schema if schema is not None else str(payload.get("schema", "unspecified")),
    )


def _precision_assumption(precision: Precision, path: str) -> str:
    return f"{path} does not record its arithmetic precision; {precision} is stated by the caller"


def candidates_from_tn_baseline_flagship(
    raw: bytes, *, path: str, precision: Precision
) -> tuple[MeasuredMethodCandidate, ...]:
    """Read MPS bond-dimension sweeps from the committed tensor-network flagship evidence.

    Parameters
    ----------
    raw
        Exact bytes of the evidence file; they are only read.
    path
        Repository-relative path recorded with each candidate.
    precision
        Precision the caller states for the evidence, which does not record it.

    Returns
    -------
    tuple[MeasuredMethodCandidate, ...]
        One compressed candidate per workload and bond dimension, with the
        workload's ``r_tolerance`` as its acceptance tolerance.

    """
    payload = json.loads(raw)
    evidence = _evidence(path, raw, payload, None)
    rows: list[MeasuredMethodCandidate] = []
    for workload in payload["workloads"]:
        for point in workload["chi_sweep"]:
            rows.append(
                MeasuredMethodCandidate(
                    method=f"mps_chi_{point['bond_dimension']}",
                    backend="mps_tensor_network",
                    family="compressed",
                    problem=str(workload["workload"]),
                    n_qubits=int(workload["n_qubits"]),
                    precision=precision,
                    error_components={
                        "r_abs_error": float(point["r_abs_error"]),
                        "max_component_abs_error": float(point["max_component_abs_error"]),
                    },
                    wall_time_ms=float(point["wall_time_ms"]),
                    memory_bytes=None,
                    acceptance_tolerance=float(workload["r_tolerance"]),
                    evidence=evidence,
                    assumptions=(_precision_assumption(precision, path),),
                )
            )
    return tuple(rows)


#: Families of the methods the reproducible comparison runs.
_COMPARISON_FAMILIES: Final[Mapping[str, Family]] = {
    "classical_exact": "exact",
    "classical_ode": "ode",
    "quantum_trotter": "trotter",
}


def candidates_from_reproducible_comparison(
    raw: bytes, *, path: str, precision: Precision
) -> tuple[MeasuredMethodCandidate, ...]:
    """Read the methods of a committed reproducible Kuramoto comparison.

    Parameters
    ----------
    raw
        Exact bytes of the evidence file; they are only read.
    path
        Repository-relative path recorded with each candidate.
    precision
        Precision the caller states for the evidence, which does not record it.

    Returns
    -------
    tuple[MeasuredMethodCandidate, ...]
        One candidate per available method. The reference method's error
        against itself is zero by definition, recorded as an assumption.

    Raises
    ------
    ValueError
        If a method has no declared family.

    """
    payload = json.loads(raw)
    evidence = _evidence(path, raw, payload, "reproducible_kuramoto_comparison")
    problem = f"reproducible_kuramoto_n{payload['n_oscillators']}"
    reference = str(payload["reference_method"])
    rows: list[MeasuredMethodCandidate] = []
    for row in payload["rows"]:
        if not row["available"]:
            continue
        method = str(row["method"])
        if method not in _COMPARISON_FAMILIES:
            raise ValueError(f"method {method!r} has no declared family")
        assumptions = [_precision_assumption(precision, path)]
        error = row["r_error_vs_exact"]
        if method == reference:
            error = 0.0
            assumptions.append(
                f"{method} is the comparison's reference; its error is zero by definition"
            )
        rows.append(
            MeasuredMethodCandidate(
                method=method,
                backend=str(row["backend"]),
                family=_COMPARISON_FAMILIES[method],
                problem=problem,
                n_qubits=int(payload["n_oscillators"]),
                precision=precision,
                error_components={"r_abs_error": None if error is None else float(error)},
                wall_time_ms=float(row["elapsed_ms"]),
                memory_bytes=None,
                acceptance_tolerance=None,
                evidence=evidence,
                assumptions=tuple(assumptions),
            )
        )
    return tuple(rows)


def render_numerical_method_selection_markdown(selection: NumericalMethodSelection) -> str:
    """Render a selection as Markdown.

    Parameters
    ----------
    selection
        The planner's answer.

    Returns
    -------
    str
        Status, choice, per-candidate verdicts, unknowns and assumptions.

    """
    lines = [
        "# Numerical method admission",
        "",
        f"- Status: `{selection.status}` — {selection.reason}",
        f"- Plan digest: `{selection.plan_digest}`",
        f"- Objective: {selection.objective}",
        f"- Claim boundary: {NUMERICAL_METHOD_ADMISSION_CLAIM_BOUNDARY}",
    ]
    if selection.selected is not None:
        lines.append(
            f"- Selected: `{selection.selected.method}` ({selection.selected.family}, "
            f"{selection.selected.wall_time_ms:.3f} ms)"
        )
    lines += ["", "| Method | Feasible | Reasons | Unmeasured |", "| --- | --- | --- | --- |"]
    for verdict in selection.verdicts:
        lines.append(
            f"| `{verdict.method}` | {'yes' if verdict.feasible else 'no'} | "
            f"{'; '.join(verdict.reasons) or '—'} | {', '.join(verdict.unknown) or '—'} |"
        )
    if selection.assumptions:
        lines += ["", "Assumptions:", *[f"- {item}" for item in selection.assumptions]]
    return "\n".join(lines) + "\n"


__all__ = [
    "METHOD_SELECTION_OBJECTIVE",
    "NUMERICAL_METHOD_ADMISSION_CLAIM_BOUNDARY",
    "NUMERICAL_METHOD_ADMISSION_SCHEMA",
    "CandidateVerdict",
    "EvidenceReference",
    "MeasuredMethodCandidate",
    "NumericalMethodRequest",
    "NumericalMethodSelection",
    "NumericalMethodSelectionCache",
    "candidates_from_reproducible_comparison",
    "candidates_from_tn_baseline_flagship",
    "plan_digest",
    "render_numerical_method_selection_markdown",
    "select_numerical_method",
]
