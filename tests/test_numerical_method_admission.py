# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — numerical method admission over committed measured evidence
"""Methods are chosen only from measured evidence, and never silently approximate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import numerical_method_admission as admission
from scpn_quantum_control.benchmarks.numerical_method_admission import (
    EvidenceReference,
    MeasuredMethodCandidate,
    NumericalMethodRequest,
    NumericalMethodSelectionCache,
    candidates_from_reproducible_comparison,
    candidates_from_tn_baseline_flagship,
    plan_digest,
    render_numerical_method_selection_markdown,
    select_numerical_method,
)

ROOT = Path(__file__).resolve().parents[1]
FLAGSHIP = "data/tn_baseline_flagship/tn_baseline_flagship_20260717T024000Z.json"
COMPARISON = "data/classical_quantum_comparison/reproducible_comparison_n8.json"


def _bytes(path: str) -> bytes:
    return (ROOT / path).read_bytes()


@pytest.fixture
def flagship() -> tuple[MeasuredMethodCandidate, ...]:
    """MPS bond-dimension sweeps, stated as float64 evidence."""
    return candidates_from_tn_baseline_flagship(
        _bytes(FLAGSHIP), path=FLAGSHIP, precision="float64"
    )


@pytest.fixture
def comparison() -> tuple[MeasuredMethodCandidate, ...]:
    """Exact, ODE and Trotter rows of the reproducible n=8 comparison, as float64 evidence."""
    return candidates_from_reproducible_comparison(
        _bytes(COMPARISON), path=COMPARISON, precision="float64"
    )


def _request(**changes: Any) -> NumericalMethodRequest:
    base: dict[str, Any] = {
        "problem": "kuramoto8",
        "n_qubits": 8,
        "exactness": "approximate",
        "precision": "float64",
        "tolerance": 1e-3,
        "required_error_components": ("r_abs_error",),
        "allow_approximation": True,
    }
    base.update(changes)
    return NumericalMethodRequest(**base)


def test_numerical_method_admission_01(
    flagship: tuple[MeasuredMethodCandidate, ...], comparison: tuple[MeasuredMethodCandidate, ...]
) -> None:
    """An exact request never selects MPS, ODE or Trotter, even when they are cheaper."""
    before = hashlib.sha256(_bytes(FLAGSHIP) + _bytes(COMPARISON)).hexdigest()
    exact = select_numerical_method(
        _request(
            problem="reproducible_kuramoto_n8",
            exactness="exact",
            tolerance=1e-10,
            allow_approximation=True,
        ),
        flagship + comparison,
    )
    assert exact.status == "selected"
    assert exact.selected is not None and exact.selected.method == "classical_exact"
    ode = next(verdict for verdict in exact.verdicts if verdict.method == "classical_ode")
    assert ode.feasible is False and "an exact request never selects it" in ode.reasons[0]
    cheaper = [c for c in comparison if c.wall_time_ms < exact.selected.wall_time_ms]
    assert [c.family for c in cheaper] == ["ode"]
    mps_only = select_numerical_method(
        _request(exactness="exact", allow_approximation=True), flagship
    )
    assert mps_only.status == "consent_required"
    assert mps_only.selected is None
    assert all("an exact request never selects it" in v.reasons[0] for v in mps_only.verdicts)
    assert hashlib.sha256(_bytes(FLAGSHIP) + _bytes(COMPARISON)).hexdigest() == before
    with pytest.raises(ValueError, match="allow_approximation must be an explicit boolean"):
        _request(allow_approximation=1)


def test_numerical_method_admission_02(flagship: tuple[MeasuredMethodCandidate, ...]) -> None:
    """Compatible measured candidates are ranked by the frozen objective."""
    selection = select_numerical_method(_request(), flagship)
    raw = json.loads(_bytes(FLAGSHIP))
    workload = next(item for item in raw["workloads"] if item["workload"] == "kuramoto8")
    limit = min(1e-3, workload["r_tolerance"])
    oracle = min(
        (row for row in workload["chi_sweep"] if row["r_abs_error"] <= limit),
        key=lambda row: row["wall_time_ms"],
    )
    assert selection.status == "selected"
    assert selection.selected is not None
    assert selection.selected.method == f"mps_chi_{oracle['bond_dimension']}" == "mps_chi_4"
    rejected = {v.method: v.reasons for v in selection.verdicts if not v.feasible}
    assert "mps_chi_1" in rejected and "mps_chi_2" in rejected
    assert "exceeds" in rejected["mps_chi_2"][0]
    assert selection.assumptions == (
        f"{FLAGSHIP} does not record its arithmetic precision; float64 is stated by the caller",
    )


def test_numerical_method_admission_03(flagship: tuple[MeasuredMethodCandidate, ...]) -> None:
    """A required error component nobody measured makes the answer unknown."""
    selection = select_numerical_method(
        _request(required_error_components=("r_abs_error", "time_discretisation_error")),
        flagship,
    )
    assert selection.status == "unknown"
    assert selection.selected is None
    assert selection.unknown == ("time_discretisation_error",)
    assert "time_discretisation_error" in selection.reason


def test_numerical_method_admission_04(flagship: tuple[MeasuredMethodCandidate, ...]) -> None:
    """A changed precision is a different plan; the cached float64 choice is not reused."""
    cache = NumericalMethodSelectionCache()
    first, hit = cache.select(_request(), flagship)
    assert (first.status, hit) == ("selected", False)
    again, hit = cache.select(_request(), flagship)
    assert hit is True and again is first
    narrower, hit = cache.select(_request(precision="float32"), flagship)
    assert hit is False
    assert narrower.plan_digest != first.plan_digest
    assert narrower.status == "refused"
    assert all("measured at float64" in v.reasons[0] for v in narrower.verdicts)
    assert len(cache) == 2
    assert cache.select(_request(), flagship) == (first, True)


def test_out_of_envelope_problem_is_refused_not_guessed(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """A size without measured rows is refused instead of extrapolated."""
    selection = select_numerical_method(_request(n_qubits=40), flagship)
    assert selection.status == "refused"
    assert "not extrapolated" in selection.reason
    assert selection.verdicts == ()


def test_approximation_without_consent_asks_for_it(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """An approximate request without consent selects nothing."""
    selection = select_numerical_method(_request(allow_approximation=False), flagship)
    assert selection.status == "consent_required"
    assert all("needs explicit consent" in v.reasons[0] for v in selection.verdicts)


def test_nothing_meets_a_tolerance_beyond_the_evidence(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """When no candidate meets the tolerance even with consent, the answer is refused."""
    selection = select_numerical_method(_request(tolerance=1e-15), flagship)
    assert selection.status == "refused"
    assert selection.reason == "no measured candidate meets the request"


def test_selected_method_meets_its_own_acceptance_profile(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """A candidate's declared acceptance tolerance binds even when the request is looser."""
    strict = tuple(
        dataclasses.replace(candidate, acceptance_tolerance=1e-6) for candidate in flagship
    )
    selection = select_numerical_method(_request(tolerance=1.0), strict)
    assert selection.selected is not None
    chosen = selection.selected
    assert chosen.error_components["r_abs_error"] is not None
    assert (chosen.error_components["r_abs_error"] or 0.0) <= 1e-6
    assert chosen.method == "mps_chi_8"


def test_ties_break_on_memory_then_method(flagship: tuple[MeasuredMethodCandidate, ...]) -> None:
    """Equal wall times fall back to measured memory, unknown memory last, then the name."""
    base = flagship[3]
    tied = (
        dataclasses.replace(base, method="b", memory_bytes=None),
        dataclasses.replace(base, method="c", memory_bytes=10),
        dataclasses.replace(base, method="a", memory_bytes=10),
    )
    selection = select_numerical_method(_request(tolerance=1.0), tied)
    assert selection.selected is not None and selection.selected.method == "a"


def test_reproducible_comparison_reading(comparison: tuple[MeasuredMethodCandidate, ...]) -> None:
    """The reference's own error is zero by definition, and says so."""
    exact = next(candidate for candidate in comparison if candidate.method == "classical_exact")
    assert exact.error_components == {"r_abs_error": 0.0}
    assert any("reference" in item for item in exact.assumptions)
    raw = json.loads(_bytes(COMPARISON))
    raw["rows"].append({**raw["rows"][0], "method": "unknown_solver"})
    with pytest.raises(ValueError, match="no declared family"):
        candidates_from_reproducible_comparison(
            json.dumps(raw).encode(), path=COMPARISON, precision="float64"
        )
    raw["rows"] = [{**raw["rows"][0], "available": False}]
    assert (
        candidates_from_reproducible_comparison(
            json.dumps(raw).encode(), path=COMPARISON, precision="float64"
        )
        == ()
    )
    null_error = {**json.loads(_bytes(COMPARISON))}
    null_error["rows"] = [{**null_error["rows"][1], "r_error_vs_exact": None}]
    (ode,) = candidates_from_reproducible_comparison(
        json.dumps(null_error).encode(), path=COMPARISON, precision="float64"
    )
    assert ode.error_components == {"r_abs_error": None}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("problem", "", "problem must be a non-empty string"),
        ("n_qubits", 0, "n_qubits must be a positive integer"),
        ("n_qubits", True, "n_qubits must be a positive integer"),
        ("exactness", "rough", "exactness must be exact or approximate"),
        ("precision", "float16", "precision must be float64 or float32"),
        ("tolerance", 0.0, "tolerance must be finite and positive"),
        ("tolerance", "1", "tolerance must be a real number"),
        ("required_error_components", (), "at least one required error component"),
        ("required_error_components", ("",), "required error component must be"),
    ],
)
def test_malformed_requests_are_refused(field: str, value: object, message: str) -> None:
    """A malformed request is refused before any candidate is considered."""
    with pytest.raises(ValueError, match=message):
        _request(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "", "method must be a non-empty string"),
        ("family", "magic", "family must be one of"),
        ("precision", "float16", "precision must be float64 or float32"),
        ("n_qubits", -1, "n_qubits must be a positive integer"),
        ("wall_time_ms", float("nan"), "wall_time_ms must be finite"),
        ("memory_bytes", -5, "memory_bytes must be finite and non-negative"),
        ("acceptance_tolerance", 0.0, "acceptance_tolerance must be finite and positive"),
        ("error_components", {"r_abs_error": -1.0}, "error component r_abs_error must be finite"),
        ("error_components", {"": 1.0}, "error component name must be"),
    ],
)
def test_malformed_candidates_are_refused(
    flagship: tuple[MeasuredMethodCandidate, ...], field: str, value: object, message: str
) -> None:
    """Malformed measurements are refused before any selection uses them."""
    change: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(flagship[0], **change)


def test_records_and_markdown_carry_the_decision(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """The record and the Markdown name the status, choice, verdicts and digest."""
    selection = select_numerical_method(_request(), flagship)
    record = selection.to_dict()
    assert record["schema"] == admission.NUMERICAL_METHOD_ADMISSION_SCHEMA
    assert record["selected"]["method"] == "mps_chi_4"
    assert record["plan_digest"] == plan_digest(_request(), flagship)
    markdown = render_numerical_method_selection_markdown(selection)
    assert "- Selected: `mps_chi_4`" in markdown and "| `mps_chi_1` | no |" in markdown
    refused = select_numerical_method(_request(n_qubits=40), flagship)
    assert refused.to_dict()["selected"] is None
    assert "Selected:" not in render_numerical_method_selection_markdown(refused)
    assert "Assumptions:" not in render_numerical_method_selection_markdown(refused)


def test_plan_digest_follows_every_request_field(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """Tolerance, consent and candidates all change the digest; the same inputs do not."""
    digest = plan_digest(_request(), flagship)
    assert digest == plan_digest(_request(), flagship)
    assert digest != plan_digest(_request(tolerance=2e-3), flagship)
    assert digest != plan_digest(_request(allow_approximation=False), flagship)
    assert digest != plan_digest(_request(), flagship[:-1])
    with pytest.raises(TypeError, match="cannot canonicalise set"):
        admission._canonical({1})


def test_evidence_reference_records_the_file_identity(
    flagship: tuple[MeasuredMethodCandidate, ...],
) -> None:
    """Each candidate names the exact evidence bytes it came from."""
    assert flagship[0].evidence == EvidenceReference(
        FLAGSHIP,
        hashlib.sha256(_bytes(FLAGSHIP)).hexdigest(),
        json.loads(_bytes(FLAGSHIP))["schema"],
    )
