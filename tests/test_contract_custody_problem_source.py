# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark problem custody tests
"""Own complete benchmark capture and the boundary of its core projection."""

from dataclasses import asdict, fields

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.benchmarks.kuramoto_competitive_types import build_default_problem
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem
from tools.contract_custody_problem_source import benchmark_problem_source


def test_benchmark_capture_replays_all_native_fields_and_digest() -> None:
    """The frozen source must preserve the initial state, time grid and seed too."""
    source = benchmark_problem_source()
    problem = build_default_problem(**source["inputs"])
    snapshot = asdict(problem)
    for name in ("coupling", "omega", "theta0"):
        snapshot[name] = snapshot[name].tolist()
    assert source["benchmark_fields"] == snapshot
    assert set(snapshot) == {field.name for field in fields(problem)}
    assert source["benchmark_sha256"] == scp.digest_stable_core_payload(snapshot)
    assert source["benchmark_type"] == f"{type(problem).__module__}.{type(problem).__qualname__}"
    assert problem.n_oscillators == 2 and problem.n_steps == 2


def test_core_projection_is_real_but_not_identity_preserving() -> None:
    """Shared arrays do not make same-named benchmark/core problem types equivalent."""
    source = benchmark_problem_source()
    problem = build_default_problem(**source["inputs"])
    core = build_kuramoto_problem(problem.coupling, problem.omega)
    assert type(core).__name__ == type(problem).__name__
    assert type(core).__module__ != type(problem).__module__
    assert source["core_type"] == f"{type(core).__module__}.{type(core).__qualname__}"
    assert source["core_fields"] == {
        "K_nm": core.K_nm.tolist(),
        "omega": core.omega.tolist(),
        "metadata": dict(core.metadata),
    }
    assert set(source["core_fields"]) == {field.name for field in fields(core)}
    assert source["core_sha256"] == scp.digest_stable_core_payload(source["core_fields"])
    assert source["benchmark_sha256"] != source["core_sha256"]
    assert source["omitted_benchmark_fields"] == ["theta0", "t_max", "dt", "seed"]
    assert source["identity_binding"]["executed"] is False
    assert source["identity_binding"]["status"] == "proposed_refusal"
    assert source["identity_binding"]["source_type"] != source["identity_binding"]["target_type"]


def test_problem_capture_owns_independent_source_snapshots() -> None:
    """Mutating benchmark JSON cannot change the captured core or future source."""
    source = benchmark_problem_source()
    core_before = source["core_fields"]["K_nm"][0][1]
    source["benchmark_fields"]["coupling"][0][1] = 99.0
    source["benchmark_fields"]["theta0"][0] = 99.0
    assert source["core_fields"]["K_nm"][0][1] == core_before
    fresh = benchmark_problem_source()
    assert fresh["benchmark_fields"]["coupling"][0][1] != 99.0
    assert fresh["benchmark_fields"]["theta0"][0] != 99.0
