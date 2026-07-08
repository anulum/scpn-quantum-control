# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio benchmark-databank bundle tests
"""Tests for the schema-B ``studio.benchmark-databank.v1`` bundle (ST-13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio import benchmark_databank_bundle as databank  # noqa: E402
from scpn_quantum_control.studio.evidence_bundle import validate_bundle  # noqa: E402
from scpn_quantum_control.studio.verbs import (  # noqa: E402
    BENCHMARK,
    BENCHMARK_DATABANK_SCHEMA,
    evidence_schemas,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / databank.DEFAULT_BENCHMARK_DATABANK_ARTIFACT_PATH


def test_bundle_federates_every_committed_row() -> None:
    """Each committed benchmark row rides in cases[] with its size and metric."""
    payload = json.loads(COMMITTED.read_text(encoding="utf-8"))
    bundle = databank.build_benchmark_databank_bundle()
    assert bundle.schema == BENCHMARK_DATABANK_SCHEMA
    assert len(bundle.cases) == len(payload["rows"])
    families = {case.operation_family for case in bundle.cases}
    assert families == {f"benchmark:{row['benchmark']}" for row in payload["rows"]}
    # the timing caveat rides verbatim as the claim-boundary note
    assert bundle.claim_boundary.validity_domain.note == payload["timing_caveat"]


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    """The bundle passes the studio federation admission gate."""
    validated = validate_bundle(databank.build_benchmark_databank_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections


def test_schema_is_declared_on_the_benchmark_verb() -> None:
    """The databank schema is an additive product of the benchmark verb."""
    assert BENCHMARK_DATABANK_SCHEMA in BENCHMARK.produces
    assert BENCHMARK_DATABANK_SCHEMA in evidence_schemas()


def test_bundle_is_deterministic() -> None:
    """Two builds of the databank bundle are equal."""
    assert databank.build_benchmark_databank_bundle().to_dict() == (
        databank.build_benchmark_databank_bundle().to_dict()
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"rust_engine_build_knm_available": True, "parity_with_python_reference": True},
            "measured",
        ),
        ({"rust_engine_build_knm_available": False}, "native-absent"),
        (
            {"rust_engine_build_knm_available": True, "parity_with_python_reference": False},
            "measured-parity-unverified",
        ),
        ({}, "measured"),  # a family that asserts neither key stays measured
    ],
)
def test_row_status_never_upgrades(row: dict[str, object], expected: str) -> None:
    """A missing key does not demote a row; only an explicit False does."""
    assert databank._row_status(row) == expected


def test_row_metric_tolerates_both_schemas() -> None:
    """The headline metric prefers speedup, then hermiticity error, then median."""
    assert databank._row_metric({"speedup_vs_python_median": 3.0}) == 3.0
    assert databank._row_metric({"hermitian_max_abs_error": 0.0}) == 0.0
    assert databank._row_metric({"median_ms": 1.5}) == 1.5
    with pytest.raises(ValueError, match="no headline metric"):
        databank._row_metric({"benchmark": "x"})


def test_build_fails_closed_on_a_malformed_databank(tmp_path: Path) -> None:
    """An artefact with no rows or no timing caveat fails closed."""
    no_rows = tmp_path / "no_rows.json"
    no_rows.write_text(json.dumps({"rows": [], "timing_caveat": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        databank.build_benchmark_databank_bundle(artifact_path=no_rows)
    no_caveat = tmp_path / "no_caveat.json"
    no_caveat.write_text(json.dumps({"rows": [{"benchmark": "b", "n": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing its timing caveat"):
        databank.build_benchmark_databank_bundle(artifact_path=no_caveat)


def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI emits the admitted bundle as JSON and exits 0."""
    assert databank.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == BENCHMARK_DATABANK_SCHEMA
    assert len(printed["cases"]) >= 1


def test_main_reports_a_rejection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bundle the federation gate rejects exits 1 with the reasons on stderr."""
    real = validate_bundle(databank.build_benchmark_databank_bundle())

    class _Verdict:
        admitted = False
        rejections = ("forced rejection",)

    class _Rejected:
        bundle = real.bundle
        verdict = _Verdict()

    monkeypatch.setattr(databank, "validate_bundle", lambda bundle: _Rejected())
    assert databank.main([]) == 1
    assert "forced rejection" in capsys.readouterr().err
