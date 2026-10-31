# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the committed QEC instrument data
"""The QEC instrument data comes from its source owners and stays current."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from scipy.stats import beta

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.fault_tolerant_resource_product import (  # noqa: E402
    EXCLUDED_FROM_BASE_ESTIMATE,
    run_toric_decoder_experiment,
)
from scpn_quantum_control.studio import qec_resource_instrument_artifact as artifact  # noqa: E402
from scpn_quantum_control.studio.qec_readiness_bundle import (  # noqa: E402
    DEFAULT_QEC_READINESS_ARTIFACT_PATH,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Copy the committed readiness artefact into an isolated repository root.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    Path
        Isolated root holding only the readiness artefact.

    """
    target = tmp_path / DEFAULT_QEC_READINESS_ARTIFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact.REPO_ROOT / DEFAULT_QEC_READINESS_ARTIFACT_PATH, target)
    return tmp_path


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Build the instrument data once from the real repository.

    Returns
    -------
    dict
        Freshly built payload.

    """
    return cast(dict[str, Any], artifact.build_qec_instrument_artifact())


def test_decoder_rows_repeat_their_seeded_samples(payload: dict[str, Any]) -> None:
    """Every grid point is the owner's sample, with an exact interval for its counts."""
    rows = payload["decoder"]["experiments"]
    assert len(rows) == len(artifact.DECODER_DISTANCES) * len(artifact.DECODER_ERROR_RATES)
    for row in rows:
        again = run_toric_decoder_experiment(
            row["distance"], row["physical_error_rate"], row["shots"], row["seed"]
        )
        assert again.confidence.failures == row["failures"]
        failures, trials = row["failures"], row["shots"]
        lower = 0.0 if failures == 0 else float(beta.ppf(0.025, failures, trials - failures + 1))
        upper = float(beta.ppf(0.975, failures + 1, trials - failures))
        assert row["lower"] == pytest.approx(lower, abs=1e-15)
        assert row["upper"] == pytest.approx(upper, abs=1e-15)
        assert row["measured_logical_error_reduction_claim_allowed"] is False
    assert any(row["failures"] == 0 for row in rows)
    syndromes = payload["decoder"]["single_error_syndromes"]
    assert {row["distance"] for row in syndromes} == set(artifact.DECODER_DISTANCES)
    assert all(row["corrected"] and len(row["defects"]) == 2 for row in syndromes)


def test_readiness_is_the_validated_bundle_and_reduction_is_unmeasured(
    payload: dict[str, Any],
) -> None:
    """The readiness section is admitted by the gate and never states a measured reduction."""
    readiness = payload["readiness"]
    assert readiness["admitted"] is True
    assert readiness["bundle_schema"] == "studio.qec-readiness.v1"
    assert readiness["hardware_logical_error_reduction"] == "not measured"
    assert readiness["blocked"] and readiness["supported"]
    assert all(row["lower"] <= row["rate"] <= row["upper"] for row in readiness["aggregates"])


def test_resources_change_with_the_error_rate_and_list_exclusions(
    payload: dict[str, Any],
) -> None:
    """Each assumed rate has its own estimate and digest; the threshold rate is refused."""
    resources = payload["resources"]
    assert resources["excluded_from_base_estimate"] == list(EXCLUDED_FROM_BASE_ESTIMATE)
    estimated = [row for row in resources["rows"] if row["status"] == "estimated"]
    refused = [row for row in resources["rows"] if row["status"] == "refused"]
    assert [row["physical_error_rate"] for row in refused] == [1e-2]
    assert "threshold" in refused[0]["reason"]
    digests = {row["payload_sha256"] for row in estimated}
    assert len(digests) == len(estimated)
    distances = [row["code_distance"] for row in estimated]
    assert distances == sorted(distances)
    for row in estimated:
        labels = [scenario["label"] for scenario in row["scenarios"]]
        assert labels == [scenario.label for scenario in artifact.OVERHEAD_SCENARIOS]
        assert all(
            scenario["total_physical_qubits"] > row["data_patch_physical_qubits"]
            for scenario in row["scenarios"]
        )


def test_build_uses_the_given_root(root: Path, payload: dict[str, Any]) -> None:
    """A copied readiness artefact gives the same section; a malformed one refuses."""
    built = artifact.build_qec_instrument_artifact(root)
    assert built["readiness"] == payload["readiness"]
    readiness = root / DEFAULT_QEC_READINESS_ARTIFACT_PATH
    data = json.loads(readiness.read_text(encoding="utf-8"))
    data["claim_boundary"]["blocked"] = []
    readiness.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="supported/blocked"):
        artifact.build_qec_instrument_artifact(root)


def test_main_prints_writes_and_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    """Print, write, then check current, drifted and unreadable files.

    The build itself is covered above; here it returns the module's payload so
    that every mode is exercised without sampling the decoders again.
    """
    monkeypatch.setattr(artifact, "build_qec_instrument_artifact", lambda: payload)
    assert artifact.main([]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == artifact.QEC_INSTRUMENT_SCHEMA
    target = tmp_path / "nested" / "instrument.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == artifact.serialise(payload)
    capsys.readouterr()
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    assert "current" in capsys.readouterr().out
    drifted = json.loads(target.read_text(encoding="utf-8"))
    drifted["decoder"]["experiments"][0]["failures"] += 1
    target.write_text(json.dumps(drifted), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    assert "drift" in capsys.readouterr().err
    assert artifact.main(["--check", "--json-path", str(tmp_path / "missing.json")]) == 2
    assert "unreadable" in capsys.readouterr().err


def test_committed_data_is_current(payload: dict[str, Any]) -> None:
    """The committed file equals a fresh build within the budget."""
    committed = json.loads(
        (artifact.REPO_ROOT / artifact.DEFAULT_QEC_INSTRUMENT_JSON_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert artifact.differences(committed, payload) == []


def test_module_runs_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m ... --check`` exits with the check's status."""
    monkeypatch.setattr(sys, "argv", ["qec_resource_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(
            "scpn_quantum_control.studio.qec_resource_instrument_artifact", run_name="__main__"
        )
    assert exited.value.code == 0
