# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio readout-mitigation bundle tests
"""Tests for the schema-B ``studio.mitigation.v1`` readout-mitigation bundle."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT
    from scpn_quantum_control.studio import readout_mitigation_bundle
    from scpn_quantum_control.studio.evidence_bundle import validate_bundle
else:
    pytest.importorskip("scpn_studio_platform.evidence", reason="studio extra not installed")
    from scpn_quantum_control.differentiable_claim_ledger import REPO_ROOT
    from scpn_quantum_control.studio import readout_mitigation_bundle
    from scpn_quantum_control.studio.evidence_bundle import validate_bundle

COMMITTED_ARTIFACT = REPO_ROOT / readout_mitigation_bundle.DEFAULT_READOUT_MITIGATION_ARTIFACT_PATH


def _committed_payload() -> dict[str, Any]:
    payload: object = json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("committed readout-mitigation summary must be an object")
    return {str(key): value for key, value in payload.items()}


# --------------------------------------------------------------------------- #
# committed-artefact federation
# --------------------------------------------------------------------------- #
def test_bundle_federates_every_committed_pair_verbatim() -> None:
    """Carry every committed raw/corrected pair into a measured case."""
    payload = _committed_payload()
    bundle = readout_mitigation_bundle.build_readout_mitigation_bundle()
    assert bundle.schema == "studio.mitigation.v1"
    assert len(bundle.cases) == len(payload["pairs"])
    for case, pair in zip(bundle.cases, payload["pairs"], strict=True):
        assert case.operation_family == (
            f"readout-mitigation:{pair['dataset']}:{pair['comparison']}"
        )
        assert case.dimension == int(pair["depth"])
        assert case.status == "measured-corrected"
        assert case.error == pytest.approx(float(pair["corrected_relative"]))


def test_bundle_claim_boundary_carries_the_caveat_verbatim() -> None:
    """Preserve the state-specific confusion-inversion caveat verbatim."""
    payload = _committed_payload()
    bundle = readout_mitigation_bundle.build_readout_mitigation_bundle()
    note = bundle.claim_boundary.validity_domain.note
    assert payload["method"] in note
    assert payload["full_confusion_matrix_note"] in note
    assert bundle.claim_boundary.status.value == "bounded-support"


def test_bundle_provenance_axes_are_hardware_mitigated() -> None:
    """Describe historical measurements with bounded mitigation provenance."""
    bundle = readout_mitigation_bundle.build_readout_mitigation_bundle()
    assert bundle.substrate.value == "hardware-mitigated"
    assert bundle.evidence_kind.value == "measured"
    assert bundle.numeric_provenance.active_backend == (
        "state-specific-parity-confusion-inversion"
    )
    assert bundle.numeric_provenance.reference_backend == "raw-counts"


def test_bundle_derivation_edge_digests_the_committed_artifact() -> None:
    """Bind the committed mitigation summary by SHA-256 provenance."""
    bundle = readout_mitigation_bundle.build_readout_mitigation_bundle()
    (edge,) = bundle.derived_from
    assert edge.entity_digest.startswith("sha256:")


def test_bundle_is_admitted_by_the_federation_gate() -> None:
    """Admit the honest bounded-support bundle through the real gate."""
    validated = validate_bundle(readout_mitigation_bundle.build_readout_mitigation_bundle())
    assert validated.verdict.admitted, validated.verdict.rejections


# --------------------------------------------------------------------------- #
# fail-closed shape checks
# --------------------------------------------------------------------------- #
def _drifted_artifact(tmp_path: Path, mutate: dict[str, Any]) -> Path:
    payload = _committed_payload()
    payload.update(mutate)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_method_fails_closed(tmp_path: Path) -> None:
    """Reject a mitigation summary without its method."""
    path = _drifted_artifact(tmp_path, {"method": ""})
    with pytest.raises(ValueError, match="missing its method"):
        readout_mitigation_bundle.build_readout_mitigation_bundle(artifact_path=path)


def test_empty_pairs_fail_closed(tmp_path: Path) -> None:
    """Reject a mitigation summary without measured pairs."""
    path = _drifted_artifact(tmp_path, {"pairs": []})
    with pytest.raises(ValueError, match="has no pairs"):
        readout_mitigation_bundle.build_readout_mitigation_bundle(artifact_path=path)


def test_missing_caveat_fails_closed(tmp_path: Path) -> None:
    """Reject a summary without its confusion-matrix caveat."""
    path = _drifted_artifact(tmp_path, {"full_confusion_matrix_note": ""})
    with pytest.raises(ValueError, match="confusion-matrix caveat"):
        readout_mitigation_bundle.build_readout_mitigation_bundle(artifact_path=path)


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    """Reject a missing committed mitigation summary."""
    with pytest.raises(FileNotFoundError):
        readout_mitigation_bundle.build_readout_mitigation_bundle(
            artifact_path=tmp_path / "absent.json"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_emits_admitted_bundle_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Emit admitted schema-B JSON through the public CLI function."""
    exit_code = readout_mitigation_bundle.main(["--artifact-path", str(COMMITTED_ARTIFACT)])
    captured = capsys.readouterr()
    assert exit_code == 0
    wire = json.loads(captured.out)
    assert wire["schema"] == "studio.mitigation.v1"
    assert len(wire["cases"]) == len(_committed_payload()["pairs"])
    assert all(
        isinstance(case["error"], str) and case["status"] == "measured-corrected"
        for case in wire["cases"]
    )


def test_main_default_artifact_path(capsys: pytest.CaptureFixture[str]) -> None:
    """Use the committed readout-mitigation summary by default."""
    exit_code = readout_mitigation_bundle.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["schema"] == "studio.mitigation.v1"


def test_main_returns_one_when_the_gate_rejects(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return failure and explain a federation-gate rejection."""

    class _RejectedVerdict:
        admitted = False
        rejections = ("forced rejection",)

    class _RejectedValidation:
        verdict = _RejectedVerdict()
        bundle = readout_mitigation_bundle.build_readout_mitigation_bundle()

    monkeypatch.setattr(
        readout_mitigation_bundle, "validate_bundle", lambda bundle: _RejectedValidation()
    )
    exit_code = readout_mitigation_bundle.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forced rejection" in captured.err


def test_module_entrypoint_executes_real_bundle(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the real ``python -m`` code path under exact coverage."""
    module_name = "scpn_quantum_control.studio.readout_mitigation_bundle"
    monkeypatch.setattr(sys, "argv", [module_name])
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(module_name, run_name="__main__")
    captured = capsys.readouterr()
    assert stopped.value.code == 0
    assert json.loads(captured.out)["schema"] == "studio.mitigation.v1"


def test_installed_module_entrypoint_runs_in_subprocess() -> None:
    """Run the installed public module without mocks or provider access."""
    completed = subprocess.run(
        [sys.executable, "-m", "scpn_quantum_control.studio.readout_mitigation_bundle"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "studio.mitigation.v1"
    assert len(payload["cases"]) == len(_committed_payload()["pairs"])
