# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed XY-compile recompute artefact tests
"""Tests for the committed browser-verifiable recompute unit artefact (ST-09)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27  # noqa: E402
from scpn_quantum_control.studio import xy_compile_recompute_artifact as artifact  # noqa: E402
from scpn_quantum_control.studio.recompute_kernel import (  # noqa: E402
    XY_COMPILE_RECOMPUTE_SCHEMA,
    build_xy_compile_recompute_unit,
    verify_xy_compile_recompute_unit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / artifact.DEFAULT_XY_COMPILE_RECOMPUTE_JSON_PATH


def _payload_with_unit(
    K_nm: np.ndarray,
    omega: np.ndarray,
    *,
    time: float = 0.1,
    trotter_steps: int = 1,
    trotter_order: int = 1,
) -> dict[str, Any]:
    """Return a Paper-27-metadata payload wrapping a custom recompute unit.

    The wrapper metadata stays the committed Paper-27 constants; only the
    embedded unit is swapped, so validation exercises the unit-binding path
    rather than the metadata gate.
    """
    payload = artifact.build_xy_compile_recompute_artifact()
    unit = build_xy_compile_recompute_unit(
        K_nm, omega, time=time, trotter_steps=trotter_steps, trotter_order=trotter_order
    )
    payload["unit"] = unit.to_dict()
    return payload


def test_payload_carries_a_reference_verified_recompute_unit() -> None:
    """The built payload wraps a unit that verifies against the reference."""
    payload = artifact.build_xy_compile_recompute_artifact()
    assert payload["schema"] == artifact.XY_COMPILE_RECOMPUTE_ARTIFACT_SCHEMA
    assert payload["artifact_id"] == artifact.XY_COMPILE_RECOMPUTE_ARTIFACT_ID
    unit = payload["unit"]
    assert isinstance(unit, dict)
    assert unit["schema"] == XY_COMPILE_RECOMPUTE_SCHEMA
    assert unit["verifiability_mode"] == "recompute"
    assert unit["exactness_class"] == "bit-exact"
    assert str(unit["claimed_digest"]).startswith("sha256:")
    assert verify_xy_compile_recompute_unit(unit).value == "match"


def test_payload_is_deterministic() -> None:
    """Two builds of the Paper-27 unit are byte-identical."""
    assert artifact.build_xy_compile_recompute_artifact() == (
        artifact.build_xy_compile_recompute_artifact()
    )


def test_compile_parameters_are_recorded_for_audit() -> None:
    """The provisional Paper-27 compile parameters ride in the artefact."""
    payload = artifact.build_xy_compile_recompute_artifact()
    parameters = payload["compile_parameters"]
    assert parameters == {
        "matrix_source": "paper27",
        "lattice": 16,
        "time": 0.1,
        "trotter_steps": 1,
        "trotter_order": 1,
    }
    assert "not a physical K_nm claim" in str(payload["claim_boundary"])


def test_committed_artifact_is_current() -> None:
    """The committed JSON matches a fresh build byte-for-byte."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    assert artifact.validate_xy_compile_recompute_artifact(committed)


def test_validation_rejects_a_tampered_payload() -> None:
    """A tampered committed payload fails validation."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    committed["unit"]["claimed_digest"] = "sha256:" + "0" * 64
    assert not artifact.validate_xy_compile_recompute_artifact(committed)


def test_main_check_passes_on_the_committed_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI check mode confirms the committed artefact is current."""
    assert artifact.main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "current" in captured.out
    assert captured.err == ""


def test_main_check_fails_on_drift(tmp_path: Path) -> None:
    """The CLI check mode reports drift and exits 1."""
    drifted = tmp_path / "unit.json"
    payload = json.loads(json.dumps(artifact.build_xy_compile_recompute_artifact()))
    payload["unit"]["claimed_digest"] = "sha256:" + "1" * 64
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(drifted)]) == 1


def test_main_write_and_default_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write mode emits the artefact and default mode prints it."""
    json_path = tmp_path / "nested" / "unit.json"
    assert artifact.main(["--write", "--json-path", str(json_path)]) == 0
    assert json_path.exists()
    assert artifact.main(["--check", "--json-path", str(json_path)]) == 0
    capsys.readouterr()
    assert artifact.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["artifact_id"] == artifact.XY_COMPILE_RECOMPUTE_ARTIFACT_ID


def test_build_fails_closed_when_the_reference_rejects_the_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit that fails its own reference verification is never serialised."""

    class _Mismatch:
        value = "mismatch"

    monkeypatch.setattr(artifact, "verify_xy_compile_recompute_unit", lambda unit: _Mismatch())
    with pytest.raises(ValueError, match="failed its own reference verification"):
        artifact.build_xy_compile_recompute_artifact()


def test_validation_rejects_wrapper_metadata_drift() -> None:
    """A payload whose fixed wrapper metadata was altered fails validation."""
    payload = artifact.build_xy_compile_recompute_artifact()
    payload["artifact_id"] = "studio-xy-compile-recompute-tampered"
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_rejects_a_non_dict_unit() -> None:
    """A payload whose unit is not a mapping fails validation."""
    payload = artifact.build_xy_compile_recompute_artifact()
    payload["unit"] = "not-a-unit"
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_rejects_a_structurally_invalid_unit() -> None:
    """A unit the reference verifier raises on fails validation, not crashes."""
    payload = artifact.build_xy_compile_recompute_artifact()
    del payload["unit"]["schema"]
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"time": 0.2},
        {"trotter_steps": 2},
        {"trotter_order": 2},
    ],
)
def test_validation_rejects_foreign_compile_parameters(override: dict[str, float]) -> None:
    """A self-verifying unit built with non-Paper-27 parameters is rejected."""
    payload = _payload_with_unit(build_knm_paper27(L=16), OMEGA_N_16, **override)
    # the swapped unit still self-verifies against its own inputs...
    assert verify_xy_compile_recompute_unit(payload["unit"]).value == "match"
    # ...but its parameters no longer bind to the committed Paper-27 constants.
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_rejects_a_different_lattice() -> None:
    """A self-verifying unit of a different lattice size fails the shape bind."""
    payload = _payload_with_unit(build_knm_paper27(L=4), OMEGA_N_16[:4])
    assert verify_xy_compile_recompute_unit(payload["unit"]).value == "match"
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_rejects_a_matrix_that_left_paper27() -> None:
    """A coupling that drifted well past tolerance fails validation."""
    K = build_knm_paper27(L=16).copy()
    K[0, 5] = K[5, 0] = K[0, 5] + 1e-3
    payload = _payload_with_unit(K, OMEGA_N_16)
    assert verify_xy_compile_recompute_unit(payload["unit"]).value == "match"
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_rejects_frequencies_that_left_paper27() -> None:
    """Natural frequencies that drifted past tolerance fail validation."""
    omega = OMEGA_N_16.copy()
    omega[0] = omega[0] + 1e-3
    payload = _payload_with_unit(build_knm_paper27(L=16), omega)
    assert verify_xy_compile_recompute_unit(payload["unit"]).value == "match"
    assert not artifact.validate_xy_compile_recompute_artifact(payload)


def test_validation_tolerates_sub_tolerance_platform_drift() -> None:
    """A last-ULP-scale coupling drift (the CI np.exp case) still validates.

    This is the regression guard: the committed digest can never be rebuilt
    bit-exactly off-host, so validation must accept a sub-tolerance difference
    between the frozen input and a fresh Paper-27 build.
    """
    K = build_knm_paper27(L=16).copy()
    K[0, 5] = K[5, 0] = K[0, 5] + 1e-15
    payload = _payload_with_unit(K, OMEGA_N_16)
    assert verify_xy_compile_recompute_unit(payload["unit"]).value == "match"
    assert artifact.validate_xy_compile_recompute_artifact(payload)
