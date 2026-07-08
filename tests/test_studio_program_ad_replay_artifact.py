# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed program-AD replay artefact tests
"""Tests for the committed browser-verifiable program-AD replay unit (ST-12)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("scpn_quantum_engine", reason="Rust engine (pyo3) not installed")

from scpn_quantum_control.studio import (  # noqa: E402
    program_ad_replay_artifact as artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / artifact.DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH


def test_build_carries_the_expected_value_and_gradient() -> None:
    """The rational program f(x,y)=x*x+2y replays to value 19, gradient [6, 2]."""
    payload = artifact.build_program_ad_replay_artifact()
    assert payload["schema"] == artifact.PROGRAM_AD_REPLAY_SCHEMA
    assert payload["artifact_id"] == artifact.PROGRAM_AD_REPLAY_ARTIFACT_ID
    assert payload["expected"] == {"value": 19.0, "gradient": [6.0, 2.0]}
    assert payload["program"]["inputs"] == [3.0, 5.0]
    assert payload["program"]["parameter_targets"] == ["%0", "%1"]
    assert "not a claim about transcendental" in payload["claim_boundary"]


def test_build_is_deterministic() -> None:
    """Two builds of the rational unit are byte-identical."""
    assert artifact.build_program_ad_replay_artifact() == (
        artifact.build_program_ad_replay_artifact()
    )


def test_input_hex_encodes_the_program_and_inputs() -> None:
    """The packed input carries ir_len, the IR bytes, and the f64 inputs."""
    payload = artifact.build_program_ad_replay_artifact()
    raw = bytes.fromhex(payload["input_hex"])
    ir_len = struct.unpack_from("<I", raw, 0)[0]
    ir = raw[4 : 4 + ir_len].decode("utf-8")
    assert ir == payload["program"]["effect_ir"]
    n_inputs = struct.unpack_from("<I", raw, 4 + ir_len)[0]
    assert n_inputs == 2
    inputs = list(struct.unpack_from("<2d", raw, 4 + ir_len + 4))
    assert inputs == [3.0, 5.0]
    assert len(raw) == 4 + ir_len + 4 + n_inputs * 8


def test_committed_artifact_is_current() -> None:
    """The committed JSON matches a fresh engine-verified build byte-for-byte."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    assert artifact.validate_program_ad_replay_artifact(committed)


def test_validation_rejects_a_tampered_expectation() -> None:
    """A forged gradient fails validation."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    committed["expected"]["gradient"] = [99.0, 2.0]
    assert not artifact.validate_program_ad_replay_artifact(committed)


def test_encode_replay_input_length() -> None:
    """The packed input length matches the kernel's canonical layout."""
    ir = "{}"
    packed = artifact.encode_replay_input(ir, (1.0, 2.0, 3.0))
    assert len(packed) == 4 + len(ir.encode()) + 4 + 3 * 8


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
    payload = json.loads(json.dumps(artifact.build_program_ad_replay_artifact()))
    payload["expected"]["value"] = 1.0
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
    assert printed["artifact_id"] == artifact.PROGRAM_AD_REPLAY_ARTIFACT_ID


def test_build_fails_closed_on_unsupported_program(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine result that is not a supported bounded replay is never serialised."""
    monkeypatch.setattr(
        artifact,
        "_engine_value_and_gradient",
        lambda ir: (_ for _ in ()).throw(ValueError("not supported")),
    )
    with pytest.raises(ValueError, match="not supported"):
        artifact.build_program_ad_replay_artifact()


def test_engine_replay_rejects_an_unsupported_program() -> None:
    """An empty-effects IR is not a supported bounded replay and fails closed."""
    empty = json.dumps(
        {
            "format": "program_ad_effect_ir.v1",
            "ssa_values": [],
            "effects": [],
            "alias_edges": [],
            "control_regions": [],
            "phi_nodes": [],
            "bytecode_offsets": [],
        }
    )
    with pytest.raises(ValueError, match="not a supported bounded replay"):
        artifact._engine_value_and_gradient(empty)
