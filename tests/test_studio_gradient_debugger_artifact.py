# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the committed gradient-debugger units
"""The gradient-debugger units come from their source owners and stay current."""

from __future__ import annotations

import json
import math
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("scpn_quantum_engine", reason="Rust engine (pyo3) not installed")
pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio import executive_differentiate  # noqa: E402
from scpn_quantum_control.studio import gradient_debugger_artifact as artifact  # noqa: E402
from scpn_quantum_control.studio.program_ad_replay_artifact import (  # noqa: E402
    encode_replay_input,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Copy the plan and reference artefacts into an isolated repository root."""
    for relative in (artifact.PLAN_PATH, artifact.REFERENCE_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact.REPO_ROOT / relative, target)
    return tmp_path


def _units(payload: dict[str, object]) -> dict[str, dict[str, Any]]:
    units = cast(list[dict[str, Any]], payload["units"])
    return {unit["unit_id"]: unit for unit in units}


def test_ry_z_unit_gradient_is_minus_sin_and_matches_the_owner_reference(root: Path) -> None:
    """RY(0.4) then Z: value cos 0.4, gradient -sin 0.4, linked to the exact-state reference."""
    unit = _units(artifact.build_gradient_debugger_artifact(root))["ry-z-expectation"]
    assert unit["expected"]["value"] == pytest.approx(math.cos(0.4), abs=1e-15)
    assert unit["expected"]["gradient"] == pytest.approx([-math.sin(0.4)], abs=1e-15)
    reference = unit["reference"]
    assert reference["case_id"] == "single_ry_z_expectation_exact_state"
    budget = 1e-10 + 1e-8 * abs(reference["gradient"][0])
    assert abs(unit["expected"]["gradient"][0] - reference["gradient"][0]) <= budget
    assert {row["backend"] for row in reference["frameworks"]} == {"qiskit", "pennylane"}
    assert unit["finite_difference"]["verified"] is True
    assert unit["cell_id"] == "ry::pauli_expectation::statevector::grad::native"


def test_units_carry_the_exact_wasm_replay_input_and_every_trainable_mask(root: Path) -> None:
    """The replay bytes match the encoder; frozen parameters have zero gradient."""
    units = cast(list[dict[str, Any]], artifact.build_gradient_debugger_artifact(root)["units"])
    for unit in units:
        values = [parameter["value"] for parameter in unit["parameters"]]
        replay = encode_replay_input(unit["effect_ir"], values)
        assert unit["input_hex"] == replay.hex()
        masks = unit["masks"]
        assert len(masks) == 2 ** len(values)
        for mask in masks:
            for trainable, component, full in zip(
                mask["trainable"], mask["gradient"], unit["expected"]["gradient"], strict=True
            ):
                assert component == (pytest.approx(full, abs=1e-12) if trainable else 0.0)
        assert unit["local_runner"]["params"] == unit["program"]


def test_build_refuses_an_engine_refusal(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit the engine cannot replay is refused, not committed."""
    monkeypatch.setattr(
        executive_differentiate,
        "_engine_value_and_gradient",
        lambda _ir, _values: (_ for _ in ()).throw(RuntimeError("engine refused")),
    )
    with pytest.raises(ValueError, match="differentiate refused unit 'ry-z-expectation'"):
        artifact.build_gradient_debugger_artifact(root)


def test_build_refuses_disagreeing_finite_differences(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gradient finite differences do not confirm is refused."""
    monkeypatch.setattr(executive_differentiate, "_GRADIENT_TOLERANCE", 0.0)
    with pytest.raises(ValueError, match="finite differences disagree"):
        artifact.build_gradient_debugger_artifact(root)


def test_build_refuses_an_undeclared_plan_cell(root: Path) -> None:
    """A unit may only link a cell the plan declares."""
    plan = root / artifact.PLAN_PATH
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["explanations"] = []
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="links an undeclared plan cell"):
        artifact.build_gradient_debugger_artifact(root)


def test_build_refuses_a_missing_reference_case(root: Path) -> None:
    """The RY/Z unit needs its reference row."""
    reference = root / artifact.REFERENCE_PATH
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["rows"] = []
    reference.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reference case 'single_ry_z_expectation_exact_state'"):
        artifact.build_gradient_debugger_artifact(root)


@pytest.mark.parametrize(
    ("committed", "fresh", "expected"),
    [
        (1.0, 1.0 + 1e-12, []),
        (1.0, 1.0 + 1e-6, ["$: 1.0 is outside the budget of 1.000001"]),
        (True, 1.0, ["$: True is outside the budget of 1.0"]),
        ({"a": 1}, {"b": 1}, ["$: keys ['a'] != ['b']"]),
        ([1, 2], [1], ["$: length 2 != 1"]),
        ({"a": [1.0, "x"]}, {"a": [1.0, "y"]}, ["$.a[1]: 'x' != 'y'"]),
        ("same", "same", []),
    ],
)
def test_differences_tolerate_float_budget_only(
    committed: object, fresh: object, expected: list[str]
) -> None:
    """Floats agree within the frozen budget; everything else must be equal."""
    assert artifact.differences(committed, fresh) == expected


def test_main_prints_writes_and_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints, writes, accepts a current file and reports drift or unreadable files."""
    target = tmp_path / "units.json"
    assert artifact.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == artifact.GRADIENT_DEBUGGER_SCHEMA
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    assert "current" in capsys.readouterr().out
    drifted = json.loads(target.read_text(encoding="utf-8"))
    drifted["units"][0]["expected"]["value"] += 1e-3
    target.write_text(json.dumps(drifted), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    assert "drift: $.units[0].expected.value" in capsys.readouterr().err
    target.write_text("{", encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(target)]) == 2
    assert artifact.main(["--check", "--json-path", str(tmp_path / "absent.json")]) == 2


def test_committed_units_are_current() -> None:
    """The committed artefact matches a fresh build from the source owners."""
    committed_path = artifact.REPO_ROOT / artifact.DEFAULT_GRADIENT_DEBUGGER_JSON_PATH
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    fresh = json.loads(artifact.serialise(artifact.build_gradient_debugger_artifact()))
    assert artifact.differences(committed, fresh) == []


def test_module_entry_point_runs_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``python -m`` runs the same checker."""
    target = tmp_path / "units.json"
    artifact.main(["--write", "--json-path", str(target)])
    monkeypatch.setattr(
        sys, "argv", ["gradient_debugger_artifact", "--check", "--json-path", str(target)]
    )
    monkeypatch.delitem(sys.modules, "scpn_quantum_control.studio.gradient_debugger_artifact")
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(
            "scpn_quantum_control.studio.gradient_debugger_artifact", run_name="__main__"
        )
    assert exit_info.value.code == 0
