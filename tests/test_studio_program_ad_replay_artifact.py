# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed program-AD replay artefact tests
"""Tests for the browser-verifiable, SHA-256-bound program-AD replay unit."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import runpy
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from scpn_quantum_control.studio import (  # noqa: E402
    program_ad_replay_artifact as artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_JSON = REPO_ROOT / artifact.DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH
_REQUIRE_NATIVE_ENGINE_ENV = "SCPN_PROGRAM_AD_REQUIRE_NATIVE"


class _ProgramADReplayEngine(Protocol):
    """Typed replay export shared by aggregate and native-required owners."""

    def program_ad_effect_ir_interpret_value_and_gradient(
        self,
        ir: str,
        inputs: list[float],
    ) -> str:
        """Replay one effect-IR program and return the engine JSON result."""


class _DeterministicReplayEngine(ModuleType):
    """Aggregate-test engine with the native replay export's exact contract."""

    def program_ad_effect_ir_interpret_value_and_gradient(
        self,
        ir: str,
        inputs: list[float],
    ) -> str:
        """Return the frozen rational result or an unsupported verdict."""
        assert inputs == [3.0, 5.0]
        return _engine_payload(supported=ir == artifact._canonical_ir())


def _native_engine_or_none() -> ModuleType | None:
    """Return the native engine, failing hard when CI requires it."""
    required = os.environ.get(_REQUIRE_NATIVE_ENGINE_ENV) == "1"
    try:
        engine = importlib.import_module("scpn_quantum_engine")
    except ModuleNotFoundError as exc:
        if exc.name != "scpn_quantum_engine" or required:
            raise
        return None
    export = getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None)
    if not callable(export):
        if required:
            raise RuntimeError("native engine lacks the Program-AD replay export")
        return None
    return engine


@pytest.fixture(autouse=True)
def _install_dependency_aware_engine(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Use a deterministic seam only when the native replay is unavailable."""
    selection_owners = ("_selected_engine", "_subprocess_engine_path")
    if any(owner in request.fixturenames for owner in selection_owners):
        return
    if _native_engine_or_none() is None:
        monkeypatch.setitem(
            sys.modules,
            "scpn_quantum_engine",
            _DeterministicReplayEngine("scpn_quantum_engine"),
        )


@pytest.fixture
def _selected_engine() -> ModuleType:
    """Return the required native engine or the aggregate contract seam."""
    engine = _native_engine_or_none()
    return engine or _DeterministicReplayEngine("scpn_quantum_engine")


@pytest.fixture
def _subprocess_engine_path(tmp_path: Path) -> Path | None:
    """Return an aggregate-only module path for the real CLI subprocess."""
    if _native_engine_or_none() is not None:
        return None
    seam_root = tmp_path / "program-ad-engine-seam"
    seam_root.mkdir()
    source = (
        "def program_ad_effect_ir_interpret_value_and_gradient(\n"
        "    ir: str, inputs: list[float]\n"
        ") -> str:\n"
        '    """Replay the frozen aggregate-test rational program."""\n'
        f"    if ir != {artifact._canonical_ir()!r} or inputs != [3.0, 5.0]:\n"
        '        raise ValueError("unsupported aggregate replay input")\n'
        f"    return {_engine_payload()!r}\n"
    )
    (seam_root / "scpn_quantum_engine.py").write_text(source, encoding="utf-8")
    return seam_root


def _committed_payload() -> dict[str, object]:
    """Return the committed JSON as an object-root payload."""
    decoded = cast(object, json.loads(COMMITTED_JSON.read_text(encoding="utf-8")))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def _object_field(payload: dict[str, object], key: str) -> dict[str, object]:
    """Return one asserted object-valued payload field."""
    value = payload[key]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _engine_payload(**overrides: object) -> str:
    """Return a JSON engine response with selected boundary mutations."""
    payload: dict[str, object] = {
        "supported": True,
        "value": 19.0,
        "gradient": [6.0, 2.0],
        "parameter_targets": ["%0", "%1"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_build_carries_the_expected_value_gradient_and_digest() -> None:
    """The canonical replay carries exact engine values and an input digest."""
    payload = artifact.build_program_ad_replay_artifact()
    assert payload["schema"] == artifact.PROGRAM_AD_REPLAY_SCHEMA
    assert payload["artifact_id"] == artifact.PROGRAM_AD_REPLAY_ARTIFACT_ID
    assert payload["expected"] == {"value": 19.0, "gradient": [6.0, 2.0]}
    program = _object_field(payload, "program")
    assert program["inputs"] == [3.0, 5.0]
    assert program["parameter_targets"] == ["%0", "%1"]
    assert "not a claim about transcendental" in cast(str, payload["claim_boundary"])
    input_bytes = bytes.fromhex(cast(str, payload["input_hex"]))
    assert payload["input_sha256"] == f"sha256:{hashlib.sha256(input_bytes).hexdigest()}"


def test_build_is_deterministic() -> None:
    """Two engine-backed builds of the rational unit are byte-identical."""
    assert artifact.build_program_ad_replay_artifact() == (
        artifact.build_program_ad_replay_artifact()
    )


def test_input_hex_encodes_the_program_and_inputs() -> None:
    """The packed input carries length-prefixed IR and finite f64 bindings."""
    payload = artifact.build_program_ad_replay_artifact()
    raw = bytes.fromhex(cast(str, payload["input_hex"]))
    ir_len = struct.unpack_from("<I", raw, 0)[0]
    ir = raw[4 : 4 + ir_len].decode("utf-8")
    program = _object_field(payload, "program")
    assert ir == program["effect_ir"]
    n_inputs = struct.unpack_from("<I", raw, 4 + ir_len)[0]
    assert n_inputs == 2
    inputs = list(struct.unpack_from("<2d", raw, 4 + ir_len + 4))
    assert inputs == [3.0, 5.0]
    assert len(raw) == 4 + ir_len + 4 + n_inputs * 8


def test_committed_artifact_is_current() -> None:
    """The dated v2 JSON matches a fresh engine-backed build exactly."""
    validation = artifact.inspect_program_ad_replay_artifact(_committed_payload())
    assert validation.passed
    assert validation.errors == ()
    assert artifact.validate_program_ad_replay_artifact(_committed_payload())


def test_validation_reports_tampered_missing_and_extra_fields() -> None:
    """Drift diagnostics identify mutations, missing fields, and extra fields."""
    tampered = _committed_payload()
    _object_field(tampered, "expected")["gradient"] = [99.0, 2.0]
    tampered.pop("schema")
    tampered["unexpected"] = True
    validation = artifact.inspect_program_ad_replay_artifact(tampered)
    assert not validation.passed
    assert validation.errors == (
        "missing top-level field 'schema'",
        "unexpected top-level field 'unexpected'",
        "field 'expected' does not match the regenerated artefact",
    )
    assert not artifact.validate_program_ad_replay_artifact(tampered)


@pytest.mark.parametrize("payload", [None, [], {1: "not-a-string-key"}])
def test_validation_rejects_non_object_or_non_string_key_roots(payload: object) -> None:
    """Public validation returns a diagnostic instead of raising on bad roots."""
    validation = artifact.inspect_program_ad_replay_artifact(payload)
    assert not validation.passed
    assert validation.errors == ("artefact payload must be a JSON object with string keys",)


def test_validation_verdict_invariants_hold() -> None:
    """Verdicts cannot disagree with their diagnostics."""
    with pytest.raises(ValueError, match="must not carry errors"):
        artifact.ProgramADReplayArtifactValidation(passed=True, errors=("drift",))
    with pytest.raises(ValueError, match="must explain its errors"):
        artifact.ProgramADReplayArtifactValidation(passed=False, errors=())


def test_encode_replay_input_accepts_bounded_finite_values() -> None:
    """The public packer emits the canonical little-endian layout."""
    ir = "{}"
    packed = artifact.encode_replay_input(ir, (1.0, 2.0, 3.0))
    assert len(packed) == 4 + len(ir.encode()) + 4 + 3 * 8
    assert struct.unpack_from("<I", packed, 0) == (2,)
    assert struct.unpack_from("<I", packed, 6) == (3,)


def test_encode_replay_input_enforces_structural_bounds() -> None:
    """Empty, oversized, and non-string IR inputs fail before packing."""
    with pytest.raises(TypeError, match="must be a string"):
        artifact.encode_replay_input(cast(str, b"{}"), ())
    with pytest.raises(ValueError, match="must not be empty"):
        artifact.encode_replay_input("", ())
    oversized = "x" * (artifact.MAX_PROGRAM_AD_REPLAY_IR_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        artifact.encode_replay_input(oversized, ())
    inputs = (0.0,) * (artifact.MAX_PROGRAM_AD_REPLAY_INPUTS + 1)
    with pytest.raises(ValueError, match="arity exceeds"):
        artifact.encode_replay_input("{}", inputs)


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (True, TypeError, "real scalar"),
        (cast(float, "bad"), TypeError, "real scalar"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (10**400, ValueError, "must be finite"),
    ],
)
def test_encode_replay_input_rejects_invalid_scalars(
    value: float, error: type[Exception], match: str
) -> None:
    """Boolean, non-numeric, overflowing, and non-finite bindings fail closed."""
    with pytest.raises(error, match=match):
        artifact.encode_replay_input("{}", (value,))


def test_engine_boundary_accepts_only_the_typed_finite_contract() -> None:
    """A valid Rust JSON response becomes a typed immutable replay result."""
    result = artifact._parse_engine_replay_result(_engine_payload())
    assert result.value == 19.0
    assert result.gradient == (6.0, 2.0)
    assert result.parameter_targets == ("%0", "%1")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (_engine_payload(supported=False), "not a supported bounded replay"),
        (_engine_payload(value="19"), "value must be a finite number"),
        (_engine_payload(value=10**400), "value must be a finite number"),
        (
            '{"supported":true,"value":1e309,"gradient":[6,2],"parameter_targets":["%0","%1"]}',
            "value must be a finite number",
        ),
        (_engine_payload(gradient=[]), "non-empty number list"),
        (_engine_payload(gradient=["6", 2]), r"gradient\[0\] must be a finite number"),
        (_engine_payload(gradient=[6]), "gradient arity"),
        (_engine_payload(parameter_targets=[]), "non-empty string list"),
        (_engine_payload(parameter_targets=["%0", "%0"]), "entries must be unique"),
        (_engine_payload(parameter_targets=["%1", "%0"]), "do not match"),
    ],
)
def test_engine_boundary_rejects_malformed_results(raw: str, match: str) -> None:
    """Malformed engine responses never reach the committed artefact."""
    with pytest.raises(ValueError, match=match):
        artifact._parse_engine_replay_result(raw)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("{", "not valid JSON"),
        ("[]", "must be a JSON object"),
        ('{"supported":true,"supported":false}', "duplicate JSON key"),
        ('{"value":NaN}', "non-standard JSON constant"),
    ],
)
def test_engine_boundary_rejects_noncanonical_json(raw: str, match: str) -> None:
    """Invalid, duplicate-key, non-object, and non-standard JSON is rejected."""
    with pytest.raises(ValueError, match=match):
        artifact._parse_engine_replay_result(raw)


def test_selected_engine_rejects_an_unsupported_program() -> None:
    """The selected replay boundary rejects an empty-effects program end to end."""
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


def test_selected_engine_replays_the_committed_rational_program(
    _selected_engine: ModuleType,
) -> None:
    """The selected owner reproduces the committed value and gradient."""
    engine = cast(_ProgramADReplayEngine, _selected_engine)
    raw = engine.program_ad_effect_ir_interpret_value_and_gradient(
        artifact._canonical_ir(),
        [3.0, 5.0],
    )
    result = artifact._parse_engine_replay_result(raw)
    assert result.value == 19.0
    assert result.gradient == (6.0, 2.0)


def test_native_engine_selection_fails_closed_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact CI cannot silently replace a missing or malformed native export."""

    def missing_engine(_name: str) -> ModuleType:
        raise ModuleNotFoundError("missing native engine", name="scpn_quantum_engine")

    monkeypatch.setattr(importlib, "import_module", missing_engine)
    monkeypatch.delenv(_REQUIRE_NATIVE_ENGINE_ENV, raising=False)
    assert _native_engine_or_none() is None
    monkeypatch.setenv(_REQUIRE_NATIVE_ENGINE_ENV, "1")
    with pytest.raises(ModuleNotFoundError, match="missing native engine"):
        _native_engine_or_none()

    malformed = ModuleType("scpn_quantum_engine")
    monkeypatch.setattr(importlib, "import_module", lambda _name: malformed)
    with pytest.raises(RuntimeError, match="lacks the Program-AD replay export"):
        _native_engine_or_none()


def test_native_engine_selection_does_not_mask_nested_import_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine dependency failure is never misreported as optional absence."""

    def missing_nested_dependency(_name: str) -> ModuleType:
        raise ModuleNotFoundError("missing runtime", name="native_runtime_dependency")

    monkeypatch.delenv(_REQUIRE_NATIVE_ENGINE_ENV, raising=False)
    monkeypatch.setattr(importlib, "import_module", missing_nested_dependency)
    with pytest.raises(ModuleNotFoundError, match="missing runtime"):
        _native_engine_or_none()


def test_main_check_passes_on_the_committed_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check mode confirms the v2 artifact and emits no warning."""
    assert artifact.main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "current" in captured.out
    assert captured.err == ""


def test_main_check_reports_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check mode reports exact drift diagnostics with exit code one."""
    drifted = tmp_path / "unit.json"
    payload = _committed_payload()
    _object_field(payload, "expected")["value"] = 1.0
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(drifted)]) == 1
    assert "field 'expected'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    ["{", "[]", '{"schema":"x","schema":"y"}', '{"value":Infinity}'],
)
def test_main_check_rejects_noncanonical_json(
    content: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check mode returns two for invalid, non-object, or non-standard JSON."""
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert artifact.main(["--check", "--json-path", str(path)]) == 2
    assert "unverifiable" in capsys.readouterr().err


def test_main_check_rejects_unreadable_and_non_utf8_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing and non-UTF-8 artifacts fail closed without tracebacks."""
    missing = tmp_path / "missing.json"
    assert artifact.main(["--check", "--json-path", str(missing)]) == 2
    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")
    assert artifact.main(["--check", "--json-path", str(invalid_utf8)]) == 2
    assert capsys.readouterr().err.count("unverifiable") == 2


def test_main_write_and_default_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write mode emits v2 JSON and default mode prints the same payload."""
    json_path = tmp_path / "nested" / "unit.json"
    assert artifact.main(["--write", "--json-path", str(json_path)]) == 0
    assert json_path.exists()
    assert artifact.main(["--check", "--json-path", str(json_path)]) == 0
    capsys.readouterr()
    assert artifact.main([]) == 0
    printed = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert printed["artifact_id"] == artifact.PROGRAM_AD_REPLAY_ARTIFACT_ID


def test_main_rejects_conflicting_modes() -> None:
    """Write and check are mutually exclusive CLI modes."""
    with pytest.raises(SystemExit) as raised:
        artifact.main(["--write", "--check"])
    assert raised.value.code == 2


def test_module_guard_executes_the_real_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Executing the module as ``__main__`` traverses the real guard path."""
    previous_argv = sys.argv[:]
    loaded_module = sys.modules.pop(artifact.__name__)
    sys.argv = [artifact.__name__]
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(artifact.__name__, run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.modules[artifact.__name__] = loaded_module
    assert raised.value.code == 0
    assert json.loads(capsys.readouterr().out)["schema"] == artifact.PROGRAM_AD_REPLAY_SCHEMA


def test_python_module_cli_checks_the_committed_artifact_in_a_real_process(
    _subprocess_engine_path: Path | None,
) -> None:
    """A separate Python process verifies the selected replay-engine contract."""
    environment = os.environ.copy()
    paths = [str(REPO_ROOT / "src"), str(REPO_ROOT / "oscillatools" / "src")]
    if _subprocess_engine_path is not None:
        paths.insert(0, str(_subprocess_engine_path))
    if current := environment.get("PYTHONPATH"):
        paths.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    process_root = _subprocess_engine_path or REPO_ROOT
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            artifact.__name__,
            "--check",
            "--json-path",
            str(COMMITTED_JSON),
        ],
        cwd=process_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "current" in completed.stdout
    assert completed.stderr == ""
