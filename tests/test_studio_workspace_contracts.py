# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace document contract tests
"""Exercise all public document parsers and immutable wire round-trips."""

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio_workspace import (
    WorkspaceDocument,
    parse_document,
    parse_experiment_revision,
    parse_local_run_record,
    parse_parameter_spec,
    parse_resolved_settings,
    parse_workspace_manifest,
    validate_parameter_binding,
)
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json

_CORPUS = json.loads((Path(__file__).parent / "data/studio_workspace/documents.json").read_text())


@pytest.mark.parametrize(
    "case", [row for row in _CORPUS["cases"] if "fixture" in row], ids=lambda row: row["id"]
)
def test_document_corpus(case: dict[str, object]) -> None:
    """Separate structural refusals from graph and execution admission."""
    name = str(case["fixture"])
    payload = _CORPUS["fixtures"][name]
    original = copy.deepcopy(payload)
    if case["expectation"] == "reject":
        with pytest.raises(ValueError):
            parse_document(payload)
    else:
        parsed = parse_document(payload)
        assert parsed.to_dict() == original
        assert (
            parse_document(cast(dict[str, object], read_json(write_json(parsed.to_dict())))).digest
            == parsed.digest
        )
    assert payload == original


def test_named_parsers_and_recursive_custody() -> None:
    """Every named schema parser snapshots nested data and refuses another schema."""
    pairs = [
        (parse_workspace_manifest, "workspace"),
        (parse_experiment_revision, "revision_root"),
        (parse_parameter_spec, "parameter"),
        (parse_resolved_settings, "settings"),
        (parse_local_run_record, "run"),
    ]
    for parser, name in pairs:
        original = copy.deepcopy(_CORPUS["fixtures"][name])
        result = parser(original)
        digest = result.digest
        original["extensions"]["later"] = [1]
        exported = result.to_dict()
        cast(dict[str, object], exported["extensions"])["later"] = [2]
        assert result.digest == digest
        assert "later" not in result.extensions
        with pytest.raises(ValueError):
            parser({"schema": "unknown.v1", "body": {}, "extensions": {}})
        with pytest.raises(ValueError):
            parser(_CORPUS["fixtures"]["run" if name == "workspace" else "workspace"])


def test_parameter_binding_keeps_units_shape_and_domain() -> None:
    """Bind typed values to the supplied spec without inferring conversions."""
    payload = copy.deepcopy(_CORPUS["fixtures"]["parameter"])
    payload["body"]["domain"] = {
        "kind": "closed_interval",
        "lower": "0000000000000000",
        "upper": "3ff0000000000000",
    }
    spec = parse_parameter_spec(payload)
    values = {"dtype": "float64", "shape": [2], "values": ["8000000000000000", "3ff0000000000000"]}
    validate_parameter_binding(spec, values, "rad")
    for changed, unit in [
        (values, "Hz"),
        ({**values, "shape": [1]}, "rad"),
        ({**values, "values": ["4000000000000000", "3ff0000000000000"]}, "rad"),
    ]:
        with pytest.raises(ValueError):
            validate_parameter_binding(spec, changed, unit)


def test_enumerated_integer_domain_and_empty_tensor() -> None:
    """Validate exact integer elements and zero-sized tensors without allocation."""
    payload = copy.deepcopy(_CORPUS["fixtures"]["parameter"])
    payload["body"].update(
        dtype="int64",
        unit="1",
        shape=[1],
        domain={"kind": "enumerated", "values": ["9007199254740993"]},
    )
    spec = parse_parameter_spec(payload)
    validate_parameter_binding(
        spec, {"dtype": "int64", "shape": [1], "values": ["9007199254740993"]}, "1"
    )
    with pytest.raises(ValueError):
        validate_parameter_binding(spec, {"dtype": "int64", "shape": [1], "values": ["1"]}, "1")
    payload["body"].update(shape=[0, 9007199254740991], domain={"kind": "finite"})
    validate_parameter_binding(
        parse_parameter_spec(payload),
        {"dtype": "int64", "shape": [0, 9007199254740991], "values": []},
        "1",
    )


_STRUCTURAL = cast(
    dict[str, object],
    read_json((Path(__file__).parent / "data/studio_workspace/structural.json").read_text()),
)


@pytest.mark.parametrize(
    "case", cast(list[dict[str, object]], _STRUCTURAL["cases"]), ids=lambda row: row["id"]
)
def test_shared_structural_boundaries(case: dict[str, object]) -> None:
    """Apply the same portable admission cases as the browser public parser."""
    payload = copy.deepcopy(_CORPUS["fixtures"][str(case["fixture"])])
    path = cast(list[str], case["path"])
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = case["value"]
    if case["accept"]:
        document = parse_document(payload)
        exported = document.to_dict()
        restored = parse_document(cast(dict[str, object], read_json(write_json(exported))))
        assert restored.digest == document.digest
        assert exported["extensions"] == payload["extensions"]
    else:
        with pytest.raises(ValueError):
            parse_document(payload)


def test_each_required_field_and_named_kind() -> None:
    """Refuse incomplete schemas and supported documents of another kind."""
    for name in ("workspace", "revision_root", "parameter", "settings", "run"):
        original = _CORPUS["fixtures"][name]
        for field in original["body"]:
            payload = copy.deepcopy(original)
            del payload["body"][field]
            with pytest.raises(ValueError, match="missing or unknown field"):
                parse_document(payload)
    with pytest.raises(ValueError):
        parse_parameter_spec(_CORPUS["fixtures"]["workspace"])


@pytest.mark.parametrize(
    "payload",
    [None, [], {"schema": 1}, {"schema": "quantum_workspace.v1", "body": [], "extensions": {}}],
)
def test_untrusted_envelope_is_rejected(payload: object) -> None:
    """Refuse non-object input at the public parse boundary without coercion."""
    with pytest.raises(ValueError):
        parse_document(payload)


def test_base_document_cannot_invent_a_schema() -> None:
    """Require a concrete supported schema for direct public construction."""
    with pytest.raises(ValueError, match="unsupported workspace schema"):
        WorkspaceDocument({})


def test_child_revision_edit_preserves_parent_and_original_child() -> None:
    """Editing an exported child never changes the immutable input ancestry."""
    parent = parse_experiment_revision(_CORPUS["fixtures"]["revision_root"])
    child = parse_experiment_revision(_CORPUS["fixtures"]["revision_child"])
    before = write_json(parent.to_dict()), child.digest
    edited = child.to_dict()
    cast(dict[str, object], edited["extensions"])["measurement_order"] = ["q1", "q0"]
    changed = parse_experiment_revision(edited)
    assert changed.digest != child.digest
    assert changed.body["parent_revision_hashes"] == (parent.digest,)
    assert (write_json(parent.to_dict()), child.digest) == before
