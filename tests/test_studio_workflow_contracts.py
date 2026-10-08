# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — typed workflow graph public contracts
"""Exercise admission, source immutability and typed graph refusal before execution."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.workflow_contracts import (
    MAX_WORKFLOW_BYTES,
    WorkflowPortType,
    parse_workflow,
    topological_order,
    validate_port_value,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_bytes
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json


def workflow_payload() -> dict[str, object]:
    """Read the shared synthetic graph through the original lossless transport.

    Returns
    -------
    dict[str, object]
        An independent source graph; it contains no numerical execution claim.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    return cast(dict[str, object], corpus["workflow"])


def test_graph_and_lossless_json_share_immutable_definition_and_order() -> None:
    """Admit an out-of-order real compiler graph without mutating its source."""
    original = workflow_payload()
    before = canonical_bytes("fixture.v1", original)
    definition = parse_workflow(original)
    assert topological_order(definition) == ("source", "trace")
    restored = parse_workflow(read_json(write_json(definition.to_dict())))
    assert canonical_bytes("fixture.v1", restored.to_dict()) == before
    assert canonical_bytes("fixture.v1", original) == before
    body = cast(dict[str, object], original["body"])
    body["workflow_id"] = "changed"
    assert definition.workflow_id == "compile-sweep"
    mutable = cast(MutableMapping[str, object], definition.stages[0].parameters)
    with pytest.raises(TypeError):
        mutable["compiler_trace"] = False


@pytest.mark.parametrize(
    "fault",
    ["cycle", "unit", "schema", "dtype", "shape", "dangling", "duplicate", "approval", "version"],
)
def test_public_graph_admission_refuses_incompatible_or_ambiguous_source(fault: str) -> None:
    """Refuse invalid graph semantics before producing an executable definition.

    Parameters
    ----------
    fault
        Concrete source mutation; the original admitted fixture is retained.

    """
    source = workflow_payload()
    body = cast(dict[str, object], source["body"])
    stages = cast(list[dict[str, object]], body["stages"])
    incoming = cast(list[dict[str, object]], stages[0]["inputs"])[0]
    port = cast(dict[str, object], incoming["type"])
    if fault == "cycle":
        stages[1]["depends_on"] = ["trace"]
    elif fault in ("unit", "schema", "dtype", "shape"):
        port[fault] = {"unit": "rad", "schema": "other-source.v1", "dtype": "json", "shape": [1]}[
            fault
        ]
    elif fault == "dangling":
        incoming["source_stage"] = "missing"
    elif fault == "duplicate":
        stages.append(deepcopy(stages[0]))
    elif fault == "approval":
        stages[0]["approved"] = True
    else:
        source["schema"] = "experiment_workflow.v2"
    with pytest.raises(ValueError):
        parse_workflow(source)


def test_unknown_extension_preserves_large_integer_float_and_signed_zero() -> None:
    """Roundtrip opaque source metadata without numeric coercion or authority."""
    source = workflow_payload()
    source["extensions"] = {"integer": 9007199254740993, "float": 1.0, "zero": -0.0}
    first = parse_workflow(source)
    second = parse_workflow(read_json(write_json(first.to_dict())))
    assert canonical_bytes("fixture.v1", first.to_dict()) == canonical_bytes(
        "fixture.v1", second.to_dict()
    )
    assert second.extensions["integer"] == 9007199254740993


def test_source_port_values_are_validated_without_converting_original_types() -> None:
    """Validate original text and reject a different runtime value type."""
    definition = parse_workflow(workflow_payload())
    port = definition.stages[1].outputs[0].type
    original = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
    assert validate_port_value(port, original) == original
    with pytest.raises(ValueError):
        validate_port_value(port, 1)


def test_public_order_refuses_a_directly_constructed_duplicate_graph() -> None:
    """Prevent a caller-created duplicate from silently dropping a stage."""
    definition = parse_workflow(workflow_payload())
    duplicate = replace(definition, stages=(definition.stages[0], definition.stages[0]))
    with pytest.raises(ValueError, match="unique"):
        topological_order(duplicate)


def test_original_uppercase_parameter_names_are_preserved_in_sweep_bindings() -> None:
    """Admit the original executive K_nm parameter without renaming the model."""
    source = workflow_payload()
    body = cast(dict[str, object], source["body"])
    network: dict[str, object] = {
        "id": "network",
        "adapter": "executive",
        "verb": "compile",
        "backend": "python",
        "parameters": {
            "K_nm": [[0.0, 0.1], [0.1, 0.0]],
            "omega": [0.0, 0.0],
            "time": 0.1,
            "trotter_steps": 1,
            "trotter_order": 1,
        },
        "inputs": [],
        "outputs": [],
        "depends_on": [],
    }
    body["stages"] = [network]
    body["sweep"] = {
        "axes": [
            {
                "stage_id": "network",
                "parameter": "K_nm",
                "values": [[[0.0, 0.1], [0.1, 0.0]], [[0.0, 0.2], [0.2, 0.0]]],
            }
        ],
        "seeds": ["0"],
        "seed_binding": None,
        "evaluation_budget": 2,
    }
    definition = parse_workflow(source)
    assert definition.sweep.axes[0][1] == "K_nm"
    assert "K_nm" in definition.stages[0].parameters
    recovered = parse_workflow(read_json(write_json(definition.to_dict())))
    assert canonical_bytes("matrix-workflow.v1", recovered.to_dict()) == canonical_bytes(
        "matrix-workflow.v1", source
    )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("body-array", "workflow body: object required"),
        ("empty-graph", "nonempty unique workflow stages"),
        ("missing-control-parent", "workflow dependency is absent"),
        ("adapter", "unsupported workflow adapter or verb"),
        ("local-verb", "local classical adapter does not implement"),
        ("backend-control-text", "backend: bounded nonempty text"),
        ("port-dtype", "unsupported port schema or dtype"),
        ("json-port-shape", "unsupported port shape"),
        ("port-dimension", "port dimension: integer between"),
        ("port-shape-product", "unsupported port shape"),
        ("output-path", "nonempty original output path"),
        ("duplicate-input", "duplicate stage port or dependency"),
        ("constant-input", "a bound input cannot also have a constant"),
        ("input-parameter-key", "original ASCII parameter key"),
        ("empty-axis", "nonempty unique sweep coordinates"),
        ("duplicate-axis", "nonempty unique sweep coordinates"),
        ("bound-axis", "sweep target is absent or already bound"),
        ("empty-seeds", "unique canonical uint64 seeds"),
        ("overflow-seed", "unique canonical uint64 seeds"),
        ("axis-seed-binding", "seed target is absent or already has a value"),
        ("missing-seed-target", "seed target is absent or already has a value"),
        ("zero-budget", "evaluation budget: integer between"),
        ("extensions-array", "extensions: object required"),
        ("stage-identifier", "lowercase ASCII identifier"),
        ("definition-byte-bound", "workflow definition exceeds byte bound"),
        ("stages-object", "stages: bounded array"),
        ("stage-count", "stages: bounded array"),
        ("constant-seed-target", "seed target is absent or already has a value"),
        ("bound-seed-target", "seed target is absent or already has a value"),
        ("insufficient-budget", "sweep exceeds cell or stage evaluation budget"),
        ("edge-count", "workflow edge bound exceeded"),
    ],
)
def test_malformed_original_graph_matches_browser_refusal_and_preserves_caller(
    fault: str, reason: str
) -> None:
    """Exercise the same source corruption matrix as the original browser graph parser.

    Parameters
    ----------
    fault
        Concrete corruption of the original shared compiler workflow document.
    reason
        Exact shared refusal category required before execution or mutation.

    """
    source = workflow_payload()
    body = cast(dict[str, object], source["body"])
    stages = cast(list[dict[str, object]], body["stages"])
    inputs = cast(list[dict[str, object]], stages[0]["inputs"])
    outputs = cast(list[dict[str, object]], stages[1]["outputs"])
    port = cast(dict[str, object], inputs[0]["type"])
    sweep = cast(dict[str, object], body["sweep"])
    axes = cast(list[dict[str, object]], sweep["axes"])
    if fault == "body-array":
        source["body"] = []
    elif fault == "empty-graph":
        body["stages"] = []
    elif fault == "missing-control-parent":
        stages[0]["depends_on"] = ["missing"]
    elif fault == "adapter":
        stages[0]["adapter"] = "provider"
    elif fault == "local-verb":
        stages[0].update(adapter="local-kuramoto", verb="compile")
    elif fault == "backend-control-text":
        stages[0]["backend"] = "invalid\nbackend"
    elif fault == "port-dtype":
        port["dtype"] = "complex"
    elif fault == "json-port-shape":
        port.update(dtype="json", shape=[1])
    elif fault == "port-dimension":
        port["shape"] = [4097]
    elif fault == "port-shape-product":
        port["shape"] = [64, 65]
    elif fault == "output-path":
        outputs[0]["path"] = []
    elif fault == "duplicate-input":
        inputs.append(deepcopy(inputs[0]))
    elif fault == "constant-input":
        cast(dict[str, object], stages[0]["parameters"])[cast(str, inputs[0]["parameter"])] = (
            "already-bound"
        )
    elif fault == "input-parameter-key":
        inputs[0]["parameter"] = "invalid-key"
    elif fault == "empty-axis":
        axes[0]["values"] = []
    elif fault == "duplicate-axis":
        axes.append(deepcopy(axes[0]))
    elif fault == "bound-axis":
        axes[0].update(stage_id="trace", parameter=inputs[0]["parameter"])
    elif fault == "empty-seeds":
        sweep["seeds"] = []
    elif fault == "overflow-seed":
        sweep["seeds"] = ["18446744073709551616"]
    elif fault == "axis-seed-binding":
        sweep["seed_binding"] = {
            "stage_id": axes[0]["stage_id"],
            "parameter": axes[0]["parameter"],
        }
    elif fault == "missing-seed-target":
        sweep["seed_binding"] = {"stage_id": "missing", "parameter": "seed"}
    elif fault == "zero-budget":
        sweep["evaluation_budget"] = 0
    elif fault == "extensions-array":
        source["extensions"] = []
    elif fault == "stage-identifier":
        stages[0]["id"] = "Mixed"
    elif fault == "stages-object":
        body["stages"] = {}
    elif fault == "stage-count":
        body["stages"] = [deepcopy(stages[1]) for _ in range(65)]
    elif fault == "constant-seed-target":
        sweep["seed_binding"] = {"stage_id": "trace", "parameter": "compiler_trace"}
    elif fault == "bound-seed-target":
        sweep["seed_binding"] = {"stage_id": "trace", "parameter": "program_source"}
    elif fault == "insufficient-budget":
        sweep["evaluation_budget"] = 1
    elif fault == "edge-count":
        body["stages"] = [
            {
                **stages[1],
                "id": f"stage-{index}",
                "depends_on": [f"stage-{parent}" for parent in range(5)],
            }
            for index in range(32)
        ]
    else:
        source["extensions"] = {"source_text": "a" * MAX_WORKFLOW_BYTES}
    before = write_json(source)
    with pytest.raises(ValueError, match=reason):
        parse_workflow(source)
    assert write_json(source) == before


def test_explicit_seed_binding_preserves_original_compiler_optimisation_setting() -> None:
    """Roundtrip the browser's same original compiler seed-binding declaration."""
    source = workflow_payload()
    body = cast(dict[str, object], source["body"])
    sweep = cast(dict[str, object], body["sweep"])
    axes = cast(list[dict[str, object]], sweep["axes"])
    sweep.update(
        axes=[axes[0]],
        seeds=["0", "1", "2"],
        seed_binding={"stage_id": "trace", "parameter": "optimisation_level"},
    )
    before = write_json(source)
    admitted = parse_workflow(source)
    assert admitted.sweep.seed_binding == ("trace", "optimisation_level")
    assert write_json(admitted.to_dict()) == before


def test_original_numeric_token_ports_preserve_bits_shapes_and_integer_domains() -> None:
    """Use the shared lossless codec oracle for exact original port-value admission."""
    corpus = cast(
        dict[str, object],
        read_json((Path(__file__).parent / "data/studio_workspace/transport.json").read_text()),
    )
    cases = cast(list[dict[str, object]], corpus["cases"])
    original = next(case for case in cases if case["id"] == "typed_numeric_tokens")
    tokens = cast(dict[str, object], read_json(cast(str, original["input_json"])))
    floating = WorkflowPortType("experiment_revision.v1", "float64", (), "1")
    assert type(validate_port_value(floating, tokens["f"])) is float
    vector = cast(
        tuple[float, ...],
        validate_port_value(replace(floating, shape=(2,)), [tokens["f"], tokens["z"]]),
    )
    assert vector[0] == tokens["f"] and math.copysign(1, vector[1]) == -1
    for malformed_shape in (tokens["f"], [tokens["f"]]):
        with pytest.raises(ValueError, match="original port shape differs"):
            validate_port_value(replace(floating, shape=(2,)), malformed_shape)
    with pytest.raises(ValueError, match="original port dtype differs"):
        validate_port_value(floating, tokens["i"])
    for nonfinite in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_port_value(floating, nonfinite)
    boolean = parse_workflow(workflow_payload()).stages[0].parameters["compiler_trace"]
    assert validate_port_value(replace(floating, dtype="bool"), boolean) is True
    assert validate_port_value(replace(floating, dtype="int64"), tokens["i"]) == 1
    assert (
        validate_port_value(replace(floating, dtype="uint64"), tokens["big"]) == 9007199254740993
    )
    for dtype, values in (
        ("bool", [tokens["i"]]),
        ("int64", [boolean, -(2**63) - 1, 2**63]),
        ("uint64", [tokens["f"], -1, 2**64]),
    ):
        for mismatched in values:
            with pytest.raises(ValueError, match="original port dtype differs"):
                validate_port_value(replace(floating, dtype=dtype), mismatched)


def test_original_compiler_json_port_retains_immutable_emitted_evidence() -> None:
    """Roundtrip actual source-owned compiler output without an execution claim."""
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    checkpoint = cast(dict[str, object], corpus["original_checkpoint"])
    body = cast(dict[str, object], checkpoint["body"])
    entry = cast(list[dict[str, object]], body["entries"])[0]
    output = cast(dict[str, object], entry["output"])
    admitted = cast(
        Mapping[str, object],
        validate_port_value(
            WorkflowPortType("studio.workflow-output.v1", "json", (), "1"), output
        ),
    )
    result = cast(Mapping[str, object], admitted["result"])
    outputs = cast(Mapping[str, object], result["outputs"])
    assert outputs["execution_status"] == "emitted_not_executed"
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], admitted)["digest"] = "changed"
