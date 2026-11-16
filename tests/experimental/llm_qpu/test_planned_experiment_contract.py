# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental planned experiment contract
"""Plan data cells and readout sentinels and check them here and in the worker."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _llm_qpu_dev_chain import (
    _ARMS,
    _REVISIONS,
    _dev_chain,
    _forge,
    _planned,
    _rebuilt,
    _request,
    _reservoir,
    _sha,
    _spec,
    _worker,
)

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    MeasurementSpec,
    PlannedCell,
    SentinelPreparation,
    build_circuit_plan_set,
    canonical_bytes,
    decode_contract,
    decode_planned_experiment,
    plan_measurements,
    validate_planned_experiment,
)


def test_experiments_plan_every_cell_and_sentinel_and_roundtrip(tmp_path: Path) -> None:
    """Data cells per plan and basis plus A/B sentinels validate here and in the worker."""
    for planned, data, sentinels in (
        (_planned(), 2 * 2 * 3, 2 * 2),
        (
            _planned(
                ("xy",), _spec(bases=("Z",), sentinel_states=("zero",), sentinel_slots=("A",))
            ),
            2,
            1,
        ),
    ):
        experiment = planned.experiment
        validate_planned_experiment(experiment, planned.plan_set, _reservoir(), planned.spec)
        roles = [cell.role for cell in experiment.cells]
        assert roles == ["data"] * data + ["sentinel"] * sentinels
        assert len(experiment.measurements) == len(experiment.cells)
        assert (
            experiment.total_shots()
            == data * planned.spec.data_shots + sentinels * planned.spec.sentinel_shots
        )
        assert decode_planned_experiment(canonical_bytes(experiment.to_wire())) == experiment
        response = _worker(
            _request(planned, planned.spec.to_wire(), experiment.to_wire()), tmp_path
        )
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["experiment_sha256"] == _sha(experiment.to_wire())
    full = _planned().experiment
    assert full.total_shots() == 12 * 256 + 4 * 1024
    first = full.cells[0]
    assert (first.key.arm_id, first.key.basis_id, first.key.sequence_id) == (
        "encoding_plus_fields_no_coupling",
        "X",
        "contextual_single",
    )
    sentinel = full.cells[-1]
    assert (sentinel.key.source_sample_id, sentinel.key.replicate_id, sentinel.role) == (
        "sentinel_zero",
        "slot_B",
        "sentinel",
    )
    operations: Any = SentinelPreparation("one", 4).to_wire()["operations"]
    assert [op["gate"] for op in operations] == ["ry"] * 4
    assert SentinelPreparation("zero", 8).to_wire()["operations"] == []


def test_sentinels_and_specs_refuse_unknown_states_bases_and_partial_maps() -> None:
    """Sentinels are the basis extremes; a spec names known bases and full bit maps."""
    for args, reason in (
        (("half", 4), "^a sentinel prepares the zero or the one state$"),
        (("zero", 5), "^a sentinel needs an n4 or n8 register$"),
    ):
        with pytest.raises(ValueError, match=reason):
            SentinelPreparation(*args)
    for changes, reason in (
        ({"bases": ("Z", "X")}, "^basis must be a nonempty sorted unique tuple$"),
        ({"bases": ()}, "^basis must be a nonempty sorted unique tuple$"),
        ({"bases": ["X"]}, "^basis must be a nonempty sorted unique tuple$"),
        ({"bases": (1,)}, "^basis must be a nonempty sorted unique tuple$"),
        ({"bases": ("W",)}, "^unknown basis: W$"),
        ({"sentinel_states": ("half",)}, "^unknown sentinel state: half$"),
        ({"sentinel_slots": ("A", "A")}, "^sentinel slot must be a nonempty sorted unique tuple$"),
        ({"sentinel_slots": ("",)}, "sentinel slot must be bounded nonempty text"),
        ({"data_shots": 0}, "data_shots must be a bounded positive integer"),
        ({"sentinel_shots": 1_000_001}, "sentinel_shots must be a bounded positive integer"),
        ({"logical_to_clbit": (0, 1, 2)}, "^bit maps must cover an n4 or n8 register$"),
        ({"logical_to_clbit": [0, 1, 2, 3]}, "^bit maps must cover an n4 or n8 register$"),
        (
            {"logical_to_clbit": (0, 1, 2, 2)},
            "^logical-to-classical map must be a complete permutation$",
        ),
        (
            {"display_order_clbits": (3, 2, 1)},
            "^display bit order must be a complete permutation$",
        ),
        (
            {"display_order_clbits": (3.0, 2, 1, 0)},
            "^display bit order must be a complete permutation$",
        ),
        ({"experiment_id": ""}, "experiment_id must be bounded nonempty text"),
    ):
        with pytest.raises(ValueError, match=reason):
            _spec(**changes)
    chain = _dev_chain()
    plan_set = build_circuit_plan_set(
        _reservoir(), chain.compressed, chain.angles, arms=_ARMS, revisions=_REVISIONS
    )
    wide = _spec(logical_to_clbit=tuple(range(8)), display_order_clbits=tuple(range(8)))
    with pytest.raises(
        ValueError, match="^the measurement spec must read the reservoir's register$"
    ):
        plan_measurements(plan_set, _reservoir(), wide, revisions=_REVISIONS)
    not_a_spec: Any = "spec"
    with pytest.raises(
        ValueError, match="^the measurement spec must read the reservoir's register$"
    ):
        plan_measurements(plan_set, _reservoir(), not_a_spec, revisions=_REVISIONS)
    assert MeasurementSpec.from_wire(_spec().to_wire()) == _spec()
    for name, value, reason in (
        ("unexpected", True, "^MeasurementSpec fields mismatch$"),
        ("schema", "x", "^unknown MeasurementSpec schema$"),
    ):
        wire = _spec().to_wire()
        wire[name] = value
        with pytest.raises(ValueError, match=reason):
            MeasurementSpec.from_wire(wire)


def test_experiment_refuses_duplicate_keys_unbound_cells_and_bad_records() -> None:
    """Every cell has a unique key and is bound to its own measurement request."""
    valid = _planned().experiment
    cells, measurements = valid.cells, valid.measurements
    stray: Any = (*cells[:-1], "cell")
    for changes, reason in (
        (
            {"sentinels": list(valid.sentinels)},
            "^sentinels must be a tuple of SentinelPreparation records$",
        ),
        ({"cells": ()}, "^cells must be a nonempty tuple of PlannedCell records$"),
        ({"cells": stray}, "^cells must be a nonempty tuple of PlannedCell records$"),
        (
            {"measurements": list(measurements)},
            "^measurements must be a tuple of MeasurementPlan records$",
        ),
        ({"measurements": measurements[:-1]}, "^every cell needs exactly one measurement plan$"),
        ({"cells": (cells[0], *cells[:-1])}, "^cell keys must be unique$"),
        (
            {"measurements": (measurements[1], measurements[0], *measurements[2:])},
            "^a cell is not bound to its own measurement request$",
        ),
        ({"plan_set_digest": "E" * 64}, "^plan_set_digest must be a lowercase SHA-256 digest$"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    first = cells[0]
    bound = {"measurement_request_digest": measurements[0].request_digest}
    for circuit, shots in (("9" * 64, first.shots), (first.circuit_plan_digest, first.shots + 1)):
        moved = PlannedCell.create(
            key=first.key,
            circuit_plan_digest=circuit,
            shots=shots,
            measurement_width=4,
            role="data",
            metadata=bound,
        )
        with pytest.raises(
            ValueError, match="^a cell is not bound to its own measurement request$"
        ):
            replace(valid, cells=(moved, *cells[1:]))
    with pytest.raises(ValueError, match="^PlannedExperiment requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="circuit_plan_set"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, sentinels=valid.sentinels[:1])


def test_experiment_validation_refuses_another_plan_set_spec_origin_or_drift() -> None:
    """Validation replans every cell from the exact plan set, reservoir and spec."""
    planned = _planned()
    experiment, plan_set, spec = planned.experiment, planned.plan_set, planned.spec
    other_set = build_circuit_plan_set(
        _reservoir(),
        planned.chain.compressed,
        planned.chain.angles,
        arms=("xy",),
        revisions=_REVISIONS,
    )
    other_spec = _spec(data_shots=512)
    drifted = plan_measurements(plan_set, _reservoir(), other_spec, revisions=_REVISIONS)
    for used_set, used_spec, candidate, reason in (
        (other_set, spec, experiment, "^the experiment names a different plan set$"),
        (plan_set, other_spec, experiment, "^the experiment names a different measurement spec$"),
        (
            plan_set,
            spec,
            _rebuilt(experiment, origin="owner_dataset"),
            "^experiment origin differs from the plan set$",
        ),
        (
            plan_set,
            spec,
            _rebuilt(drifted, measurement_spec_digest=_sha(spec.to_wire())),
            "^a cell, measurement or sentinel differs from its exact plan$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_planned_experiment(candidate, used_set, _reservoir(), used_spec)
    not_an_experiment: Any = "experiment"
    with pytest.raises(ValueError, match="^experiment validation needs a PlannedExperiment$"):
        validate_planned_experiment(not_an_experiment, plan_set, _reservoir(), spec)


def test_experiment_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires and specs are refused by the decoder and the worker."""
    planned = _planned()
    experiment = planned.experiment
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(experiment.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^PlannedExperiment fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.planned_experiment.v2",
            "^unknown PlannedExperiment schema$",
        ),
        (("cells",), {"first": 1}, "^sentinels, cells and measurements must be lists on wire$"),
        (("sentinels", 0), {"state_id": "zero"}, "^SentinelPreparation fields mismatch$"),
        (
            ("sentinels", 0, "purpose"),
            "mitigation",
            "^a sentinel preparation differs from its canonical form$",
        ),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_planned_experiment(canonical_bytes(_forge(experiment.to_wire(), path, value)))
    spec_wire = planned.spec.to_wire()
    experiment_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "PlannedExperiment fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.planned_experiment.v2",
            "unknown PlannedExperiment schema",
        ),
        (("plan_set_digest",), "7" * 64, "names a different plan set or measurement spec"),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, forgery, refusal in experiment_cases:
        response = _worker(
            _request(planned, spec_wire, _forge(experiment.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    drifted = plan_measurements(
        planned.plan_set, _reservoir(), _spec(data_shots=512), revisions=_REVISIONS
    )
    edited = _rebuilt(drifted, measurement_spec_digest=_sha(spec_wire))
    response = _worker(_request(planned, spec_wire, edited.to_wire()), tmp_path)
    assert response.returncode == 2
    assert "differs from its exact plan" in json.loads(response.stdout)["reason"]
    spec_cases: tuple[tuple[str, object, str], ...] = (
        ("unexpected", True, "MeasurementSpec fields mismatch"),
        (
            "schema",
            "scpn.experimental.llm_qpu.measurement_spec.v2",
            "unknown MeasurementSpec schema",
        ),
        ("experiment_id", "", "experiment_id must be bounded normalized text"),
        ("bases", ["Z", "X"], "bases must be a sorted unique list of known values"),
        ("bases", ["W"], "bases must be a sorted unique list of known values"),
        ("bases", [], "bases must be a nonempty sorted unique list"),
        (
            "sentinel_states",
            ["half"],
            "sentinel_states must be a sorted unique list of known values",
        ),
        ("data_shots", 0, "data_shots must be a bounded positive integer"),
        ("logical_to_clbit", [0, 1, 2], "logical_to_clbit must be a complete permutation"),
    )
    for name, forgery, refusal in spec_cases:
        response = _worker(
            _request(
                planned, _forge(planned.spec.to_wire(), (name,), forgery), experiment.to_wire()
            ),
            tmp_path,
        )
        assert response.returncode == 2, name
        assert refusal in json.loads(response.stdout)["reason"], name
    missing = _request(planned, spec_wire, experiment.to_wire())
    del missing["measurement_spec"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "experiment request fields mismatch"
