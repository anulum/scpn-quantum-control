# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental planned experiment contract
"""Plan data cells and readout sentinels and check them here and in the worker."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    PLANNED_EXPERIMENT_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    CircuitPlanSet,
    CompressedLatentBatch,
    LatentBatch,
    MeasurementSpec,
    ModelDescriptor,
    PlannedCell,
    PlannedExperiment,
    ReservoirSpec,
    SentinelPreparation,
    SplitManifest,
    TaskSpec,
    build_circuit_plan_set,
    canonical_bytes,
    decode_contract,
    decode_planned_experiment,
    plan_measurements,
    validate_planned_experiment,
)
from scpn_quantum_control.experimental.llm_qpu.data.compression_fit import (
    FittedCompressor,
    compress_latents,
    fit_train_compressor,
)

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_REVISIONS = (_BASE, _REVISION)
_ARMS = ("encoding_plus_fields_no_coupling", "xy")
_TRAIN = tuple(f"train-{index:02d}" for index in range(10))


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header bound to the content and parents."""
    return ArtifactHeader(
        kind,
        _sha(content),
        tuple(sorted(parents)),
        _BASE,
        _REVISION,
        "offline_design",
        origin,
        "design_only",
    )


def _reservoir(
    fields: tuple[str, ...] = (
        "0x1.999999999999ap-4",
        "-0x1.999999999999ap-4",
        "0x1.999999999999ap-3",
        "-0x1.999999999999ap-3",
    ),
) -> ReservoirSpec:
    """Return the n4 open-chain reservoir of the W07 preview."""
    return ReservoirSpec(
        kernel_id="xy_static_digital_v1",
        n_qubits=4,
        edges=((0, 1), (1, 2), (2, 3)),
        couplings_hex=("0x1.0000000000000p-1",) * 3,
        fields_hex=fields,
        tau_hex="0x1.0000000000000p-1",
        repetitions=1,
        initial_state="zero",
        angle_unit="radian",
        gate_order="encode_then_fields_even_odd_v1",
    )


@dataclass(frozen=True)
class _Chain:
    """Packaged compression of one train batch, with every record and payload."""

    task: TaskSpec
    split: SplitManifest
    model: ModelDescriptor
    latent: LatentBatch
    latent_payload: bytes
    fitted: FittedCompressor
    compressed: CompressedLatentBatch
    angles: bytes


def _chain() -> _Chain:
    """Fit and compress a ten-row synthetic train batch to four angles per row."""
    task_fields: dict[str, Any] = {
        "task_id": "synthetic-plan-task",
        "objective": "Predict a label from a prefix",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": "accuracy",
        "group_definition": "One generated scene per source group",
    }
    task_content = {
        "schema": "scpn.experimental.llm_qpu.task_spec.v2",
        "object_kind": "task_spec",
        **task_fields,
    }
    task = TaskSpec(
        **task_fields,
        header=_header("task_spec", task_content, ("a" * 64,), "synthetic_classical"),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": _TRAIN,
        "dev_groups": ("dev-00", "dev-01"),
        "test_groups": ("test-00",),
        "seed": 19,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{k: list(v) if k.endswith("_groups") else v for k, v in split_fields.items()},
    }
    split = SplitManifest(
        **split_fields,
        header=_header(
            "split_manifest",
            split_content,
            (split_fields["task_digest"], "b" * 64),
            "synthetic_classical",
        ),
    )
    model_fields: dict[str, Any] = {
        "model_id": "synthetic-hidden-source",
        "checkpoint_digest": "c" * 64,
        "tokenizer_digest": "d" * 64,
        "chat_template_digest": "e" * 64,
        "runtime_build_digest": "f" * 64,
        "loader_id": "fixture-loader",
        "quantization": "none",
        "tensor_dtype": "float32",
        "block_count": 4,
        "hidden_width": 8,
        "tap_block_index": 2,
        "tap_stream": "residual_hidden_state",
        "tap_boundary": "before_norm",
        "probe_evidence_digest": "1" * 64,
    }
    model_content = {
        "schema": "scpn.experimental.llm_qpu.model_descriptor.v1",
        "object_kind": "model_descriptor",
        **model_fields,
    }
    names = (
        "checkpoint_digest",
        "tokenizer_digest",
        "chat_template_digest",
        "runtime_build_digest",
        "probe_evidence_digest",
    )
    model = ModelDescriptor(
        **model_fields,
        header=_header(
            "model_descriptor",
            model_content,
            tuple(model_fields[n] for n in names),
            "owner_checkpoint",
        ),
    )
    values = np.random.default_rng(7).normal(size=(10, 8)).astype(np.float32)
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    tensor = ArrayDescriptor("<f4", (10, 8), len(payload), hashlib.sha256(payload).hexdigest())
    latent_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "model_digest": _sha(model.to_wire()),
        "split_name": "train",
        "layout": "contextual",
        "sample_ids": tuple(f"sample-{i:02d}" for i in range(10)),
        "source_ids": tuple(f"source-{i:02d}" for i in range(10)),
        "group_ids": _TRAIN,
        "lengths": (1,) * 10,
        "mask": ((True,),) * 10,
        "token_positions": ((0,),) * 10,
        "answer_start_positions": ((1,),) * 10,
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            k: v.to_wire()
            if k == "tensor"
            else [list(r) for r in v]
            if k in nested
            else list(v)
            if type(v) is tuple
            else v
            for k, v in latent_fields.items()
        },
    }
    parents = (
        latent_fields["task_digest"],
        latent_fields["split_digest"],
        latent_fields["model_digest"],
        tensor.sha256,
    )
    latent = LatentBatch(
        **latent_fields, header=_header("latent_batch", content, parents, "synthetic_classical")
    )
    fitted = fit_train_compressor(
        task, split, model, latent, payload, output_width=4, fit_seed=3, revisions=_REVISIONS
    )
    compressed, angles = compress_latents(
        task, split, model, latent, payload, fitted, revisions=_REVISIONS
    )
    return _Chain(task, split, model, latent, payload, fitted, compressed, angles)


def _dev_chain() -> _Chain:
    """Compress a two-row dev batch with the train-fitted compressor."""
    train = _chain()
    values = np.random.default_rng(13).normal(size=(2, 8)).astype(np.float32)
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    tensor = ArrayDescriptor("<f4", (2, 8), len(payload), hashlib.sha256(payload).hexdigest())
    base = train.latent
    fields: dict[str, Any] = {
        "task_digest": base.task_digest,
        "split_digest": base.split_digest,
        "model_digest": base.model_digest,
        "split_name": "dev",
        "layout": "contextual",
        "sample_ids": ("dev-sample-0", "dev-sample-1"),
        "source_ids": ("dev-source-0", "dev-source-1"),
        "group_ids": ("dev-00", "dev-01"),
        "lengths": (1, 1),
        "mask": ((True,), (True,)),
        "token_positions": ((0,), (0,)),
        "answer_start_positions": ((1,), (1,)),
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            k: v.to_wire()
            if k == "tensor"
            else [list(r) for r in v]
            if k in nested
            else list(v)
            if type(v) is tuple
            else v
            for k, v in fields.items()
        },
    }
    parents = (base.task_digest, base.split_digest, base.model_digest, tensor.sha256)
    latent = LatentBatch(
        **fields, header=_header("latent_batch", content, parents, "synthetic_classical")
    )
    compressed, angles = compress_latents(
        train.task, train.split, train.model, latent, payload, train.fitted, revisions=_REVISIONS
    )
    return _Chain(
        train.task, train.split, train.model, latent, payload, train.fitted, compressed, angles
    )


def _spec(**changes: Any) -> MeasurementSpec:
    """Return the W07 preview readout: three bases, two sentinels in slots A and B."""
    fields: dict[str, Any] = {
        "experiment_id": "llm_qpu_n4_engineering_v1",
        "bases": ("X", "Y", "Z"),
        "data_shots": 256,
        "logical_to_clbit": (0, 1, 2, 3),
        "display_order_clbits": (3, 2, 1, 0),
        "register_name": "readout",
        "sentinel_states": ("one", "zero"),
        "sentinel_slots": ("A", "B"),
        "sentinel_shots": 1024,
        "replicate_id": "r0",
        "planned_epoch_id": "unbound",
        **changes,
    }
    return MeasurementSpec(**fields)


@dataclass(frozen=True)
class _Planned:
    """One dev chain, its plan set and its planned experiment."""

    chain: _Chain
    plan_set: CircuitPlanSet
    spec: MeasurementSpec
    experiment: PlannedExperiment


def _planned(arms: tuple[str, ...] = _ARMS, spec: MeasurementSpec | None = None) -> _Planned:
    """Plan the dev chain's cells under the spec."""
    chain = _dev_chain()
    plan_set = build_circuit_plan_set(
        _reservoir(), chain.compressed, chain.angles, arms=arms, revisions=_REVISIONS
    )
    used = spec or _spec()
    experiment = plan_measurements(plan_set, _reservoir(), used, revisions=_REVISIONS)
    return _Planned(chain, plan_set, used, experiment)


def _rebuilt(
    experiment: PlannedExperiment, *, origin: str | None = None, **changes: Any
) -> PlannedExperiment:
    """Return the experiment with changed fields and a header that hashes them."""
    names = ("plan_set_digest", "measurement_spec_digest", "sentinels", "cells", "measurements")
    edited = {name: changes.get(name, getattr(experiment, name)) for name in names}
    content = {
        "schema": PLANNED_EXPERIMENT_SCHEMA,
        "object_kind": "planned_experiment",
        "plan_set_digest": edited["plan_set_digest"],
        "measurement_spec_digest": edited["measurement_spec_digest"],
        "sentinels": [s.to_wire() for s in edited["sentinels"]],
        "cells": [c.to_wire() for c in edited["cells"]],
        "measurements": [m.to_wire() for m in edited["measurements"]],
    }
    parents = (edited["plan_set_digest"], edited["measurement_spec_digest"])
    header = _header(
        "planned_experiment", content, parents, origin or experiment.header.data_origin
    )
    return PlannedExperiment(**edited, header=header)


def _b64(payload: bytes) -> str:
    """Return a base64 string of the payload."""
    return base64.b64encode(payload).decode("ascii")


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=30,
        check=False,
    )


def _request(planned: _Planned, spec: object, experiment: object) -> dict[str, object]:
    """Return one worker experiment request carrying the whole chain."""
    chain, fitted = planned.chain, planned.chain.fitted
    return {
        "op": "roundtrip_planned_experiment",
        "task": chain.task.to_wire(),
        "split": chain.split.to_wire(),
        "model": chain.model.to_wire(),
        "latent": chain.latent.to_wire(),
        "latent_base64": _b64(chain.latent_payload),
        "compressor": fitted.artifact.to_wire(),
        "centering_base64": _b64(fitted.centering_payload),
        "projection_base64": _b64(fitted.projection_payload),
        "scales_base64": _b64(fitted.scales_payload),
        "compressed": chain.compressed.to_wire(),
        "angles_base64": _b64(chain.angles),
        "reservoir": _reservoir().to_wire(),
        "plan_set": planned.plan_set.to_wire(),
        "measurement_spec": spec,
        "experiment": experiment,
    }


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


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
