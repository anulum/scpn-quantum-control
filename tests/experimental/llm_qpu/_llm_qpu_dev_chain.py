# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — shared experimental LLM-QPU dev-chain test fixtures
"""One dev chain (task, split, model, latents, fitted compressor, angles), its plans, a synthetic target and the worker runner."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    PLANNED_EXPERIMENT_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    CircuitPlanSet,
    CompressedLatentBatch,
    ExperimentProtocol,
    LatentBatch,
    MeasurementSpec,
    ModelDescriptor,
    PlannedExperiment,
    ReservoirSpec,
    SplitManifest,
    TargetSnapshot,
    TaskSpec,
    build_circuit_plan_set,
    canonical_bytes,
    f64,
    plan_measurements,
)
from scpn_quantum_control.experimental.llm_qpu.contracts.target import ReportedValue
from scpn_quantum_control.experimental.llm_qpu.contracts.wire import TARGET_SNAPSHOT_SCHEMA
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
    """Return the n4 open-chain reservoir of the engineering preview."""
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
    """Return the engineering-preview readout: three bases, two sentinels in slots A and B."""
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


_LINE = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 8))
_BASIS = ("cz", "measure", "rz", "sx", "x")


def _both_ways(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Return the sorted edges in both directions."""
    return tuple(sorted({*edges, *((b, a) for a, b in edges)}))


def _target(**changes: Any) -> TargetSnapshot:
    """Return a synthetic nine-qubit target whose header hashes exactly its fields."""
    fields: dict[str, Any] = {
        "provider": "ibm",
        "account_route": "fixture",
        "device_name": "synthetic_line_9",
        "sdk_name": "qiskit",
        "sdk_version": "2.6.2",
        "sdk_target_digest": "b" * 64,
        "capability_provenance": "synthetic_fixture",
        "num_qubits": 9,
        "basis_gates": _BASIS,
        "coupling_edges": _both_ways(_LINE),
        "calibration_id": ReportedValue("unknown", None, "a synthetic target has no calibration"),
        "calibration_time_utc": ReportedValue(
            "unknown", None, "a synthetic target has no calibration"
        ),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": ReportedValue("known", 300, None),
        **changes,
    }
    scientific = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **{
            name: [list(edge) for edge in value]
            if name == "coupling_edges"
            else list(value)
            if name == "basis_gates"
            else value.to_wire()
            if type(value) is ReportedValue
            else value
            for name, value in fields.items()
        },
    }
    header = ArtifactHeader(
        "target_snapshot",
        hashlib.sha256(canonical_bytes(scientific)).hexdigest(),
        (fields["sdk_target_digest"],),
        _BASE,
        _REVISION,
        "offline_design",
        "synthetic_classical",
        "design_only",
    )
    return TargetSnapshot(**fields, header=header)


def _dev_design(
    dev_groups: tuple[str, ...],
    arms: tuple[str, str] = ("classical", "quantum"),
    metric: str = "mean_log_loss",
) -> tuple[SplitManifest, ExperimentProtocol]:
    """Return a split whose dev groups are given and an exploratory log-loss protocol of two arms."""
    split_fields: dict[str, Any] = {
        "task_digest": "a" * 64,
        "dataset_digest": "b" * 64,
        "train_groups": ("train",),
        "dev_groups": tuple(sorted(set(dev_groups))),
        "test_groups": ("test-a",),
        "seed": 7,
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
            "split_manifest", split_content, ("a" * 64, "b" * 64), "synthetic_classical"
        ),
    )
    fields: dict[str, Any] = {
        "experiment_id": "fixture-protocol",
        "evaluation_id": "dev-exploration",
        "task_digest": "a" * 64,
        "split_digest": _sha(split.to_wire()),
        "model_digest": "c" * 64,
        "compressor_digest": "d" * 64,
        "kernel_plan_digest": "e" * 64,
        "measurement_plan_digest": "f" * 64,
        "analysis_plan_digest": "0" * 64,
        "mode": "contextual_latent_transform",
        "prediction_mode": "teacher_forced_trace",
        "arms": tuple(sorted(arms)),
        "classical_arm": arms[0],
        "quantum_arm": arms[1],
        "primary_metric": metric,
        "contrast_direction": "classical_minus_quantum"
        if metric in ("mean_log_loss", "mse", "mae")
        else "quantum_minus_classical",
        "minimum_relevant_delta": 0.02,
        "max_dev_fits_per_arm": 80,
        "max_qpu_evaluations": 1,
        "stopping_rule": "fixed_split_no_optional_stopping",
        "failure_policy": "retain_all_ids_report_missing_and_sensitivity",
        "inference_unit": "source_group_paired",
        "analysis_scope": "exploratory",
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.experiment_protocol.v2",
        "object_kind": "experiment_protocol",
        **{
            k: list(v) if k == "arms" else f64(v) if k == "minimum_relevant_delta" else v
            for k, v in fields.items()
        },
    }
    names: tuple[str, ...] = (
        "task_digest",
        "split_digest",
        "model_digest",
        "compressor_digest",
        "kernel_plan_digest",
        "measurement_plan_digest",
        "analysis_plan_digest",
    )
    protocol = ExperimentProtocol(
        **fields,
        header=_header(
            "experiment_protocol", content, tuple(fields[n] for n in names), "synthetic_classical"
        ),
    )
    return split, protocol
