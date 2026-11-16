# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental circuit plan set contract
"""Build circuit plan sets from packaged compression and check them here and in the worker."""

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
    CIRCUIT_PLAN_SET_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    CircuitPlanSet,
    CompressedLatentBatch,
    LatentBatch,
    ModelDescriptor,
    ReservoirSpec,
    SplitManifest,
    TaskSpec,
    build_circuit_plan_set,
    build_static_circuit_plan,
    canonical_bytes,
    decode_circuit_plan_set,
    decode_contract,
    validate_circuit_plan_set,
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
        "dev_groups": ("dev-00",),
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


def _rebuilt(
    plan_set: CircuitPlanSet, *, origin: str | None = None, **changes: Any
) -> CircuitPlanSet:
    """Return the plan set with changed fields and a header that hashes them."""
    names = ("reservoir_digest", "compressed_digest", "row_count", "arms", "plans")
    edited = {name: changes.get(name, getattr(plan_set, name)) for name in names}
    content = {
        "schema": CIRCUIT_PLAN_SET_SCHEMA,
        "object_kind": "circuit_plan_set",
        **edited,
        "arms": list(edited["arms"]),
        "plans": [plan.to_wire() for plan in edited["plans"]],
    }
    parents = (edited["reservoir_digest"], edited["compressed_digest"])
    header = _header("circuit_plan_set", content, parents, origin or plan_set.header.data_origin)
    return CircuitPlanSet(**edited, header=header)


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


def _request(chain: _Chain, reservoir: object, plan_set: object) -> dict[str, object]:
    """Return one worker plan-set request carrying the whole compression chain."""
    fitted = chain.fitted
    return {
        "op": "roundtrip_circuit_plan_set",
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
        "reservoir": reservoir,
        "plan_set": plan_set,
    }


def test_plan_sets_rebuild_every_row_and_arm_and_roundtrip(tmp_path: Path) -> None:
    """Both arms and a single arm validate here and in the worker, plan for plan."""
    chain = _chain()
    spec = _reservoir()
    for arms in (_ARMS, ("xy",)):
        plan_set = build_circuit_plan_set(
            spec, chain.compressed, chain.angles, arms=arms, revisions=_REVISIONS
        )
        validate_circuit_plan_set(plan_set, spec, chain.compressed, chain.angles)
        assert plan_set.row_count == 10 and len(plan_set.plans) == 10 * len(arms)
        expected = tuple(
            build_static_circuit_plan(
                spec, chain.compressed, chain.angles, row_index=row, arm_id=arm
            )
            for row in range(10)
            for arm in arms
        )
        assert plan_set.plans == expected
        assert plan_set.header.data_origin == chain.compressed.header.data_origin
        assert decode_circuit_plan_set(canonical_bytes(plan_set.to_wire())) == plan_set
        response = _worker(_request(chain, spec.to_wire(), plan_set.to_wire()), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["plan_set_sha256"] == _sha(plan_set.to_wire())
    xy = plan_set.plans[0].operations
    assert [gate for gate, _, _ in xy[:4]] == ["ry"] * 4 and ("rxx", (0, 1)) in [
        (g, w) for g, w, _ in xy
    ]


def test_plan_set_refuses_missing_repeated_foreign_or_unordered_plans() -> None:
    """The inventory is exactly every row in every arm, in row then arm order."""
    chain = _chain()
    spec = _reservoir()
    valid = build_circuit_plan_set(
        spec, chain.compressed, chain.angles, arms=_ARMS, revisions=_REVISIONS
    )
    plans = valid.plans
    foreign = replace(plans[0], reservoir_digest="9" * 64)
    stray: Any = (*plans[:-1], "plan")
    for changes, reason in (
        (
            {"arms": ("xy", "encoding_plus_fields_no_coupling")},
            "^arms must be a sorted unique tuple$",
        ),
        ({"arms": ()}, "^arms must be a sorted unique tuple$"),
        ({"arms": ("xy", "xy")}, "^arms must be a sorted unique tuple$"),
        ({"arms": ["xy"]}, "^arms must be a sorted unique tuple$"),
        ({"arms": ("encoding_plus_fields_no_coupling", "zz")}, "^unknown static arm$"),
        ({"plans": list(plans)}, "^plans must be a tuple of StaticCircuitPlan records$"),
        ({"plans": stray}, "^plans must be a tuple of StaticCircuitPlan records$"),
        ({"plans": plans[:-1]}, "^plans must cover every row in every arm once, in row then arm"),
        ({"plans": (plans[1], plans[0], *plans[2:])}, "^plans must cover every row in every arm"),
        (
            {"plans": (foreign, *plans[1:])},
            "^every plan names this reservoir and compressed batch$",
        ),
        ({"row_count": 0}, "plan row count must be a bounded positive integer"),
        (
            {"compressed_digest": "E" * 64},
            "^compressed_digest must be a lowercase SHA-256 digest$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    with pytest.raises(ValueError, match="^CircuitPlanSet requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="raw_result_set"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, plans=plans[:2], row_count=1)


def test_plan_set_validation_refuses_another_reservoir_batch_origin_or_recipe() -> None:
    """Validation rebuilds every plan from the exact reservoir and angle bytes."""
    chain = _chain()
    spec = _reservoir()
    valid = build_circuit_plan_set(
        spec, chain.compressed, chain.angles, arms=_ARMS, revisions=_REVISIONS
    )
    other_spec = _reservoir(("0x0.0p+0",) * 4)
    first = valid.plans[0]
    drifted = replace(
        first,
        operations=(
            (first.operations[0][0], first.operations[0][1], "0x1.0000000000000p-2"),
            *first.operations[1:],
        ),
    )
    for plan_set, used_spec, reason in (
        (valid, other_spec, "^the plan set names a different reservoir$"),
        (
            _rebuilt(
                valid,
                compressed_digest="7" * 64,
                plans=tuple(replace(p, compressed_digest="7" * 64) for p in valid.plans),
            ),
            spec,
            "^the plan set names a different compressed batch$",
        ),
        (
            _rebuilt(valid, row_count=1, plans=valid.plans[:2]),
            spec,
            "^the plan set does not cover every compressed row$",
        ),
        (
            _rebuilt(valid, origin="owner_dataset"),
            spec,
            "^plan-set origin differs from the compressed batch$",
        ),
        (
            _rebuilt(valid, plans=(drifted, *valid.plans[1:])),
            spec,
            "^static plan differs from its frozen source and recipe$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_circuit_plan_set(plan_set, used_spec, chain.compressed, chain.angles)
    not_a_set: Any = "plans"
    with pytest.raises(ValueError, match="^plan-set validation needs a CircuitPlanSet$"):
        validate_circuit_plan_set(not_a_set, spec, chain.compressed, chain.angles)


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


def test_plan_set_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires and reservoirs are refused by the decoder and the worker."""
    chain = _chain()
    spec = _reservoir()
    valid = build_circuit_plan_set(
        spec, chain.compressed, chain.angles, arms=_ARMS, revisions=_REVISIONS
    )
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(valid.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^CircuitPlanSet fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.circuit_plan_set.v2",
            "^unknown CircuitPlanSet schema$",
        ),
        (("arms",), "xy", "^arms and plans must be lists on wire$"),
        (("plans",), {"first": 1}, "^arms and plans must be lists on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_circuit_plan_set(canonical_bytes(_forge(valid.to_wire(), path, value)))
    plan_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "CircuitPlanSet fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.circuit_plan_set.v2",
            "unknown CircuitPlanSet schema",
        ),
        (("row_count",), 9, "does not cover every compressed row"),
        (
            ("arms",),
            ["xy", "encoding_plus_fields_no_coupling"],
            "arms must be a sorted unique list",
        ),
        (("arms",), ["encoding_plus_fields_no_coupling", "zz"], "unknown static arm"),
        (("plans",), [], "plans must cover every row in every arm once"),
        (("reservoir_digest",), "7" * 64, "names a different reservoir or compressed batch"),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, forgery, refusal in plan_cases:
        response = _worker(
            _request(chain, spec.to_wire(), _forge(valid.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    first = valid.plans[0]
    drifted = replace(
        first,
        operations=(
            (first.operations[0][0], first.operations[0][1], "0x1.0000000000000p-2"),
            *first.operations[1:],
        ),
    )
    edited = _rebuilt(valid, plans=(drifted, *valid.plans[1:]))
    response = _worker(_request(chain, spec.to_wire(), edited.to_wire()), tmp_path)
    assert response.returncode == 2
    assert (
        "a static plan differs from its frozen source and recipe"
        in json.loads(response.stdout)["reason"]
    )
    spec_cases: tuple[tuple[str, object, str], ...] = (
        ("unexpected", True, "reservoir spec fields mismatch"),
        (
            "schema",
            "scpn.experimental.llm_qpu.reservoir_spec.v2",
            "unsupported reservoir spec schema",
        ),
        ("kernel_id", "xy_sequence_digital_v1", "supported n4/n8 kernel"),
        ("edges", [[0, 2], [1, 2], [2, 3]], "explicit open chain"),
        ("couplings_hex", ["0x1.0000000000000p-1"], "one coupling is required per ordered edge"),
        ("fields_hex", ["0x0.0p+0"], "one local field is required per qubit"),
        ("tau_hex", "0x0.0p+0", "tau must be bounded canonical hex float"),
        ("tau_hex", "0x1p-1", "tau must be bounded canonical hex float"),
        (
            "couplings_hex",
            ["0x1.4p+3"] * 3,
            "reservoir parameter must be bounded canonical hex float",
        ),
        ("repetitions", 0, "repetitions must be a bounded positive integer"),
        ("gate_order", "fields_then_encode", "unsupported static preparation or gate convention"),
    )
    for name, forgery, refusal in spec_cases:
        response = _worker(
            _request(chain, _forge(spec.to_wire(), (name,), forgery), valid.to_wire()), tmp_path
        )
        assert response.returncode == 2, name
        assert refusal in json.loads(response.stdout)["reason"], name
    other = _reservoir(("0x0.0p+0",) * 4)
    response = _worker(_request(chain, other.to_wire(), valid.to_wire()), tmp_path)
    assert response.returncode == 2
    assert (
        "names a different reservoir or compressed batch" in json.loads(response.stdout)["reason"]
    )
    missing = _request(chain, spec.to_wire(), valid.to_wire())
    del missing["reservoir"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "plan-set request fields mismatch"
