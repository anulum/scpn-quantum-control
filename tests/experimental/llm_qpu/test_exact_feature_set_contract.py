# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental exact feature set contract
"""Exact ideal-digital features from Qiskit, checked by two independent statevectors."""

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
    EXACT_FEATURE_SET_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    CircuitPlanSet,
    CompressedLatentBatch,
    ExactFeatureSet,
    LatentBatch,
    ModelDescriptor,
    ReservoirSpec,
    SplitManifest,
    TaskSpec,
    build_circuit_plan_set,
    canonical_bytes,
    decode_contract,
    decode_exact_feature_set,
    exact_expectations,
    validate_exact_feature_set,
)
from scpn_quantum_control.experimental.llm_qpu.data.compression_fit import (
    FittedCompressor,
    compress_latents,
    fit_train_compressor,
)
from scpn_quantum_control.experimental.llm_qpu.data.digital_simulation import simulate_exact

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


def _plan_set(arms: tuple[str, ...] = _ARMS) -> tuple[_Chain, CircuitPlanSet]:
    """Return the dev chain and its plan set."""
    chain = _dev_chain()
    return chain, build_circuit_plan_set(
        _reservoir(), chain.compressed, chain.angles, arms=arms, revisions=_REVISIONS
    )


def _rebuilt(
    features: ExactFeatureSet, *, origin: str | None = None, **changes: Any
) -> ExactFeatureSet:
    """Return the feature set with changed fields and a header that hashes them."""
    names = ("plan_set_digest", "simulator", "simulator_version", "rows", "tolerance_hex")
    edited = {name: changes.get(name, getattr(features, name)) for name in names}
    content = {
        "schema": EXACT_FEATURE_SET_SCHEMA,
        "object_kind": "exact_feature_set",
        **edited,
        "rows": [row.to_wire() for row in edited["rows"]],
    }
    header = _header(
        "exact_feature_set",
        content,
        (edited["plan_set_digest"],),
        origin or features.header.data_origin,
    )
    return ExactFeatureSet(**edited, header=header)


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


def _request(chain: _Chain, plan_set: CircuitPlanSet, features: object) -> dict[str, object]:
    """Return one worker exact-feature request carrying the whole chain."""
    fitted = chain.fitted
    return {
        "op": "roundtrip_exact_feature_set",
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
        "plan_set": plan_set.to_wire(),
        "features": features,
    }


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


def test_exact_features_agree_across_three_statevectors_and_physics(tmp_path: Path) -> None:
    """Qiskit, the contract's statevector and the worker agree, and no-coupling Z is cos(theta)."""
    chain, plan_set = _plan_set()
    features = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    validate_exact_feature_set(features, plan_set, 4)
    assert len(features.rows) == 4 and features.simulator == "qiskit_statevector"
    assert decode_exact_feature_set(canonical_bytes(features.to_wire())) == features
    angles = np.frombuffer(chain.angles, dtype="<f4").astype(np.float64).reshape(2, 4)
    for row in features.rows:
        values = dict(row.expectations)
        if row.arm_id == "encoding_plus_fields_no_coupling":
            cosines = np.cos(angles[row.row_index])
            assert np.allclose(values["Z"][:4], cosines, atol=1e-12)
            assert np.allclose(values["Z"][4:], cosines[:-1] * cosines[1:], atol=1e-12)
    for plan, row in zip(plan_set.plans, features.rows, strict=True):
        assert dict(row.expectations)["X"] == pytest.approx(
            exact_expectations(plan, 4, "X"), abs=1e-12
        )
    response = _worker(_request(chain, plan_set, features.to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["features_sha256"] == _sha(features.to_wire())
    single = simulate_exact(_plan_set(("xy",))[1], 4, bases=("Z",), revisions=_REVISIONS)
    assert [basis for basis, _ in single.rows[0].expectations] == ["Z"]


def test_exact_rows_and_sets_refuse_malformed_records() -> None:
    """Rows carry finite [-1, 1] values per sorted basis; sets name a bounded tolerance."""
    _, plan_set = _plan_set()
    valid = simulate_exact(plan_set, 4, bases=("X", "Z"), revisions=_REVISIONS)
    row = valid.rows[0]
    values = dict(row.expectations)["X"]
    for changes, reason in (
        ({"row_index": -1}, "^row index must be a bounded non-negative integer$"),
        ({"sample_id": ""}, "sample_id must be bounded nonempty text"),
        ({"plan_digest": "E" * 64}, "^plan digest must be a lowercase SHA-256 digest$"),
        ({"expectations": ()}, "^expectations must be \\(basis, values\\) pairs$"),
        ({"expectations": (("X",),)}, "^expectations must be \\(basis, values\\) pairs$"),
        (
            {"expectations": (("Z", values), ("X", values))},
            "^expectation bases must be sorted, unique",
        ),
        (
            {"expectations": (("W", values),)},
            "^expectation bases must be sorted, unique and X, Y or Z$",
        ),
        ({"expectations": (("X", values[:6]),)}, "^an n4 or n8 basis has 7 or 15 observables$"),
        (
            {"expectations": (("X", (2.0, *values[1:])),)},
            "^an ideal Pauli expectation is a finite float",
        ),
        (
            {"expectations": (("X", (1, *values[1:])),)},
            "^an ideal Pauli expectation is a finite float",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(row, **changes)
    stray: Any = (valid.rows[0], "row")
    set_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"simulator": "aer_statevector"}, "^unknown ideal-digital simulator$"),
        ({"simulator_version": ""}, "simulator version must be bounded nonempty text"),
        ({"rows": ()}, "^rows must be a nonempty tuple of ExactRow records$"),
        ({"rows": stray}, "^rows must be a nonempty tuple of ExactRow records$"),
        ({"rows": (valid.rows[0], valid.rows[0])}, "^rows must name each plan once$"),
        (
            {"tolerance_hex": (0.0).hex()},
            "^tolerance must be a canonical hex float in \\(0, 1e-9\\]$",
        ),
        ({"tolerance_hex": (1e-8).hex()}, "^tolerance must be a canonical hex float in"),
        ({"tolerance_hex": "0x1p-40"}, "^tolerance must be a canonical hex float in"),
        ({"tolerance_hex": "tiny"}, "^tolerance must be a canonical hex float in"),
        ({"plan_set_digest": "E" * 64}, "^plan set digest must be a lowercase SHA-256 digest$"),
    )
    for changes, reason in set_cases:
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    assert _rebuilt(valid, tolerance_hex=(1e-9).hex()).tolerance_hex == (1e-9).hex()
    with pytest.raises(ValueError, match="^ExactFeatureSet requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="planned_experiment"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, rows=valid.rows[:1])


def test_exact_validation_refuses_another_plan_set_origin_rows_or_values() -> None:
    """Each row follows its plan and matches the independent statevector within tolerance."""
    _, plan_set = _plan_set()
    valid = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    other = _plan_set(("xy",))[1]
    rows = valid.rows
    moved = replace(rows[0], sample_id="dev-sample-9")
    x = dict(rows[0].expectations)
    nudged = replace(
        rows[0],
        expectations=(("X", (x["X"][0] + 1e-6, *x["X"][1:])), ("Y", x["Y"]), ("Z", x["Z"])),
    )
    validate_exact_feature_set(valid, plan_set, 4)
    for features, size, reason in (
        (
            _rebuilt(valid, plan_set_digest=_sha(other.to_wire())),
            4,
            "^the features name a different plan set$",
        ),
        (_rebuilt(valid, origin="owner_dataset"), 4, "^feature origin differs from the plan set$"),
        (_rebuilt(valid, rows=rows[:3]), 4, "^every plan needs exactly one exact row$"),
        (_rebuilt(valid, rows=(moved, *rows[1:])), 4, "^an exact row does not follow its plan$"),
        (valid, 8, "^an exact row has the wrong observable count$"),
        (
            _rebuilt(valid, rows=(nudged, *rows[1:])),
            4,
            "^an exact expectation differs from the independent",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_exact_feature_set(features, plan_set, size)
    not_a_set: Any = "features"
    with pytest.raises(ValueError, match="^exact validation needs an ExactFeatureSet$"):
        validate_exact_feature_set(not_a_set, plan_set, 4)


def test_exact_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    chain, plan_set = _plan_set()
    features = simulate_exact(plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(features.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^ExactFeatureSet fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.exact_feature_set.v2",
            "^unknown ExactFeatureSet schema$",
        ),
        (("rows",), {"first": 1}, "^rows must be a list on wire$"),
        (("rows", 0, "unexpected"), True, "^ExactRow fields mismatch$"),
        (
            ("rows", 0, "expectations", 0),
            "X",
            "^expectations must be \\[basis, values\\] pairs on wire$",
        ),
        (("rows", 0, "expectations", 0, 1, 0), "0.5", "^expectations must be f64 tags on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_exact_feature_set(canonical_bytes(_forge(features.to_wire(), path, value)))
    worker_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "ExactFeatureSet fields mismatch"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.exact_feature_set.v2",
            "unknown ExactFeatureSet schema",
        ),
        (("simulator",), "aer_statevector", "unknown ideal-digital simulator"),
        (("simulator_version",), "", "simulator version must be bounded normalized text"),
        (("tolerance_hex",), (1e-8).hex(), "tolerance must be bounded canonical hex float"),
        (("plan_set_digest",), "7" * 64, "the features name a different plan set"),
        (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
    )
    for path, forgery, refusal in worker_cases:
        response = _worker(
            _request(chain, plan_set, _forge(features.to_wire(), path, forgery)), tmp_path
        )
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    rows = features.rows
    x = dict(rows[0].expectations)
    nudged = replace(
        rows[0],
        expectations=(("X", (x["X"][0] + 1e-6, *x["X"][1:])), ("Y", x["Y"]), ("Z", x["Z"])),
    )
    consistent: tuple[tuple[ExactFeatureSet, str], ...] = (
        (_rebuilt(features, rows=rows[:3]), "every plan needs exactly one exact row"),
        (
            _rebuilt(features, rows=(replace(rows[0], sample_id="dev-sample-9"), *rows[1:])),
            "does not follow its plan",
        ),
        (_rebuilt(features, rows=(nudged, *rows[1:])), "differs from the independent statevector"),
    )
    for edited, refusal in consistent:
        response = _worker(_request(chain, plan_set, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    missing = _request(chain, plan_set, features.to_wire())
    del missing["features"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "exact request fields mismatch"
