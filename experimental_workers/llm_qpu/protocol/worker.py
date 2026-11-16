# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — isolated experimental worker boundary
"""Report the worker boundary and refuse all unimplemented operations."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import random
import re
import struct
import sys
import unicodedata
from typing import Any

_MAX_REQUEST_BYTES = 65_536
_SCHEMA = "scpn.experimental.llm_qpu.worker_boundary.v1"
_CELL_SCHEMA = "scpn.experimental.llm_qpu.planned_cell.v1"
_ARRAY_SCHEMA = "scpn.experimental.llm_qpu.array_descriptor.v1"
_TASK_SCHEMA = "scpn.experimental.llm_qpu.task_spec.v2"
_SPLIT_SCHEMA = "scpn.experimental.llm_qpu.split_manifest.v2"
_HEADER_SCHEMA = "scpn.experimental.llm_qpu.artifact_header.v1"
_MODEL_SCHEMA = "scpn.experimental.llm_qpu.model_descriptor.v1"
_LATENT_SCHEMA = "scpn.experimental.llm_qpu.latent_batch.v1"
_COMPRESSOR_SCHEMA = "scpn.experimental.llm_qpu.compressor_artifact.v1"
_COMPRESSED_SCHEMA = "scpn.experimental.llm_qpu.compressed_latent_batch.v1"
_EXPERIMENT_PROTOCOL_SCHEMA = "scpn.experimental.llm_qpu.experiment_protocol.v2"
_TARGET_SNAPSHOT_SCHEMA = "scpn.experimental.llm_qpu.target_snapshot.v1"
_COMPILED_BUNDLE_SCHEMA = "scpn.experimental.llm_qpu.compiled_bundle.v1"
_COST_QUOTE_SCHEMA = "scpn.experimental.llm_qpu.cost_quote.v1"
_OWNER_APPROVAL_SCHEMA = "scpn.experimental.llm_qpu.owner_approval.v1"
_RECEIPT_SCHEMA = "scpn.experimental.llm_qpu.provider_batch_receipt.v1"
_RESULT_SET_SCHEMA = "scpn.experimental.llm_qpu.raw_result_set.v1"
_FEATURE_SCHEMA = "scpn.experimental.llm_qpu.feature_batch.v1"
_REPORT_SCHEMA = "scpn.experimental.llm_qpu.evaluation_report.v1"
_STUDENT_SCHEMA = "scpn.experimental.llm_qpu.student_artifact.v1"
_TEACHER_ORIGINS = {
    "hardware_features": "provider_result",
    "simulated_features": "synthetic_classical",
}
_COST_ITEMS = ["provider_resource_units", "price_minor_units", "classical_compute_seconds"]
_MEASUREMENT_SCHEMA = "scpn.experimental.llm_qpu.measurement_plan.v1"
_RESULT_ORIGINS = {
    "provider_metadata": "provider_result",
    "synthetic_classical": "synthetic_classical",
}
_UUID_TEXT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_PROVIDER_JOB_ID = {"iqm": _UUID_TEXT, "ibm": re.compile(r"[a-z0-9]{20}\Z")}
_APPROVAL_ORIGINS = {"live": "owner_decision", "test": "synthetic_classical"}
_APPROVAL_EGRESS = ("private_derived", "reviewed_synthetic", "value_free")
_QUOTE_COUNTS = ("job_count", "overhead_executions", "resource_units", "price_minor_units")
_BUNDLE_CELL_FIELDS = (
    "planned_cell_digest",
    "logical_plan_digest",
    "compiled_payload_digest",
    "binding_digest",
)
_TARGET_ORIGINS = {
    "provider_target_query": "provider_metadata",
    "synthetic_fixture": "synthetic_classical",
}
_TARGET_CAPABILITIES = (
    "calibration_id",
    "calibration_time_utc",
    "max_shots",
    "max_circuits_per_job",
)
_UTC_SECOND = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_KEY_FIELDS = {
    "experiment_id",
    "source_sample_id",
    "sequence_id",
    "arm_id",
    "basis_id",
    "replicate_id",
    "planned_epoch_id",
    "parameter_binding_id",
}
_CELL_FIELDS = {
    "schema",
    "object_kind",
    "key",
    "circuit_plan_digest",
    "shots",
    "measurement_width",
    "role",
    "metadata",
}
_RESERVED = {"shots", "provider", "device", "job_id", "origin", "approval", "status"}
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DTYPE_SIZE = {"<f8": 8, "<f4": 4, "<i8": 8, "<i4": 4, "<u8": 8, "<u4": 4}


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _canonical(value: object, depth: int = 0) -> object:
    if depth > 16:
        raise ValueError("wire nesting exceeds limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError("wire integer exceeds range")
        return value
    if type(value) is str:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized or len(normalized) > 256 or any(ord(ch) < 32 for ch in normalized):
            raise ValueError("invalid wire text")
        return normalized
    if type(value) is list:
        if len(value) > 4096:
            raise ValueError("wire list exceeds limit")
        return [_canonical(item, depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > 4096:
            raise ValueError("wire object exceeds limit")
        if set(value) == {"$f64"}:
            token = value["$f64"]
            if type(token) is not str:
                raise ValueError("f64 tag must be text")
            try:
                parsed = float.fromhex(token)
            except ValueError as exc:
                raise ValueError("invalid f64 hex") from exc
            if not math.isfinite(parsed) or (0.0 if parsed == 0.0 else parsed).hex() != token:
                raise ValueError("noncanonical f64")
            return value
        for key in value:
            if type(key) is not str or _FIELD_NAME.fullmatch(key) is None:
                raise ValueError("invalid wire field name")
        return {key: _canonical(item, depth + 1) for key, item in value.items()}
    raise ValueError("unsupported wire value")


def _canonical_bytes(value: object) -> bytes:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 1_048_576:
        raise ValueError("wire exceeds byte limit")
    return encoded


def _validate_planned_cell(cell: object) -> dict[str, object]:
    if type(cell) is not dict or set(cell) != _CELL_FIELDS:
        raise ValueError("planned cell fields mismatch")
    if cell["schema"] != _CELL_SCHEMA or cell["object_kind"] != "planned_cell":
        raise ValueError("unknown planned cell schema")
    key = cell["key"]
    if type(key) is not dict or set(key) != _KEY_FIELDS:
        raise ValueError("CellKey fields mismatch")
    if any(type(value) is not str or not value for value in key.values()):
        raise ValueError("invalid CellKey text")
    digest = cell["circuit_plan_digest"]
    if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid circuit plan digest")
    shots = cell["shots"]
    width = cell["measurement_width"]
    if type(shots) is not int or not 0 < shots <= 1_000_000:
        raise ValueError("invalid shots")
    if type(width) is not int or not 0 < width <= 64:
        raise ValueError("invalid measurement width")
    if cell["role"] not in ("data", "sentinel", "calibration"):
        raise ValueError("unknown cell role")
    metadata = cell["metadata"]
    if type(metadata) is not dict or _RESERVED.intersection(metadata):
        raise ValueError("metadata shadows reserved fields")
    if len(_canonical_bytes(metadata)) > 8192:
        raise ValueError("metadata bytes exceed limit")
    return cell


def _roundtrip_contract(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "cell", "array", "array_base64"}:
        raise ValueError("contract request fields mismatch")
    cell_wire = _canonical_bytes(_validate_planned_cell(request["cell"]))
    array = request["array"]
    raw = request["array_base64"]
    if type(array) is not dict or set(array) != {
        "schema",
        "object_kind",
        "dtype",
        "shape",
        "byte_length",
        "sha256",
    }:
        raise ValueError("array descriptor fields mismatch")
    if array["schema"] != _ARRAY_SCHEMA or array["object_kind"] != "array_descriptor":
        raise ValueError("unknown array schema")
    dtype = array["dtype"]
    shape = array["shape"]
    if type(dtype) is not str or dtype not in _DTYPE_SIZE:
        raise ValueError("unsupported array dtype")
    if type(shape) is not list or not 1 <= len(shape) <= 4:
        raise ValueError("invalid array rank")
    elements = 1
    for axis in shape:
        if type(axis) is not int or not 0 < axis <= 1_000_000:
            raise ValueError("invalid array axis")
        elements *= axis
        if elements > 1_000_000:
            raise ValueError("array element limit exceeded")
    byte_length = array["byte_length"]
    if type(byte_length) is not int or byte_length != elements * _DTYPE_SIZE[dtype]:
        raise ValueError("array byte length mismatch")
    if type(raw) is not str or len(raw) > 60_000:
        raise ValueError("invalid array base64")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid array base64") from exc
    if len(payload) != byte_length or hashlib.sha256(payload).hexdigest() != array["sha256"]:
        raise ValueError("array payload mismatch")
    if dtype in ("<f8", "<f4"):
        code = "<d" if dtype == "<f8" else "<f"
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack(code, payload)):
            raise ValueError("nonfinite array value")
    array_wire = _canonical_bytes(array)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "cell": json.loads(cell_wire),
        "cell_sha256": hashlib.sha256(cell_wire).hexdigest(),
        "array": json.loads(array_wire),
        "array_sha256": hashlib.sha256(array_wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _validate_artifact_header(
    record: dict[str, object], parents: tuple[str, ...], data_origin: str
) -> None:
    header = record["header"]
    fields = {
        "schema",
        "lane_id",
        "object_kind",
        "content_digest",
        "parents",
        "base_repo_commit",
        "implementation_revision",
        "execution_origin",
        "data_origin",
        "claim_scope",
    }
    if type(header) is not dict or set(header) != fields:
        raise ValueError("ArtifactHeader fields mismatch")
    if header["schema"] != _HEADER_SCHEMA or header["lane_id"] != "llm-qpu":
        raise ValueError("unknown ArtifactHeader schema or lane")
    if header["object_kind"] != record["object_kind"]:
        raise ValueError("artifact object kind mismatch")
    if type(header["parents"]) is not list or header["parents"] != sorted(parents):
        raise ValueError("artifact parent lineage mismatch")
    if len(header["parents"]) != len(set(header["parents"])):
        raise ValueError("duplicate artifact parent")
    for name in ("base_repo_commit", "implementation_revision"):
        value = header[name]
        if type(value) is not str or _COMMIT.fullmatch(value) is None:
            raise ValueError("invalid artifact code revision")
    if header["base_repo_commit"] == header["implementation_revision"]:
        raise ValueError("artifact revision equals base")
    if header["execution_origin"] != "offline_design" or header["claim_scope"] != "design_only":
        raise ValueError("artifact origin or claim exceeds W02")
    if header["data_origin"] != data_origin:
        raise ValueError("artifact data origin mismatch")
    content = {name: value for name, value in record.items() if name != "header"}
    if header["content_digest"] != hashlib.sha256(_canonical_bytes(content)).hexdigest():
        raise ValueError("artifact content digest mismatch")


def _roundtrip_task_split(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "task", "split"}:
        raise ValueError("task/split request fields mismatch")
    task = request["task"]
    split = request["split"]
    task_fields = {
        "schema",
        "object_kind",
        "task_id",
        "objective",
        "source_kind",
        "target_origin",
        "label_schema_digest",
        "causal_cutoff",
        "primary_metric",
        "group_definition",
        "header",
    }

    split_fields = {
        "schema",
        "object_kind",
        "task_digest",
        "dataset_digest",
        "train_groups",
        "dev_groups",
        "test_groups",
        "seed",
        "dedup_rule",
        "test_target_custodian",
        "transform_fit_split",
        "header",
    }
    if type(task) is not dict or set(task) != task_fields:
        raise ValueError("TaskSpec fields mismatch")
    if task["schema"] != _TASK_SCHEMA or task["object_kind"] != "task_spec":
        raise ValueError("unknown TaskSpec schema")
    if task["source_kind"] not in ("owner_dataset", "external_dataset", "synthetic_classical"):
        raise ValueError("unknown task source")
    if task["target_origin"] not in ("independent_ground_truth", "classical_generator"):
        raise ValueError("QPU-generated or unknown target")
    if task["primary_metric"] not in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mse",
        "mae",
        "mean_log_loss",
    ):
        raise ValueError("unknown primary metric")
    if type(task["causal_cutoff"]) is not int or not 0 <= task["causal_cutoff"] <= 1_000_000:
        raise ValueError("invalid causal cutoff")
    if (
        type(task["label_schema_digest"]) is not str
        or _DIGEST.fullmatch(task["label_schema_digest"]) is None
    ):
        raise ValueError("invalid label schema digest")
    _validate_artifact_header(task, (task["label_schema_digest"],), task["source_kind"])
    task_wire = _canonical_bytes(task)
    if type(split) is not dict or set(split) != split_fields:
        raise ValueError("SplitManifest fields mismatch")
    if split["schema"] != _SPLIT_SCHEMA or split["object_kind"] != "split_manifest":
        raise ValueError("unknown SplitManifest schema")
    if split["task_digest"] != hashlib.sha256(task_wire).hexdigest():
        raise ValueError("task digest does not bind TaskSpec")
    if (
        type(split["dataset_digest"]) is not str
        or _DIGEST.fullmatch(split["dataset_digest"]) is None
    ):
        raise ValueError("invalid dataset digest")
    groups = []
    for name in ("train_groups", "dev_groups", "test_groups"):
        values = split[name]
        if type(values) is not list or not values or len(values) > 4096:
            raise ValueError("invalid source groups")
        if any(type(item) is not str or not item for item in values):
            raise ValueError("invalid source group ID")
        if values != sorted(set(values)):
            raise ValueError("source groups must be sorted and unique")
        groups.extend(values)
    if len(groups) != len(set(groups)):
        raise ValueError("source groups overlap")
    if type(split["seed"]) is not int or not 0 <= split["seed"] < 2**63:
        raise ValueError("invalid split seed")
    if split["dedup_rule"] not in ("exact_source_digest", "normalized_source_digest"):
        raise ValueError("unknown source dedup rule")
    if split["test_target_custodian"] != "separate_locked_evaluator":
        raise ValueError("test target custody not locked")
    if split["transform_fit_split"] != "train_only":
        raise ValueError("test-driven transform fit forbidden")
    _validate_artifact_header(
        split, (split["task_digest"], split["dataset_digest"]), task["source_kind"]
    )
    split_wire = _canonical_bytes(split)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "task": json.loads(task_wire),
        "split": json.loads(split_wire),
        "task_sha256": hashlib.sha256(task_wire).hexdigest(),
        "split_sha256": hashlib.sha256(split_wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _roundtrip_model_descriptor(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "model"}:
        raise ValueError("model request fields mismatch")
    model = request["model"]
    fields = {
        "schema",
        "object_kind",
        "model_id",
        "checkpoint_digest",
        "tokenizer_digest",
        "chat_template_digest",
        "runtime_build_digest",
        "loader_id",
        "quantization",
        "tensor_dtype",
        "block_count",
        "hidden_width",
        "tap_block_index",
        "tap_stream",
        "tap_boundary",
        "probe_evidence_digest",
        "header",
    }
    if type(model) is not dict or set(model) != fields:
        raise ValueError("ModelDescriptor fields mismatch")
    if model["schema"] != _MODEL_SCHEMA or model["object_kind"] != "model_descriptor":
        raise ValueError("unknown ModelDescriptor schema")
    for name in (
        "checkpoint_digest",
        "tokenizer_digest",
        "chat_template_digest",
        "runtime_build_digest",
        "probe_evidence_digest",
    ):
        value = model[name]
        if type(value) is not str or _DIGEST.fullmatch(value) is None:
            raise ValueError("invalid model identity digest")
    if model["quantization"] not in ("none", "gguf_q8_0", "gguf_q6_k", "gguf_q4_k_m"):
        raise ValueError("unknown model quantization")
    if model["tensor_dtype"] not in ("float16", "float32", "bfloat16"):
        raise ValueError("unknown hidden-state dtype")
    blocks = model["block_count"]
    width = model["hidden_width"]
    index = model["tap_block_index"]
    if type(blocks) is not int or not 0 < blocks <= 1024:
        raise ValueError("invalid model block count")
    if type(width) is not int or not 0 < width <= 65_536:
        raise ValueError("invalid observed hidden width")
    if type(index) is not int or not 0 <= index < blocks:
        raise ValueError("invalid tap block index")
    if model["tap_stream"] != "residual_hidden_state":
        raise ValueError("completion or embedding fallback forbidden")
    if model["tap_boundary"] not in ("before_norm", "after_norm"):
        raise ValueError("tap boundary missing")
    parents = tuple(
        model[name]
        for name in (
            "checkpoint_digest",
            "tokenizer_digest",
            "chat_template_digest",
            "runtime_build_digest",
            "probe_evidence_digest",
        )
    )
    _validate_artifact_header(model, parents, "owner_checkpoint")
    wire = _canonical_bytes(model)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "model": json.loads(wire),
        "model_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _roundtrip_latent_batch(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "task", "split", "model", "latent", "tensor_base64"}:
        raise ValueError("latent request fields mismatch")
    # The two calls below refuse a task, split or model that is not a complete record.
    task: Any = request["task"]
    split: Any = request["split"]
    model: Any = request["model"]
    latent = request["latent"]
    _roundtrip_task_split({"op": "roundtrip_task_split", "task": task, "split": split})
    _roundtrip_model_descriptor({"op": "roundtrip_model_descriptor", "model": model})
    fields = {
        "schema",
        "object_kind",
        "task_digest",
        "split_digest",
        "model_digest",
        "split_name",
        "layout",
        "sample_ids",
        "source_ids",
        "group_ids",
        "lengths",
        "mask",
        "token_positions",
        "answer_start_positions",
        "tap_block_index",
        "tap_boundary",
        "tensor",
        "header",
    }
    if type(latent) is not dict or set(latent) != fields:
        raise ValueError("LatentBatch fields mismatch")
    if latent["schema"] != _LATENT_SCHEMA or latent["object_kind"] != "latent_batch":
        raise ValueError("unknown LatentBatch schema")
    for name, record in (("task", task), ("split", split), ("model", model)):
        if latent[f"{name}_digest"] != hashlib.sha256(_canonical_bytes(record)).hexdigest():
            raise ValueError(f"latent {name} digest mismatch")
    split_name = latent["split_name"]
    if split_name not in ("train", "dev", "test"):
        raise ValueError("unknown latent split")
    layout = latent["layout"]
    if layout not in ("contextual", "chunk_isolated"):
        raise ValueError("unknown latent layout")
    tensor = latent["tensor"]
    if type(tensor) is not dict or set(tensor) != {
        "schema",
        "object_kind",
        "dtype",
        "shape",
        "byte_length",
        "sha256",
    }:
        raise ValueError("latent tensor descriptor fields mismatch")
    if tensor["schema"] != _ARRAY_SCHEMA or tensor["object_kind"] != "array_descriptor":
        raise ValueError("unknown latent tensor schema")
    shape = tensor["shape"]
    if (
        tensor["dtype"] != "<f4"
        or type(shape) is not list
        or len(shape) != (2 if layout == "contextual" else 3)
        or any(type(axis) is not int or not 0 < axis <= 1_000_000 for axis in shape)
    ):
        raise ValueError("invalid latent tensor dtype or shape")
    elements = math.prod(shape)
    if elements > 1_000_000 or tensor["byte_length"] != elements * 4:
        raise ValueError("latent tensor byte length mismatch")
    if type(tensor["sha256"]) is not str or _DIGEST.fullmatch(tensor["sha256"]) is None:
        raise ValueError("invalid latent tensor digest")
    if model["tensor_dtype"] != "float32" or shape[-1] != model["hidden_width"]:
        raise ValueError("latent dtype or hidden width mismatch")
    if (
        latent["tap_block_index"] != model["tap_block_index"]
        or latent["tap_boundary"] != model["tap_boundary"]
    ):
        raise ValueError("latent tap differs from model descriptor")
    rows = shape[0]
    steps = 1 if layout == "contextual" else shape[1]
    for name in ("sample_ids", "source_ids", "group_ids", "lengths"):
        values = latent[name]
        if type(values) is not list or len(values) != rows:
            raise ValueError("latent row inventory mismatch")
    if any(
        type(item) is not str or not item
        for item in (*latent["sample_ids"], *latent["source_ids"], *latent["group_ids"])
    ):
        raise ValueError("invalid latent row ID")
    if len(set(latent["sample_ids"])) != rows:
        raise ValueError("duplicate latent sample ID")
    if any(group not in split[f"{split_name}_groups"] for group in latent["group_ids"]):
        raise ValueError("latent group outside frozen split")
    for name in ("mask", "token_positions", "answer_start_positions"):
        values = latent[name]
        if (
            type(values) is not list
            or len(values) != rows
            or any(type(row) is not list or len(row) != steps for row in values)
        ):
            raise ValueError("latent mask or position shape mismatch")
    for index, length in enumerate(latent["lengths"]):
        if type(length) is not int or not 0 < length <= steps:
            raise ValueError("latent length exceeds tensor steps")
        for step in range(steps):
            active = latent["mask"][index][step]
            selected = latent["token_positions"][index][step]
            answer = latent["answer_start_positions"][index][step]
            if type(active) is not bool or active != (step < length):
                raise ValueError("latent mask and length disagree")
            if not active:
                if selected is not None or answer is not None:
                    raise ValueError("latent padding carries token positions")
            elif (
                type(selected) is not int
                or type(answer) is not int
                or not 0 <= selected < answer <= 1_000_001
                or selected > task["causal_cutoff"]
            ):
                raise ValueError("latent token position reaches answer or causal cutoff")
    raw = request["tensor_base64"]
    if type(raw) is not str or len(raw) > 60_000:
        raise ValueError("invalid latent tensor base64")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid latent tensor base64") from exc
    if (
        len(payload) != tensor["byte_length"]
        or hashlib.sha256(payload).hexdigest() != tensor["sha256"]
    ):
        raise ValueError("latent tensor payload mismatch")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload)):
        raise ValueError("latent tensor contains nonfinite float")
    if layout == "chunk_isolated":
        row_bytes = shape[-1] * 4
        for index, length in enumerate(latent["lengths"]):
            for step in range(length, steps):
                offset = (index * steps + step) * row_bytes
                if payload[offset : offset + row_bytes] != bytes(row_bytes):
                    raise ValueError("latent padding must be zero")
    _validate_artifact_header(
        latent,
        (latent["task_digest"], latent["split_digest"], latent["model_digest"], tensor["sha256"]),
        task["source_kind"],
    )
    wire = _canonical_bytes(latent)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "latent": json.loads(wire),
        "latent_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _positive_hex(value: object, name: str, maximum: float) -> float:
    if type(value) is not str:
        raise ValueError(f"invalid {name} hex")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name} hex") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= maximum or parsed.hex() != value:
        raise ValueError(f"invalid {name} hex")
    return parsed


def _float_array(descriptor: object, raw: object, shape: list[int]) -> tuple[float, ...]:
    fields = {"schema", "object_kind", "dtype", "shape", "byte_length", "sha256"}
    if type(descriptor) is not dict or set(descriptor) != fields:
        raise ValueError("compression array descriptor fields mismatch")
    if (
        descriptor["schema"] != _ARRAY_SCHEMA
        or descriptor["object_kind"] != "array_descriptor"
        or descriptor["dtype"] != "<f4"
        or descriptor["shape"] != shape
        or type(descriptor["byte_length"]) is not int
        or descriptor["byte_length"] != math.prod(shape) * 4
        or math.prod(shape) > 1_000_000
    ):
        raise ValueError("compression array shape or dtype mismatch")
    if type(raw) is not str or len(raw) > 60_000:
        raise ValueError("invalid compression array base64")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid compression array base64") from exc
    if (
        len(payload) != descriptor["byte_length"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("compression array payload digest mismatch")
    values = tuple(item[0] for item in struct.iter_unpack("<f", payload))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("compression array contains nonfinite float")
    return values


def _roundtrip_compression(request: dict[str, object]) -> dict[str, object]:
    fields = {
        "op",
        "task",
        "split",
        "model",
        "latent",
        "latent_base64",
        "compressor",
        "centering_base64",
        "projection_base64",
        "scales_base64",
        "compressed",
        "angles_base64",
    }
    if set(request) != fields:
        raise ValueError("compression request fields mismatch")
    # The call below refuses a task, split, model, latent batch or latent payload
    # that is not a complete record of its kind.
    task: Any = request["task"]
    split: Any = request["split"]
    model: Any = request["model"]
    latent: Any = request["latent"]
    latent_base64: Any = request["latent_base64"]
    compressor = request["compressor"]
    compressed = request["compressed"]
    _roundtrip_latent_batch(
        {
            "op": "roundtrip_latent_batch",
            "task": task,
            "split": split,
            "model": model,
            "latent": latent,
            "tensor_base64": latent_base64,
        }
    )
    compressor_fields = {
        "schema",
        "object_kind",
        "model_digest",
        "split_digest",
        "train_group_digest",
        "software_digest",
        "fit_seed",
        "fit_method",
        "map_id",
        "scale_policy",
        "input_width",
        "output_width",
        "epsilon_hex",
        "centering",
        "projection",
        "scales",
        "header",
    }
    if type(compressor) is not dict or set(compressor) != compressor_fields:
        raise ValueError("CompressorArtifact fields mismatch")
    if (
        compressor["schema"] != _COMPRESSOR_SCHEMA
        or compressor["object_kind"] != "compressor_artifact"
    ):
        raise ValueError("unknown CompressorArtifact schema")
    if compressor["model_digest"] != hashlib.sha256(_canonical_bytes(model)).hexdigest():
        raise ValueError("compressor model digest mismatch")
    if compressor["split_digest"] != hashlib.sha256(_canonical_bytes(split)).hexdigest():
        raise ValueError("compressor split digest mismatch")
    groups = hashlib.sha256(_canonical_bytes(split["train_groups"])).hexdigest()
    if compressor["train_group_digest"] != groups or split["transform_fit_split"] != "train_only":
        raise ValueError("compressor fit groups differ from train split")
    if (
        type(compressor["software_digest"]) is not str
        or _DIGEST.fullmatch(compressor["software_digest"]) is None
    ):
        raise ValueError("invalid compressor software digest")
    if type(compressor["fit_seed"]) is not int or not 0 <= compressor["fit_seed"] < 2**63:
        raise ValueError("invalid compressor fit seed")
    if (
        compressor["fit_method"] != "pca_train_svd_v1"
        or compressor["map_id"] != "rms_normalize_center_project_tanh_v2"
        or compressor["scale_policy"] != "block_zero_variance"
    ):
        raise ValueError("unknown compressor fit or map")
    width = compressor["input_width"]
    output_width = compressor["output_width"]
    if (
        type(width) is not int
        or width != model["hidden_width"]
        or type(output_width) is not int
        or output_width not in (4, 8)
        or output_width > width
        or model["tensor_dtype"] != "float32"
    ):
        raise ValueError("compressor width differs from model")
    epsilon = _positive_hex(compressor["epsilon_hex"], "epsilon", 1.0)
    centers = _float_array(compressor["centering"], request["centering_base64"], [width])
    projection = _float_array(
        compressor["projection"], request["projection_base64"], [width, output_width]
    )
    scales = _float_array(compressor["scales"], request["scales_base64"], [output_width])
    if any(value <= 0 for value in scales):
        raise ValueError("compressor scales must be positive")
    _validate_artifact_header(
        compressor,
        (
            compressor["model_digest"],
            compressor["split_digest"],
            compressor["train_group_digest"],
            compressor["software_digest"],
            compressor["centering"]["sha256"],
            compressor["projection"]["sha256"],
            compressor["scales"]["sha256"],
        ),
        task["source_kind"],
    )
    compressed_fields = {
        "schema",
        "object_kind",
        "latent_digest",
        "compressor_digest",
        "split_name",
        "layout",
        "sample_ids",
        "source_ids",
        "group_ids",
        "lengths",
        "mask",
        "token_positions",
        "answer_start_positions",
        "angle_unit",
        "comparison_tolerance_hex",
        "tensor",
        "header",
    }
    if type(compressed) is not dict or set(compressed) != compressed_fields:
        raise ValueError("CompressedLatentBatch fields mismatch")
    if (
        compressed["schema"] != _COMPRESSED_SCHEMA
        or compressed["object_kind"] != "compressed_latent_batch"
    ):
        raise ValueError("unknown CompressedLatentBatch schema")
    if compressed["latent_digest"] != hashlib.sha256(_canonical_bytes(latent)).hexdigest():
        raise ValueError("compressed source latent digest mismatch")
    if compressed["compressor_digest"] != hashlib.sha256(_canonical_bytes(compressor)).hexdigest():
        raise ValueError("compressed compressor digest mismatch")
    for name in (
        "split_name",
        "layout",
        "sample_ids",
        "source_ids",
        "group_ids",
        "lengths",
        "mask",
        "token_positions",
        "answer_start_positions",
    ):
        if compressed[name] != latent[name]:
            raise ValueError(f"compressed {name} differs from latent")
    if compressed["angle_unit"] != "radian":
        raise ValueError("compressed angles must be radians")
    tolerance = _positive_hex(compressed["comparison_tolerance_hex"], "angle tolerance", 1e-5)
    source_shape = latent["tensor"]["shape"]
    output_shape = [*source_shape[:-1], output_width]
    angles = _float_array(compressed["tensor"], request["angles_base64"], output_shape)
    _validate_artifact_header(
        compressed,
        (
            compressed["latent_digest"],
            compressed["compressor_digest"],
            compressed["tensor"]["sha256"],
        ),
        task["source_kind"],
    )
    source_raw = base64.b64decode(latent_base64, validate=True)
    source_values = tuple(item[0] for item in struct.iter_unpack("<f", source_raw))
    steps = 1 if latent["layout"] == "contextual" else source_shape[1]
    for row_index, length in enumerate(latent["lengths"]):
        for step in range(steps):
            flat_row = row_index * steps + step
            source = source_values[flat_row * width : (flat_row + 1) * width]
            output = angles[flat_row * output_width : (flat_row + 1) * output_width]
            if step >= length:
                if any(value != 0 for value in output):
                    raise ValueError("compressed padding must be zero")
                continue
            rms = math.sqrt(math.fsum(value * value for value in source) / width + epsilon)
            normalized = tuple(value / rms - centers[index] for index, value in enumerate(source))
            for column, actual in enumerate(output):
                if not -math.pi / 2 < actual < math.pi / 2:
                    raise ValueError("compressed angle outside open radian interval")
                projected = math.fsum(
                    projection[index * output_width + column] * normalized[index]
                    for index in range(width)
                )
                expected = (math.pi / 2) * math.tanh(projected / scales[column])
                if abs(actual - expected) > tolerance:
                    raise ValueError("compressed angle differs from frozen map")
    compressor_wire = _canonical_bytes(compressor)
    compressed_wire = _canonical_bytes(compressed)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "compressor": json.loads(compressor_wire),
        "compressed": json.loads(compressed_wire),
        "compressor_sha256": hashlib.sha256(compressor_wire).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed_wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _roundtrip_experiment_protocol(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "protocol"}:
        raise ValueError("protocol request fields mismatch")
    record = request["protocol"]
    fields = {
        "schema",
        "object_kind",
        "experiment_id",
        "evaluation_id",
        "task_digest",
        "split_digest",
        "model_digest",
        "compressor_digest",
        "kernel_plan_digest",
        "measurement_plan_digest",
        "analysis_plan_digest",
        "mode",
        "prediction_mode",
        "arms",
        "classical_arm",
        "quantum_arm",
        "primary_metric",
        "contrast_direction",
        "minimum_relevant_delta",
        "max_dev_fits_per_arm",
        "max_qpu_evaluations",
        "stopping_rule",
        "failure_policy",
        "inference_unit",
        "analysis_scope",
        "header",
    }
    if type(record) is not dict or set(record) != fields:
        raise ValueError("ExperimentProtocol fields mismatch")
    if (
        record["schema"] != _EXPERIMENT_PROTOCOL_SCHEMA
        or record["object_kind"] != "experiment_protocol"
    ):
        raise ValueError("unknown ExperimentProtocol schema")
    digest_fields = (
        "task_digest",
        "split_digest",
        "model_digest",
        "compressor_digest",
        "kernel_plan_digest",
        "measurement_plan_digest",
        "analysis_plan_digest",
    )
    digests = tuple(record[name] for name in digest_fields)
    if any(type(value) is not str or _DIGEST.fullmatch(value) is None for value in digests):
        raise ValueError("invalid protocol parent digest")
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate protocol parent digest")
    for name in ("experiment_id", "evaluation_id"):
        if type(record[name]) is not str or not record[name]:
            raise ValueError("invalid protocol identity")
    if record["mode"] not in ("contextual_latent_transform", "chunk_isolated_sequence"):
        raise ValueError("unknown memory experiment mode")
    if record["prediction_mode"] not in ("teacher_forced_trace", "free_generation"):
        raise ValueError("unknown prediction mode")
    arms = record["arms"]
    if type(arms) is not list or not 2 <= len(arms) <= 32:
        raise ValueError("invalid protocol arms")
    if any(type(arm) is not str or not arm for arm in arms) or arms != sorted(set(arms)):
        raise ValueError("protocol arms must be sorted and unique")
    if (
        record["classical_arm"] not in arms
        or record["quantum_arm"] not in arms
        or record["classical_arm"] == record["quantum_arm"]
    ):
        raise ValueError("invalid primary contrast")
    if record["primary_metric"] not in (
        "mean_log_loss",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mse",
        "mae",
    ):
        raise ValueError("unknown primary metric")
    lower_is_better = record["primary_metric"] in ("mean_log_loss", "mse", "mae")
    expected_direction = (
        "classical_minus_quantum" if lower_is_better else "quantum_minus_classical"
    )
    if record["contrast_direction"] != expected_direction:
        raise ValueError("primary contrast direction differs from metric")
    delta = record["minimum_relevant_delta"]
    if type(delta) is not dict or set(delta) != {"$f64"}:
        raise ValueError("minimum relevant delta must be tagged f64")
    try:
        threshold = float.fromhex(delta["$f64"])
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum relevant delta must be tagged f64") from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum relevant delta must be positive and finite")
    if not lower_is_better and threshold > 1.0:
        raise ValueError("bounded classification metric delta cannot exceed one")
    for name in ("max_dev_fits_per_arm", "max_qpu_evaluations"):
        value = record[name]
        if type(value) is not int or not 0 < value <= 1_000_000:
            raise ValueError("invalid protocol budget")
    if record["stopping_rule"] != "fixed_split_no_optional_stopping":
        raise ValueError("outcome-dependent stopping is forbidden")
    if record["failure_policy"] != "retain_all_ids_report_missing_and_sensitivity":
        raise ValueError("invalid missingness policy")
    if record["inference_unit"] != "source_group_paired":
        raise ValueError("invalid inference unit")
    if record["analysis_scope"] not in ("exploratory", "confirmatory"):
        raise ValueError("unknown analysis scope")
    header = record["header"]
    if type(header) is not dict or header.get("data_origin") not in (
        "owner_dataset",
        "external_dataset",
        "synthetic_classical",
    ):
        raise ValueError("protocol data origin must be a TaskSpec source kind")
    _validate_artifact_header(record, tuple(sorted(digests)), header["data_origin"])
    wire = _canonical_bytes(record)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "protocol": json.loads(wire),
        "protocol_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _target_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be text")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value or not value or len(value) > 256 or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must be bounded normalized text")
    return value


def _validate_reported_capability(value: object, name: str) -> None:
    if type(value) is not dict or set(value) != {"state", "value", "blocker"}:
        raise ValueError("ReportedValue fields mismatch")
    counted = name in ("max_shots", "max_circuits_per_job")
    if value["state"] == "unknown":
        if value["value"] is not None:
            raise ValueError("an unknown capability cannot carry a value")
        _target_text(value["blocker"], "capability blocker")
        return
    if value["state"] != "known" or value["blocker"] is not None:
        raise ValueError("a known capability needs a value and no blocker")
    reported = value["value"]
    if counted:
        if type(reported) is not int or not 0 < reported <= 10**9:
            raise ValueError(f"{name} must be a bounded positive integer")
        return
    _target_text(reported, name)
    if name == "calibration_time_utc" and _UTC_SECOND.fullmatch(str(reported)) is None:
        raise ValueError("calibration time must be an ISO-8601 UTC second")


def _validate_target_topology(target: dict[str, object]) -> None:
    qubits = target["num_qubits"]
    if type(qubits) is not int or not 0 < qubits <= 10_000:
        raise ValueError("invalid target qubit count")
    gates = target["basis_gates"]
    if type(gates) is not list or not gates:
        raise ValueError("basis gates must be a nonempty list")
    names = [_target_text(gate, "basis gate") for gate in gates]
    if names != sorted(set(names)):
        raise ValueError("basis gates must be sorted and unique")
    edges = target["coupling_edges"]
    if type(edges) is not list or len(edges) > 100_000:
        raise ValueError("coupling edges must be a bounded list")
    pairs: list[tuple[int, int]] = []
    for edge in edges:
        if type(edge) is not list or len(edge) != 2:
            raise ValueError("a coupling edge must be a qubit pair")
        if any(type(qubit) is not int or not 0 <= qubit < qubits for qubit in edge):
            raise ValueError("coupling edge qubit is outside the target")
        if edge[0] == edge[1]:
            raise ValueError("a coupling edge must join two different qubits")
        pairs.append((edge[0], edge[1]))
    if pairs != sorted(set(pairs)):
        raise ValueError("coupling edges must be sorted and unique")


def _roundtrip_target_snapshot(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "target"}:
        raise ValueError("target request fields mismatch")
    target = request["target"]
    fields = {
        "schema",
        "object_kind",
        "provider",
        "account_route",
        "device_name",
        "sdk_name",
        "sdk_version",
        "sdk_target_digest",
        "capability_provenance",
        "num_qubits",
        "basis_gates",
        "coupling_edges",
        *_TARGET_CAPABILITIES,
        "header",
    }
    if type(target) is not dict or set(target) != fields:
        raise ValueError("TargetSnapshot fields mismatch")
    if target["schema"] != _TARGET_SNAPSHOT_SCHEMA or target["object_kind"] != "target_snapshot":
        raise ValueError("unknown TargetSnapshot schema")
    if target["provider"] not in ("ibm", "iqm"):
        raise ValueError("unsupported target provider")
    for name in ("account_route", "device_name", "sdk_name", "sdk_version"):
        _target_text(target[name], name)
    digest = target["sdk_target_digest"]
    if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid SDK target digest")
    provenance = target["capability_provenance"]
    if provenance not in _TARGET_ORIGINS:
        raise ValueError("catalog or unknown capability provenance")
    _validate_target_topology(target)
    for name in _TARGET_CAPABILITIES:
        _validate_reported_capability(target[name], name)
    _validate_artifact_header(target, (digest,), _TARGET_ORIGINS[str(provenance)])
    wire = _canonical_bytes(target)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "target": json.loads(wire),
        "target_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


def _worker_digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _sorted_unique_names(values: object, name: str, *, allow_empty: bool = False) -> list[str]:
    if type(values) is not list or (not values and not allow_empty):
        raise ValueError(f"{name} must be a nonempty list")
    names = [_target_text(value, name) for value in values]
    if names != sorted(set(names)):
        raise ValueError(f"{name} must be sorted and unique")
    return names


def _wire_pairs(values: object, name: str) -> list[list[object]]:
    if type(values) is not list or any(
        type(item) is not list or len(item) != 2 for item in values
    ):
        raise ValueError(f"{name} must be a list of pairs")
    return [list(item) for item in values]


def _validate_bundle_cells(bundle: dict[str, object]) -> list[int]:
    cells = bundle["cells"]
    if type(cells) is not list or not cells or len(cells) > 4096:
        raise ValueError("compiled cells must be a bounded nonempty list")
    shots: list[int] = []
    planned: list[str] = []
    for cell in cells:
        if type(cell) is not dict or set(cell) != {*_BUNDLE_CELL_FIELDS, "shots"}:
            raise ValueError("CompiledCell fields mismatch")
        for name in _BUNDLE_CELL_FIELDS:
            _worker_digest(cell[name], name)
        count = cell["shots"]
        if type(count) is not int or not 0 < count <= 10_000_000:
            raise ValueError("compiled cell shots must be bounded")
        planned.append(str(cell["planned_cell_digest"]))
        shots.append(count)
    if len(set(planned)) != len(planned):
        raise ValueError("a planned cell appears twice in the bundle")
    layout = bundle["layout"]
    if type(layout) is not list or not layout:
        raise ValueError("layout must be a nonempty list")
    if any(type(qubit) is not int or qubit < 0 for qubit in layout) or len(set(layout)) != len(
        layout
    ):
        raise ValueError("layout must map logical qubits to distinct physical qubits")
    return shots


def _validate_bundle_gates(bundle: dict[str, object]) -> set[str]:
    logical = _sorted_unique_names(bundle["logical_gates"], "logical gates")
    counts = _wire_pairs(bundle["compiled_gate_counts"], "compiled gate counts")
    if not counts:
        raise ValueError("compiled gate counts must be nonempty")
    names = _sorted_unique_names([pair[0] for pair in counts], "compiled gates")
    for _, count in counts:
        if type(count) is not int or not 0 < count <= 10**9:
            raise ValueError("compiled gate count must be bounded")
    substitutions = _wire_pairs(bundle["substitutions"], "substitutions")
    replaced = _sorted_unique_names(
        [pair[0] for pair in substitutions], "substituted gates", allow_empty=True
    )
    introduced: set[str] = set()
    for source, targets in substitutions:
        if source not in logical:
            raise ValueError("a substitution must replace a logical gate")
        introduced.update(_sorted_unique_names(targets, "substitution targets"))
    if set(names) - set(logical) - introduced:
        raise ValueError("unreported gate substitution")
    if set(logical) - set(names) - set(replaced):
        raise ValueError("logical gate vanished without substitution")
    options = _wire_pairs(bundle["provider_options"], "provider options")
    keys = _sorted_unique_names(
        [pair[0] for pair in options], "provider option names", allow_empty=True
    )
    if "shots" in keys or len(options) > 64:
        raise ValueError("provider options are bounded and never carry shots")
    for _, option in options:
        if type(option) not in (int, str, bool):
            raise ValueError("provider option values must be scalars")
        if type(option) is str:
            _target_text(option, "provider option value")
    return set(names)


def _bundle_payload_digest(bundle: dict[str, object]) -> str:
    payload = {
        name: bundle[name]
        for name in (
            "cells",
            "layout",
            "logical_gates",
            "compiled_gate_counts",
            "substitutions",
            "compiled_depth",
            "provider_options",
        )
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _cross_check_bundle(
    bundle: dict[str, object], target: dict[str, object], shots: list[int], gates: set[str]
) -> None:
    if bundle["target_snapshot_digest"] != hashlib.sha256(_canonical_bytes(target)).hexdigest():
        raise ValueError("bundle was compiled against a different target snapshot")
    if (bundle["provider"], bundle["device_name"]) != (target["provider"], target["device_name"]):
        raise ValueError("bundle provider or device differs from its target")
    layout = bundle["layout"]
    if type(layout) is not list or max(layout) >= int(str(target["num_qubits"])):
        raise ValueError("layout uses a qubit the target does not have")
    basis = target["basis_gates"]
    if type(basis) is not list or gates - set(basis):
        raise ValueError("compiled gates outside the target basis")
    header, target_header = bundle["header"], target["header"]
    if type(header) is not dict or type(target_header) is not dict:
        raise ValueError("artifact headers must be objects")
    if header["data_origin"] != target_header["data_origin"]:
        raise ValueError("bundle and target data origins differ")
    limit = target["max_shots"]
    if type(limit) is dict and limit["state"] == "known" and max(shots) > int(str(limit["value"])):
        raise ValueError("a compiled cell requests more shots than the target allows")


_BUNDLE_FIELDS = {
    "schema",
    "object_kind",
    "target_snapshot_digest",
    "provider",
    "device_name",
    "sdk_lock_digest",
    "cells",
    "layout",
    "logical_gates",
    "compiled_gate_counts",
    "substitutions",
    "compiled_depth",
    "compile_log_digest",
    "provider_options",
    "payload_digest",
    "header",
}


def _validate_bundle_fields(bundle: object) -> dict[str, object]:
    if type(bundle) is not dict or set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("CompiledBundle fields mismatch")
    if bundle["schema"] != _COMPILED_BUNDLE_SCHEMA or bundle["object_kind"] != "compiled_bundle":
        raise ValueError("unknown CompiledBundle schema")
    for name in (
        "target_snapshot_digest",
        "sdk_lock_digest",
        "compile_log_digest",
        "payload_digest",
    ):
        _worker_digest(bundle[name], name)
    for name in ("provider", "device_name"):
        _target_text(bundle[name], name)
    depth = bundle["compiled_depth"]
    if type(depth) is not int or not 0 < depth <= 10_000_000:
        raise ValueError("compiled depth must be bounded")
    header = bundle["header"]
    origin = header.get("data_origin") if type(header) is dict else None
    if origin not in ("provider_metadata", "synthetic_classical"):
        raise ValueError("a compiled bundle derives from provider metadata or a fixture")
    return bundle


def _roundtrip_compiled_bundle(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "target", "bundle"}:
        raise ValueError("bundle request fields mismatch")
    _roundtrip_target_snapshot({"op": "roundtrip_target_snapshot", "target": request["target"]})
    target = request["target"]
    bundle = _validate_bundle_fields(request["bundle"])
    if type(target) is not dict:
        raise ValueError("TargetSnapshot fields mismatch")
    shots = _validate_bundle_cells(bundle)
    gates = _validate_bundle_gates(bundle)
    if bundle["payload_digest"] != _bundle_payload_digest(bundle):
        raise ValueError("compiled payload digest does not match the bundle")
    parents = tuple(
        str(bundle[name])
        for name in ("target_snapshot_digest", "sdk_lock_digest", "compile_log_digest")
    )
    header = bundle["header"]
    origin = str(header["data_origin"]) if type(header) is dict else ""
    _validate_artifact_header(bundle, parents, origin)
    _cross_check_bundle(bundle, target, shots, gates)
    wire = _canonical_bytes(bundle)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "bundle": json.loads(wire),
        "bundle_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_QUOTE_FIELDS = {
    "schema",
    "object_kind",
    "compiled_bundle_digest",
    "bundle_payload_digest",
    "circuit_count",
    "total_shots",
    "quoted_options",
    *_QUOTE_COUNTS,
    "resource_unit",
    "resource_derivation",
    "currency",
    "pricing_source",
    "priced_at_utc",
    "header",
}


def _validate_quoted_count(value: object, name: str) -> None:
    if type(value) is not dict or set(value) != {"state", "value", "blocker"}:
        raise ValueError("QuotedCount fields mismatch")
    if value["state"] == "unknown":
        if value["value"] is not None:
            raise ValueError("an unknown count cannot carry a value; zero is not unknown")
        _target_text(value["blocker"], "count blocker")
        return
    number = value["value"]
    if value["state"] != "known" or value["blocker"] is not None:
        raise ValueError("a known count needs a value and no blocker")
    if type(number) is not int or not 0 <= number <= 10**15:
        raise ValueError(f"{name} must be a bounded non-negative integer")


def _validate_quote_fields(quote: object) -> dict[str, object]:
    if type(quote) is not dict or set(quote) != _QUOTE_FIELDS:
        raise ValueError("CostQuote fields mismatch")
    if quote["schema"] != _COST_QUOTE_SCHEMA or quote["object_kind"] != "cost_quote":
        raise ValueError("unknown CostQuote schema")
    for name in ("compiled_bundle_digest", "bundle_payload_digest"):
        _worker_digest(quote[name], name)
    for name, maximum in (("circuit_count", 10_000_000), ("total_shots", 10**12)):
        count = quote[name]
        if type(count) is not int or not 0 < count <= maximum:
            raise ValueError(f"{name} must be bounded")
    _sorted_unique_names(quote["quoted_options"], "quoted options", allow_empty=True)
    for name in _QUOTE_COUNTS:
        _validate_quoted_count(quote[name], name)
    job = quote["job_count"]
    if type(job) is dict and job["value"] == 0:
        raise ValueError("a known job count must be positive")
    unit = _target_text(quote["resource_unit"], "resource unit")
    if unit.lower() in ("shot", "shots") or quote["resource_derivation"] == "shots_as_credits":
        raise ValueError("provider credits are not shots")
    if quote["resource_derivation"] not in ("provider_estimate", "published_rate_card"):
        raise ValueError("unknown resource derivation")
    if type(quote["currency"]) is not str or re.fullmatch(r"[A-Z]{3}", quote["currency"]) is None:
        raise ValueError("currency must be an ISO 4217 code")
    _target_text(quote["pricing_source"], "pricing source")
    price, priced_at = quote["price_minor_units"], quote["priced_at_utc"]
    known_price = type(price) is dict and price["state"] == "known"
    if known_price and (type(priced_at) is not str or _UTC_SECOND.fullmatch(priced_at) is None):
        raise ValueError("a known price needs its ISO-8601 UTC quote time")
    if not known_price and priced_at is not None:
        raise ValueError("an unknown price cannot carry a quote time")
    return quote


def _cross_check_quote(
    quote: dict[str, object], bundle: dict[str, object], target: dict[str, object]
) -> None:
    if quote["compiled_bundle_digest"] != hashlib.sha256(_canonical_bytes(bundle)).hexdigest():
        raise ValueError("quote prices a different compiled bundle")
    cells, options = bundle["cells"], bundle["provider_options"]
    if type(cells) is not list or type(options) is not list:
        raise ValueError("bundle sequences must be lists")
    if quote["bundle_payload_digest"] != bundle["payload_digest"] or quote["circuit_count"] != len(
        cells
    ):
        raise ValueError("quote payload or circuit count differs from the bundle")
    if quote["total_shots"] != sum(
        int(str(cell["shots"])) for cell in cells if type(cell) is dict
    ):
        raise ValueError("quoted total shots differ from the bundle")
    if quote["quoted_options"] != [pair[0] for pair in options if type(pair) is list]:
        raise ValueError("quote does not name every provider option the bundle runs with")
    header, bundle_header = quote["header"], bundle["header"]
    if type(header) is not dict or type(bundle_header) is not dict:
        raise ValueError("artifact headers must be objects")
    if header["data_origin"] != bundle_header["data_origin"]:
        raise ValueError("quote and bundle data origins differ")
    limit, job = target["max_circuits_per_job"], quote["job_count"]
    if type(limit) is not dict or type(job) is not dict:
        raise ValueError("limit and job count must be objects")
    if limit["state"] == "known":
        expected = -(-len(cells) // int(str(limit["value"])))
        if job["value"] != expected:
            raise ValueError("quoted job count differs from the target's circuit batch limit")
    elif job["state"] == "known":
        raise ValueError("a job count cannot be known while the batch limit is unknown")


def _roundtrip_cost_quote(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "target", "bundle", "quote"}:
        raise ValueError("quote request fields mismatch")
    bundle_request = {
        "op": "roundtrip_compiled_bundle",
        "target": request["target"],
        "bundle": request["bundle"],
    }
    _roundtrip_compiled_bundle(bundle_request)
    target, bundle = request["target"], request["bundle"]
    if type(target) is not dict or type(bundle) is not dict:
        raise ValueError("target and bundle must be objects")
    quote = _validate_quote_fields(request["quote"])
    header = quote["header"]
    origin = str(header["data_origin"]) if type(header) is dict else ""
    if origin not in ("provider_metadata", "synthetic_classical"):
        raise ValueError("a cost quote derives from provider metadata or a fixture")
    _validate_artifact_header(quote, (str(quote["compiled_bundle_digest"]),), origin)
    _cross_check_quote(quote, bundle, target)
    wire = _canonical_bytes(quote)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "quote": json.loads(wire),
        "quote_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_APPROVAL_FIELDS = {
    "schema",
    "object_kind",
    "approval_id",
    "compiled_bundle_digest",
    "bundle_payload_digest",
    "cost_quote_digest",
    "provider",
    "account_route",
    "device_name",
    "max_shots",
    "max_jobs",
    "max_resource_units",
    "max_price_minor_units",
    "currency",
    "valid_from_utc",
    "expires_at_utc",
    "approved_egress",
    "signature",
    "header",
}


def _validate_approval_signature(signature: object) -> str:
    fields = {"scheme", "key_id", "key_environment", "signature_sha256"}
    if type(signature) is not dict or set(signature) != fields:
        raise ValueError("ApprovalSignature fields mismatch")
    if signature["scheme"] != "ml_dsa_65":
        raise ValueError("unsupported approval signature scheme")
    _worker_digest(signature["key_id"], "approval key id")
    environment = signature["key_environment"]
    if environment not in _APPROVAL_ORIGINS:
        raise ValueError("approval key environment must be live or test")
    _worker_digest(signature["signature_sha256"], "approval signature digest")
    return _APPROVAL_ORIGINS[str(environment)]


def _validate_approval_fields(approval: object) -> dict[str, object]:
    if type(approval) is not dict or set(approval) != _APPROVAL_FIELDS:
        raise ValueError("OwnerApproval fields mismatch")
    if approval["schema"] != _OWNER_APPROVAL_SCHEMA or approval["object_kind"] != "owner_approval":
        raise ValueError("unknown OwnerApproval schema")
    for name in ("approval_id", "provider", "account_route", "device_name"):
        _target_text(approval[name], name)
    for name in ("compiled_bundle_digest", "bundle_payload_digest", "cost_quote_digest"):
        _worker_digest(approval[name], name)
    for name in ("max_shots", "max_jobs", "max_resource_units", "max_price_minor_units"):
        cap = approval[name]
        if type(cap) is not int or not 0 < cap <= 10**15:
            raise ValueError(f"{name} must be a bounded positive integer")
    currency = approval["currency"]
    if type(currency) is not str or re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("currency must be an ISO 4217 code")
    start, end = approval["valid_from_utc"], approval["expires_at_utc"]
    for value in (start, end):
        if type(value) is not str or _UTC_SECOND.fullmatch(value) is None:
            raise ValueError("approval validity bounds must be ISO-8601 UTC seconds")
    if str(end) <= str(start):
        raise ValueError("an approval must expire after it becomes valid")
    if approval["approved_egress"] not in _APPROVAL_EGRESS:
        raise ValueError("unknown approved data-egress scope")
    return approval


def _cross_check_approval(
    approval: dict[str, object],
    quote: dict[str, object],
    bundle: dict[str, object],
    target: dict[str, object],
) -> None:
    if approval["compiled_bundle_digest"] != hashlib.sha256(_canonical_bytes(bundle)).hexdigest():
        raise ValueError("approval covers a different compiled bundle")
    if approval["bundle_payload_digest"] != bundle["payload_digest"]:
        raise ValueError("approval covers a different compiled payload")
    if approval["cost_quote_digest"] != hashlib.sha256(_canonical_bytes(quote)).hexdigest():
        raise ValueError("approval covers a different cost quote")
    scope = [approval[name] for name in ("provider", "account_route", "device_name")]
    if scope != [target[name] for name in ("provider", "account_route", "device_name")]:
        raise ValueError("approval scope differs from the target route and device")
    known: dict[str, int] = {}
    for name in _QUOTE_COUNTS:
        count = quote[name]
        if type(count) is not dict or count["state"] != "known":
            raise ValueError(f"an approval cannot cover an unknown {name}")
        known[name] = int(str(count["value"]))
    for label, value, cap in (
        ("shots", quote["total_shots"], approval["max_shots"]),
        ("job_count", known["job_count"], approval["max_jobs"]),
        ("resource_units", known["resource_units"], approval["max_resource_units"]),
        ("price_minor_units", known["price_minor_units"], approval["max_price_minor_units"]),
    ):
        if int(str(value)) > int(str(cap)):
            raise ValueError(f"quoted {label} above the approved cap")
    if quote["currency"] != approval["currency"]:
        raise ValueError("quote and approval currencies differ")


def _roundtrip_owner_approval(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "target", "bundle", "quote", "approval"}:
        raise ValueError("approval request fields mismatch")
    quote_request = {name: request[name] for name in ("target", "bundle", "quote")}
    _roundtrip_cost_quote({"op": "roundtrip_cost_quote", **quote_request})
    target, bundle, quote = request["target"], request["bundle"], request["quote"]
    if type(target) is not dict or type(bundle) is not dict or type(quote) is not dict:
        raise ValueError("target, bundle and quote must be objects")
    approval = _validate_approval_fields(request["approval"])
    origin = _validate_approval_signature(approval["signature"])
    parents = (str(approval["compiled_bundle_digest"]), str(approval["cost_quote_digest"]))
    _validate_artifact_header(approval, parents, origin)
    _cross_check_approval(approval, quote, bundle, target)
    wire = _canonical_bytes(approval)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "approval": json.loads(wire),
        "approval_sha256": hashlib.sha256(wire).hexdigest(),
        "signature_verified": False,
        "hardware_submission_enabled": False,
    }


_JOB_FIELDS = {
    "ordinal",
    "name",
    "payload_sha256",
    "cell_ordinals",
    "state",
    "provider_job_id",
    "id_source",
    "boundary_at_utc",
    "provider_created_at_utc",
    "blocker",
}
_RECEIPT_FIELDS = {
    "schema",
    "object_kind",
    "attempt_id",
    "compiled_bundle_digest",
    "bundle_payload_digest",
    "owner_approval_digest",
    "provider",
    "account_route",
    "device_name",
    "jobs",
    "header",
}


def _validate_job_record(job: object, ordinal: int) -> list[int]:
    if type(job) is not dict or set(job) != _JOB_FIELDS:
        raise ValueError("SubmittedJob fields mismatch")
    if job["ordinal"] != ordinal:
        raise ValueError("receipt jobs must be ordered by ordinal from zero")
    name = job["name"]
    if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name) is None:
        raise ValueError("invalid provider job phase name")
    _worker_digest(job["payload_sha256"], "submitted payload digest")
    cells = job["cell_ordinals"]
    if type(cells) is not list or not cells or len(set(map(str, cells))) != len(cells):
        raise ValueError("a job maps a nonempty list of distinct cells")
    if any(type(cell) is not int or not 0 <= cell < 10_000_000 for cell in cells):
        raise ValueError("cell ordinals must be bounded non-negative integers")
    boundary = job["boundary_at_utc"]
    if type(boundary) is not str or _UTC_SECOND.fullmatch(boundary) is None:
        raise ValueError("submit boundary time must be an ISO-8601 UTC second")
    created = job["provider_created_at_utc"]
    if job["state"] == "submitted":
        if type(job["provider_job_id"]) is not str or job["id_source"] not in (
            "submit_response",
            "read_only_recovery",
        ):
            raise ValueError("a submitted job needs its provider job ID and its source")
        if job["blocker"] is not None:
            raise ValueError("a submitted job cannot carry a blocker")
        if created is not None and (
            type(created) is not str or _UTC_SECOND.fullmatch(created) is None
        ):
            raise ValueError("provider creation time must be an ISO-8601 UTC second")
    elif job["state"] == "recovery_required":
        if (
            job["provider_job_id"] is not None
            or job["id_source"] is not None
            or created is not None
        ):
            raise ValueError("an ambiguous job has no provider job ID or creation time yet")
        _target_text(job["blocker"], "recovery blocker")
    else:
        raise ValueError("a missing job ID is ambiguous, never a failed zero-cost job")
    return [int(str(cell)) for cell in cells]


def _validate_receipt_fields(receipt: object) -> dict[str, object]:
    if type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("ProviderBatchReceipt fields mismatch")
    if receipt["schema"] != _RECEIPT_SCHEMA or receipt["object_kind"] != "provider_batch_receipt":
        raise ValueError("unknown ProviderBatchReceipt schema")
    attempt = receipt["attempt_id"]
    if type(attempt) is not str or _UUID_TEXT.fullmatch(attempt) is None:
        raise ValueError("attempt ID must be canonical UUID text")
    for name in ("compiled_bundle_digest", "bundle_payload_digest", "owner_approval_digest"):
        _worker_digest(receipt[name], name)
    provider = receipt["provider"]
    if provider not in _PROVIDER_JOB_ID:
        raise ValueError("receipt provider must be ibm or iqm")
    for name in ("account_route", "device_name"):
        _target_text(receipt[name], name)
    jobs = receipt["jobs"]
    if type(jobs) is not list or not jobs:
        raise ValueError("a receipt needs a nonempty list of jobs")
    seen: set[str] = {str(attempt)}
    for ordinal, job in enumerate(jobs):
        _validate_job_record(job, ordinal)
        job_id = job["provider_job_id"] if type(job) is dict else None
        if job_id is None:
            continue
        if _PROVIDER_JOB_ID[str(provider)].fullmatch(str(job_id)) is None:
            raise ValueError("provider job ID does not have the provider's form")
        if str(job_id) in seen:
            raise ValueError("a provider job ID repeats or reuses the attempt ID")
        seen.add(str(job_id))
    return receipt


def _cross_check_receipt(
    receipt: dict[str, object], approval: dict[str, object], bundle: dict[str, object]
) -> None:
    bundle_digest = hashlib.sha256(_canonical_bytes(bundle)).hexdigest()
    if receipt["compiled_bundle_digest"] != bundle_digest:
        raise ValueError("receipt names a different compiled bundle")
    if receipt["bundle_payload_digest"] != bundle["payload_digest"]:
        raise ValueError("receipt names a different compiled payload")
    if receipt["owner_approval_digest"] != hashlib.sha256(_canonical_bytes(approval)).hexdigest():
        raise ValueError("receipt names a different owner approval")
    names = ("provider", "account_route", "device_name")
    if [receipt[name] for name in names] != [approval[name] for name in names]:
        raise ValueError("receipt route or device differs from the approval")
    jobs, cells = receipt["jobs"], bundle["cells"]
    if type(jobs) is not list or type(cells) is not list:
        raise ValueError("receipt jobs and bundle cells must be lists")
    if len(jobs) > int(str(approval["max_jobs"])):
        raise ValueError("receipt has more provider jobs than approved")
    mapped = sorted(int(str(cell)) for job in jobs for cell in job["cell_ordinals"])
    if mapped != list(range(len(cells))):
        raise ValueError("receipt does not map every compiled cell exactly once")


def _roundtrip_provider_batch_receipt(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "target", "bundle", "quote", "approval", "receipt"}:
        raise ValueError("receipt request fields mismatch")
    approval_request = {name: request[name] for name in ("target", "bundle", "quote", "approval")}
    _roundtrip_owner_approval({"op": "roundtrip_owner_approval", **approval_request})
    bundle, approval = request["bundle"], request["approval"]
    if type(bundle) is not dict or type(approval) is not dict:
        raise ValueError("bundle and approval must be objects")
    receipt = _validate_receipt_fields(request["receipt"])
    header, bundle_header, signature = receipt["header"], bundle["header"], approval["signature"]
    if type(header) is not dict or type(bundle_header) is not dict or type(signature) is not dict:
        raise ValueError("headers and the approval signature must be objects")
    origin = str(header["data_origin"])
    if origin not in ("provider_metadata", "synthetic_classical"):
        raise ValueError("a receipt derives from provider responses or a fixture")
    parents = (str(receipt["compiled_bundle_digest"]), str(receipt["owner_approval_digest"]))
    _validate_artifact_header(receipt, parents, origin)
    _cross_check_receipt(receipt, approval, bundle)
    if origin != bundle_header["data_origin"]:
        raise ValueError("receipt and bundle data origins differ")
    if origin == "provider_metadata" and signature["key_environment"] == "test":
        raise ValueError("a test-key approval cannot back a provider receipt")
    wire = _canonical_bytes(receipt)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "receipt": json.loads(wire),
        "receipt_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_RESULT_FIELDS = {
    "cell_ordinal",
    "key",
    "job_ordinal",
    "provider_job_id",
    "pub_index",
    "binding_index",
    "measured_qubits",
    "requested_shots",
    "actual_shots",
    "counts",
    "raw_payload_sha256",
}
_RESULT_SET_FIELDS = {
    "schema",
    "object_kind",
    "attempt_id",
    "provider_batch_receipt_digest",
    "compiled_bundle_digest",
    "expected_cell_count",
    "results",
    "gaps",
    "header",
}


def _bounded(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value < maximum:
        raise ValueError(f"{name} must be a bounded non-negative integer")
    return value


def _validate_counts(counts: object, width: int, shots: int) -> None:
    if type(counts) is not list or not 0 < len(counts) <= 4096:
        raise ValueError("raw counts must be a nonempty list of at most 4096 outcomes")
    total, names = 0, []
    for entry in counts:
        if type(entry) is not list or len(entry) != 2:
            raise ValueError("each raw count is one bitstring and one count")
        bits, count = entry
        if type(bits) is not str or len(bits) != width or set(bits) - {"0", "1"}:
            raise ValueError("raw bitstring width differs from the measured qubits")
        if type(count) is not int or count < 1:
            raise ValueError("raw counts must be positive integers; zeros pad sparse data")
        total += count
        names.append(bits)
    if names != sorted(set(names)) or total != shots:
        raise ValueError("raw counts must be sorted, unique and sum to the actual shots")


def _validate_cell_result(result: object) -> dict[str, object]:
    if type(result) is not dict or set(result) != _RESULT_FIELDS:
        raise ValueError("CellResult fields mismatch")
    for name, maximum in (
        ("cell_ordinal", 10_000_000),
        ("job_ordinal", 100_000),
        ("pub_index", 10_000_000),
        ("binding_index", 1_000_000),
    ):
        _bounded(result[name], name, maximum)
    key = result["key"]
    if type(key) is not dict or set(key) != _KEY_FIELDS:
        raise ValueError("CellKey fields mismatch")
    _target_text(result["provider_job_id"], "provider job ID")
    qubits = result["measured_qubits"]
    if type(qubits) is not list or not 0 < len(qubits) <= 64:
        raise ValueError("measured qubits must be a list of one to 64 qubits")
    if len({_bounded(qubit, "measured qubit", 100_000) for qubit in qubits}) != len(qubits):
        raise ValueError("a qubit cannot be measured into two bits")
    requested, actual = result["requested_shots"], result["actual_shots"]
    if type(requested) is not int or not 0 < requested <= 10_000_000:
        raise ValueError("requested shots must be a bounded positive integer")
    if type(actual) is not int or actual != requested:
        raise ValueError("actual shots differ from the requested shots; quarantine the cell")
    _validate_counts(result["counts"], len(qubits), requested)
    _worker_digest(result["raw_payload_sha256"], "raw payload digest")
    return result


def _validate_cell_gap(gap: object) -> int:
    if type(gap) is not dict or set(gap) != {
        "cell_ordinal",
        "status",
        "reason",
        "raw_payload_sha256",
    }:
        raise ValueError("CellGap fields mismatch")
    ordinal = _bounded(gap["cell_ordinal"], "cell_ordinal", 10_000_000)
    _target_text(gap["reason"], "gap reason")
    if gap["status"] == "quarantined":
        _worker_digest(gap["raw_payload_sha256"], "quarantined raw payload digest")
    elif gap["status"] != "missing" or gap["raw_payload_sha256"] is not None:
        raise ValueError("a gap is missing without raw data or quarantined with it")
    return ordinal


def _validate_result_set_fields(result_set: object) -> dict[str, object]:
    if type(result_set) is not dict or set(result_set) != _RESULT_SET_FIELDS:
        raise ValueError("RawResultSet fields mismatch")
    if result_set["schema"] != _RESULT_SET_SCHEMA or result_set["object_kind"] != "raw_result_set":
        raise ValueError("unknown RawResultSet schema")
    attempt = result_set["attempt_id"]
    if type(attempt) is not str or _UUID_TEXT.fullmatch(attempt) is None:
        raise ValueError("attempt ID must be canonical UUID text")
    for name in ("provider_batch_receipt_digest", "compiled_bundle_digest"):
        _worker_digest(result_set[name], name)
    expected = result_set["expected_cell_count"]
    if type(expected) is not int or not 0 < expected <= 10_000_000:
        raise ValueError("expected cell count must be a bounded positive integer")
    results, gaps = result_set["results"], result_set["gaps"]
    if type(results) is not list or type(gaps) is not list:
        raise ValueError("results and gaps must be lists")
    found = [int(str(_validate_cell_result(item)["cell_ordinal"])) for item in results]
    missing = [_validate_cell_gap(gap) for gap in gaps]
    if found != sorted(set(found)) or missing != sorted(set(missing)):
        raise ValueError("results and gaps must be ordered by unique cell ordinal")
    if sorted(found + missing) != list(range(expected)):
        raise ValueError("every expected cell needs one result or one recorded gap")
    units = {(r["provider_job_id"], r["pub_index"], r["binding_index"]) for r in results}
    if len(units) != len(results):
        raise ValueError("two cells cannot share one provider result unit (merged PUB)")
    return result_set


def _check_planned_cells(planned: object, bundle: dict[str, object]) -> list[dict[str, object]]:
    cells = bundle["cells"]
    if type(planned) is not list or type(cells) is not list or len(planned) != len(cells):
        raise ValueError("validation needs one planned cell per compiled cell")
    checked = []
    for cell, compiled in zip(planned, cells, strict=True):
        record = _validate_planned_cell(cell)
        if type(compiled) is not dict:
            raise ValueError("compiled cells must be objects")
        if hashlib.sha256(_canonical_bytes(record)).hexdigest() != compiled["planned_cell_digest"]:
            raise ValueError("planned cell differs from the compiled bundle")
        if record["shots"] != compiled["shots"]:
            raise ValueError("compiled shots differ from the planned cell")
        checked.append(record)
    return checked


def _cross_check_result(
    result: dict[str, object],
    planned: dict[str, object],
    jobs: list[dict[str, object]],
    layout: set[object],
) -> None:
    if result["key"] != planned["key"] or result["requested_shots"] != planned["shots"]:
        raise ValueError("result key or requested shots differ from the planned cell")
    qubits = result["measured_qubits"]
    if type(qubits) is not list or len(qubits) != planned["measurement_width"]:
        raise ValueError("measured width differs from the planned cell")
    if not layout.issuperset(qubits):
        raise ValueError("a measured qubit is outside the compiled layout")
    ordinal = int(str(result["job_ordinal"]))
    if ordinal >= len(jobs):
        raise ValueError("result names a job the receipt does not hold")
    job = jobs[ordinal]
    if job["state"] != "submitted" or job["provider_job_id"] != result["provider_job_id"]:
        raise ValueError("result job is ambiguous or its ID differs from the receipt")
    cells = job["cell_ordinals"]
    if type(cells) is not list or result["cell_ordinal"] not in cells:
        raise ValueError("result cell is not in its receipt job")
    if cells.index(result["cell_ordinal"]) != result["pub_index"] or result["binding_index"] != 0:
        raise ValueError("result is not at its receipt circuit position with binding zero")


def _cross_check_result_set(
    result_set: dict[str, object], receipt: dict[str, object], bundle: dict[str, object]
) -> None:
    if (
        result_set["provider_batch_receipt_digest"]
        != hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    ):
        raise ValueError("results name a different provider receipt")
    if result_set["compiled_bundle_digest"] != receipt["compiled_bundle_digest"]:
        raise ValueError("results, receipt and bundle name different compiled bundles")
    if result_set["attempt_id"] != receipt["attempt_id"]:
        raise ValueError("results belong to another attempt")
    cells = bundle["cells"]
    if type(cells) is not list or result_set["expected_cell_count"] != len(cells):
        raise ValueError("expected cell count differs from the compiled bundle")
    header, receipt_header = result_set["header"], receipt["header"]
    if type(header) is not dict or type(receipt_header) is not dict:
        raise ValueError("artifact headers must be objects")
    if header["data_origin"] != _RESULT_ORIGINS[str(receipt_header["data_origin"])]:
        raise ValueError("result origin does not match the receipt; a fixture is never hardware")


def _roundtrip_raw_result_set(request: dict[str, object]) -> dict[str, object]:
    names = ("target", "bundle", "quote", "approval", "receipt")
    if set(request) != {"op", *names, "planned", "results"}:
        raise ValueError("result request fields mismatch")
    receipt_request = {name: request[name] for name in names}
    _roundtrip_provider_batch_receipt(
        {"op": "roundtrip_provider_batch_receipt", **receipt_request}
    )
    bundle, receipt = request["bundle"], request["receipt"]
    if type(bundle) is not dict or type(receipt) is not dict:
        raise ValueError("bundle and receipt must be objects")
    result_set = _validate_result_set_fields(request["results"])
    header = result_set["header"]
    origin = str(header["data_origin"]) if type(header) is dict else ""
    if origin not in ("provider_result", "synthetic_classical"):
        raise ValueError("a raw result set derives from provider results or a fixture")
    parents = (
        str(result_set["provider_batch_receipt_digest"]),
        str(result_set["compiled_bundle_digest"]),
    )
    _validate_artifact_header(result_set, parents, origin)
    _cross_check_result_set(result_set, receipt, bundle)
    planned = _check_planned_cells(request["planned"], bundle)
    jobs, layout, results = receipt["jobs"], bundle["layout"], result_set["results"]
    if type(jobs) is not list or type(layout) is not list or type(results) is not list:
        raise ValueError("receipt jobs, layout and results must be lists")
    for result in results:
        _cross_check_result(result, planned[int(str(result["cell_ordinal"]))], jobs, set(layout))
    wire = _canonical_bytes(result_set)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "results": json.loads(wire),
        "results_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_PLAN_FIELDS = {
    "schema",
    "object_kind",
    "circuit_plan_digest",
    "n_qubits",
    "basis_id",
    "shots",
    "observable_ids",
    "logical_to_clbit",
    "display_order_clbits",
    "register_name",
    "physical_qubits",
    "pauli_label_convention",
}
_ROW_FIELDS = {
    "cell_ordinal",
    "key",
    "measurement_request_digest",
    "shots",
    "values",
    "covariance",
    "missing_reason",
}


def _observables(basis: str, width: int) -> list[str]:
    singles = [f"{basis}{wire}" for wire in range(width)]
    return singles + [f"{basis}{wire}{basis}{wire + 1}" for wire in range(width - 1)]


def _validate_measurement_plan(plan: object) -> dict[str, object]:
    if type(plan) is not dict or set(plan) != _PLAN_FIELDS:
        raise ValueError("measurement plan fields mismatch")
    if plan["schema"] != _MEASUREMENT_SCHEMA or plan["object_kind"] != "measurement_plan":
        raise ValueError("unsupported measurement plan schema")
    _worker_digest(plan["circuit_plan_digest"], "circuit plan digest")
    width, basis, shots = plan["n_qubits"], plan["basis_id"], plan["shots"]
    if type(width) is not int or width not in (4, 8) or basis not in ("X", "Y", "Z"):
        raise ValueError("measurement plan requires n4/n8 and an X, Y or Z basis")
    if type(shots) is not int or not 0 < shots <= 1_000_000:
        raise ValueError("measurement shots must be a bounded positive integer")
    if plan["observable_ids"] != _observables(str(basis), width):
        raise ValueError("measurement observable order or basis mismatch")
    for name in ("logical_to_clbit", "display_order_clbits"):
        bits = plan[name]
        if type(bits) is not list or any(type(bit) is not int for bit in bits):
            raise ValueError("measurement bit maps must be integer lists")
        if sorted(bits) != list(range(width)):
            raise ValueError("measurement bit maps must be complete permutations")
    _target_text(plan["register_name"], "measurement register")
    physical = plan["physical_qubits"]
    if physical is not None and (
        type(physical) is not list
        or len(physical) != width
        or any(type(wire) is not int or not 0 <= wire < 128 for wire in physical)
        or len(set(physical)) != width
    ):
        raise ValueError("physical qubits must be explicit unique bounded indices")
    if plan["pauli_label_convention"] != "logical_q0_low_index_v1":
        raise ValueError("unsupported Pauli label convention")
    return plan


def _sampled_estimate(
    plan: dict[str, object], result: dict[str, object]
) -> tuple[int, list[float], list[list[float]] | None]:
    physical, measured = plan["physical_qubits"], result["measured_qubits"]
    if (
        type(physical) is not list
        or type(measured) is not list
        or sorted(physical) != sorted(measured)
    ):
        raise ValueError("measurement plan layout differs from the measured qubits")
    position = [measured.index(wire) for wire in physical]
    width = len(physical) * 2 - 1
    counts = result["counts"]
    if type(counts) is not list:
        raise ValueError("raw counts must be a list")
    totals, joint, shots = [0] * width, [[0] * width for _ in range(width)], 0
    for bits, count in counts:
        single = [1 - 2 * int(bits[index]) for index in position]
        values = single + [single[i] * single[i + 1] for i in range(len(single) - 1)]
        shots += count
        for i, value in enumerate(values):
            totals[i] += count * value
            for j, other in enumerate(values):
                joint[i][j] += count * value * other
    if shots != plan["shots"]:
        raise ValueError("raw counts do not match requested shots")
    means = [total / shots for total in totals]
    if shots == 1:
        return shots, means, None
    denominator = shots * shots * (shots - 1)
    covariance = [
        [(shots * joint[i][j] - totals[i] * totals[j]) / denominator for j in range(width)]
        for i in range(width)
    ]
    return shots, means, covariance


def _validate_feature_row(row: object, width: int) -> dict[str, object]:
    if type(row) is not dict or set(row) != _ROW_FIELDS:
        raise ValueError("FeatureRow fields mismatch")
    _bounded(row["cell_ordinal"], "cell_ordinal", 10_000_000)
    key = row["key"]
    if type(key) is not dict or set(key) != _KEY_FIELDS:
        raise ValueError("CellKey fields mismatch")
    _worker_digest(row["measurement_request_digest"], "measurement request digest")
    values, covariance, shots = row["values"], row["covariance"], row["shots"]
    if values is None:
        if shots is not None or covariance is not None:
            raise ValueError("a missing row carries no shots or covariance")
        _target_text(row["missing_reason"], "missing reason")
        return row
    if row["missing_reason"] is not None or type(shots) is not int or shots < 1:
        raise ValueError("a measured row has positive shots and no missing reason")
    if type(values) is not list or len(values) != width:
        raise ValueError("every measured row has one value per observable")
    if any(not -1.0 <= _f64_token(value) <= 1.0 for value in values):
        raise ValueError("a sampled mean of a ±1 observable lies in [-1, 1]")
    if shots == 1:
        if covariance is not None:
            raise ValueError("one shot has no sampling covariance")
        return row
    if type(covariance) is not list or len(covariance) != width:
        raise ValueError("covariance must be a square list matching the features")
    for line in covariance:
        if type(line) is not list or len(line) != width:
            raise ValueError("covariance must be a square list matching the features")
    return row


def _f64_token(value: object) -> float:
    if type(value) is not dict or set(value) != {"$f64"} or type(value["$f64"]) is not str:
        raise ValueError("feature numbers must be f64 tags")
    parsed = float.fromhex(value["$f64"])
    if not math.isfinite(parsed):
        raise ValueError("feature numbers must be finite")
    return parsed


def _validate_feature_fields(batch: object) -> dict[str, object]:
    fields = {"schema", "object_kind", "raw_result_set_digest", "estimator", "basis_id"}
    if type(batch) is not dict or set(batch) != fields | {"observable_ids", "rows", "header"}:
        raise ValueError("FeatureBatch fields mismatch")
    if batch["schema"] != _FEATURE_SCHEMA or batch["object_kind"] != "feature_batch":
        raise ValueError("unknown FeatureBatch schema")
    _worker_digest(batch["raw_result_set_digest"], "raw result set digest")
    if batch["estimator"] != "sampled_mean_v1":
        raise ValueError(
            "only sampled means of raw counts are features; quasi-probabilities are not"
        )
    if batch["basis_id"] not in ("X", "Y", "Z"):
        raise ValueError("feature basis must be X, Y or Z")
    names, rows = batch["observable_ids"], batch["rows"]
    if type(names) is not list or not names or len(set(map(str, names))) != len(names):
        raise ValueError("observable IDs must be a nonempty unique list")
    if type(rows) is not list or not rows:
        raise ValueError("feature rows must be a nonempty list")
    ordinals = [int(str(_validate_feature_row(row, len(names))["cell_ordinal"])) for row in rows]
    if ordinals != sorted(set(ordinals)):
        raise ValueError("feature rows must follow the result set's cell order")
    return batch


def _cross_check_feature_row(
    row: dict[str, object],
    plan: dict[str, object],
    planned: dict[str, object],
    result: dict[str, object] | None,
    names: object,
) -> None:
    if row["key"] != planned["key"] or plan["observable_ids"] != names:
        raise ValueError("feature row key or columns differ from the planned cell")
    if row["measurement_request_digest"] != hashlib.sha256(_canonical_bytes(plan)).hexdigest():
        raise ValueError("feature row names another measurement request")
    if result is None:
        if row["values"] is not None:
            raise ValueError("a cell without a result must stay missing, never zero-filled")
        return
    if row["values"] is None:
        raise ValueError("a measured cell cannot be reported missing")
    shots, means, covariance = _sampled_estimate(plan, result)
    values, reported = row["values"], row["covariance"]
    if type(values) is not list:
        raise ValueError("feature values must be a list")
    found = [_f64_token(value) for value in values]
    matrix: list[list[float]] | None = None
    if type(reported) is list:
        matrix = [[_f64_token(entry) for entry in line] for line in reported]
    if (row["shots"], found, matrix) != (shots, means, covariance):
        raise ValueError("feature values differ from the raw counts")


def _checked_plans(plans: object, planned: object) -> list[dict[str, object]]:
    if type(plans) is not list or type(planned) is not list or len(plans) != len(planned):
        raise ValueError("validation needs one measurement plan per compiled cell")
    checked = [_validate_measurement_plan(plan) for plan in plans]
    for plan, cell in zip(checked, planned, strict=True):
        if type(cell) is not dict or plan["circuit_plan_digest"] != cell["circuit_plan_digest"]:
            raise ValueError("measurement plan does not measure the planned circuit")
    return checked


def _roundtrip_feature_batch(request: dict[str, object]) -> dict[str, object]:
    names = ("target", "bundle", "quote", "approval", "receipt", "planned", "results")
    if set(request) != {"op", *names, "plans", "features"}:
        raise ValueError("feature request fields mismatch")
    _roundtrip_raw_result_set({"op": "roundtrip_raw_result_set", **{n: request[n] for n in names}})
    result_set, planned = request["results"], request["planned"]
    if type(result_set) is not dict or type(planned) is not list:
        raise ValueError("results must be an object and planned cells a list")
    plans = _checked_plans(request["plans"], planned)
    batch = _validate_feature_fields(request["features"])
    header, result_header = batch["header"], result_set["header"]
    if type(header) is not dict or type(result_header) is not dict:
        raise ValueError("artifact headers must be objects")
    origin = str(header["data_origin"])
    if origin not in ("provider_result", "synthetic_classical"):
        raise ValueError("a feature batch derives from provider results or a fixture")
    _validate_artifact_header(batch, (str(batch["raw_result_set_digest"]),), origin)
    if batch["raw_result_set_digest"] != hashlib.sha256(_canonical_bytes(result_set)).hexdigest():
        raise ValueError("features name a different raw result set")
    if origin != result_header["data_origin"]:
        raise ValueError("feature origin differs from the raw results")
    rows, found = batch["rows"], result_set["results"]
    if type(rows) is not list or type(found) is not list:
        raise ValueError("feature rows and results must be lists")
    expected = [index for index, plan in enumerate(plans) if plan["basis_id"] == batch["basis_id"]]
    if [row["cell_ordinal"] for row in rows] != expected:
        raise ValueError("feature rows must cover every cell measured in this basis, in order")
    by_cell = {result["cell_ordinal"]: result for result in found}
    for row in rows:
        index = row["cell_ordinal"]
        args = (plans[index], planned[index], by_cell.get(index), batch["observable_ids"])
        _cross_check_feature_row(row, *args)
    wire = _canonical_bytes(batch)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "features": json.loads(wire),
        "features_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_REPORT_FIELDS = {
    "schema",
    "object_kind",
    "protocol_digest",
    "split_digest",
    "evaluated_split",
    "execution_class",
    "provider",
    "receipt_digests",
    "metric",
    "arms",
    "outcomes",
    "arm_metrics",
    "group_deltas",
    "mean_delta",
    "interval_method",
    "interval_seed",
    "interval",
    "relevant",
    "costs",
    "currency",
    "header",
}


def _validate_report_execution(report: dict[str, object]) -> None:
    if report["evaluated_split"] not in ("dev", "test"):
        raise ValueError("an evaluation reads the dev or the test split, never train")
    execution, provider, receipts = (
        report["execution_class"],
        report["provider"],
        report["receipt_digests"],
    )
    if execution not in ("hardware", "sampled_simulation", "exact_simulation", "classical_only"):
        raise ValueError("unknown execution class")
    if type(receipts) is not list or receipts != sorted(set(map(str, receipts))):
        raise ValueError("receipt digests must be a sorted unique list")
    for digest in receipts:
        _worker_digest(digest, "receipt digest")
    if execution == "hardware":
        if provider not in ("ibm", "iqm") or not receipts:
            raise ValueError("a hardware evaluation names one provider and its receipts")
    elif provider is not None or receipts:
        raise ValueError("only a hardware evaluation names a provider or receipts")
    if report["metric"] not in ("accuracy", "mean_log_loss", "mse", "mae"):
        raise ValueError("v1 reports per-sample decomposable metrics only")


def _validate_outcomes(report: dict[str, object]) -> list[tuple[str, str, str, float]]:
    arms, outcomes = report["arms"], report["outcomes"]
    if type(arms) is not list or not 2 <= len(arms) <= 32 or arms != sorted(set(map(str, arms))):
        raise ValueError("arms must be a sorted unique list of two to 32")
    if type(outcomes) is not list or not outcomes:
        raise ValueError("outcomes must be a nonempty list")
    parsed = []
    for outcome in outcomes:
        if type(outcome) is not dict or set(outcome) != {
            "sample_id",
            "source_group",
            "arm",
            "value",
        }:
            raise ValueError("SampleOutcome fields mismatch")
        names = [
            _target_text(outcome[name], name) for name in ("arm", "sample_id", "source_group")
        ]
        value = _f64_token(outcome["value"])
        if value < 0.0 or (report["metric"] == "accuracy" and value not in (0.0, 1.0)):
            raise ValueError("a per-sample contribution is non-negative, and 0 or 1 for accuracy")
        parsed.append((names[0], names[1], names[2], value))
    keys = [(arm, sample) for arm, sample, _, _ in parsed]
    if keys != sorted(set(keys)) or any(arm not in arms for arm, _ in keys):
        raise ValueError("outcomes must be ordered by arm and unique sample, in registered arms")
    return parsed


def _validate_report_numbers(report: dict[str, object]) -> None:
    for name in ("arm_metrics", "group_deltas"):
        pairs = report[name]
        if type(pairs) is not list or not pairs:
            raise ValueError(f"{name} must be a nonempty list of named values")
        for pair in pairs:
            if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
                raise ValueError(f"{name} must be a nonempty list of named values")
            _f64_token(pair[1])
    _f64_token(report["mean_delta"])
    bounds, seed = report["interval"], report["interval_seed"]
    if report["interval_method"] != "paired_group_bootstrap_percentile_v1":
        raise ValueError("unknown interval method")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("interval seed must be a non-negative 63-bit integer")
    if type(bounds) is not list or len(bounds) != 2:
        raise ValueError("interval must be a lower and an upper bound")
    if _f64_token(bounds[0]) > _f64_token(bounds[1]):
        raise ValueError("interval lower bound exceeds the upper bound")
    if type(report["relevant"]) is not bool:
        raise ValueError("relevance must be a boolean")
    costs = report["costs"]
    if (
        type(costs) is not list
        or [item[0] if type(item) is list and item else None for item in costs] != _COST_ITEMS
    ):
        raise ValueError("costs must name each cost item once, in order")
    for item in costs:
        if len(item) != 2:
            raise ValueError("costs must be [name, QuotedCount] pairs")
        _validate_quoted_count(item[1], str(item[0]))
    currency = report["currency"]
    if type(currency) is not str or re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("currency must be an ISO 4217 code")


def _worker_statistics(
    parsed: list[tuple[str, str, str, float]], protocol: dict[str, object], seed: int
) -> tuple[list[list[object]], list[list[object]], float, list[float]]:
    by_arm: dict[str, dict[str, tuple[str, float]]] = {}
    for arm, sample, group, value in parsed:
        by_arm.setdefault(arm, {})[sample] = (group, value)
    metrics: list[list[object]] = [
        [arm, sum(by_arm[arm][s][1] for s in sorted(by_arm[arm])) / len(by_arm[arm])]
        for arm in sorted(by_arm)
    ]
    classical = by_arm.get(str(protocol["classical_arm"]), {})
    quantum = by_arm.get(str(protocol["quantum_arm"]), {})
    if not quantum or set(classical) != set(quantum):
        raise ValueError("the primary arms must hold paired outcomes for the same samples")
    sign = 1.0 if protocol["contrast_direction"] == "quantum_minus_classical" else -1.0
    groups: dict[str, list[float]] = {}
    for sample in sorted(quantum):
        group, value = quantum[sample]
        groups.setdefault(group, []).append(sign * (value - classical[sample][1]))
    deltas = [sum(groups[g]) / len(groups[g]) for g in sorted(groups)]
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(2000)
    )
    named: list[list[object]] = [[g, d] for g, d in zip(sorted(groups), deltas, strict=True)]
    return metrics, named, sum(deltas) / len(deltas), [means[50], means[1949]]


def _check_report_pairing(
    report: dict[str, object], parsed: list[tuple[str, str, str, float]], split: dict[str, object]
) -> None:
    groups = split["dev_groups"] if report["evaluated_split"] == "dev" else split["test_groups"]
    if type(groups) is not list or any(group not in groups for _, _, group, _ in parsed):
        raise ValueError(
            "an evaluation cannot mix splits; every group belongs to the evaluated one"
        )
    samples: dict[str, set[str]] = {}
    group_of: dict[str, str] = {}
    for arm, sample, group, _ in parsed:
        samples.setdefault(arm, set()).add(sample)
        if group_of.setdefault(sample, group) != group:
            raise ValueError("one sample belongs to one source group in every arm")
    arms = report["arms"]
    if type(arms) is not list or set(samples) != set(arms):
        raise ValueError("every evaluated sample needs a paired outcome in every arm")
    if any(found != set(group_of) for found in samples.values()):
        raise ValueError("every evaluated sample needs a paired outcome in every arm")


def _check_report_receipts(report: dict[str, object], receipts: object) -> None:
    if type(receipts) is not list:
        raise ValueError("receipts must be a list")
    digests = []
    for receipt in receipts:
        record = _validate_receipt_fields(receipt)
        header = record["header"]
        origin = str(header["data_origin"]) if type(header) is dict else ""
        parents = (str(record["compiled_bundle_digest"]), str(record["owner_approval_digest"]))
        _validate_artifact_header(record, parents, origin)
        if record["provider"] != report["provider"]:
            raise ValueError("an evaluation cannot mix providers")
        if origin != "provider_metadata":
            raise ValueError("a hardware evaluation needs provider receipts, not fixtures")
        digests.append(hashlib.sha256(_canonical_bytes(record)).hexdigest())
    if sorted(digests) != report["receipt_digests"]:
        raise ValueError("report receipts differ from the receipts given")


def _cross_check_report(
    report: dict[str, object],
    protocol: dict[str, object],
    split: dict[str, object],
    parsed: list[tuple[str, str, str, float]],
) -> None:
    if report["protocol_digest"] != hashlib.sha256(_canonical_bytes(protocol)).hexdigest():
        raise ValueError("the report names a different frozen protocol")
    split_digest = hashlib.sha256(_canonical_bytes(split)).hexdigest()
    if report["split_digest"] != split_digest or protocol["split_digest"] != split_digest:
        raise ValueError("report, protocol and split name different splits")
    if report["metric"] != protocol["primary_metric"] or report["arms"] != protocol["arms"]:
        raise ValueError("the report's metric or arms differ from the protocol's locked ones")
    _check_report_pairing(report, parsed, split)


def _check_report_statistics(
    report: dict[str, object],
    parsed: list[tuple[str, str, str, float]],
    protocol: dict[str, object],
) -> None:
    metrics, deltas, mean, bounds = _worker_statistics(
        parsed, protocol, int(str(report["interval_seed"]))
    )
    reported = [report[name] for name in ("arm_metrics", "group_deltas")]
    decoded = [
        [[pair[0], _f64_token(pair[1])] for pair in pairs]
        for pairs in reported
        if type(pairs) is list
    ]
    if decoded != [metrics, deltas] or _f64_token(report["mean_delta"]) != mean:
        raise ValueError("reported metrics or paired deltas differ from the outcomes")
    interval = report["interval"]
    if type(interval) is not list or [_f64_token(bound) for bound in interval] != bounds:
        raise ValueError("reported interval differs from the seeded group bootstrap")
    threshold = _f64_token(protocol["minimum_relevant_delta"])
    if report["relevant"] != (bounds[0] > threshold):
        raise ValueError("relevance must be the interval's lower bound above the protocol delta")


def _roundtrip_evaluation_report(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "task", "split", "protocol", "receipts", "report"}:
        raise ValueError("evaluation request fields mismatch")
    _roundtrip_task_split(
        {"op": "roundtrip_task_split", "task": request["task"], "split": request["split"]}
    )
    _roundtrip_experiment_protocol(
        {"op": "roundtrip_experiment_protocol", "protocol": request["protocol"]}
    )
    protocol, split, report = request["protocol"], request["split"], request["report"]
    if type(protocol) is not dict or type(split) is not dict:
        raise ValueError("protocol and split must be objects")
    if type(report) is not dict or set(report) != _REPORT_FIELDS:
        raise ValueError("EvaluationReport fields mismatch")
    if report["schema"] != _REPORT_SCHEMA or report["object_kind"] != "evaluation_report":
        raise ValueError("unknown EvaluationReport schema")
    for name in ("protocol_digest", "split_digest"):
        _worker_digest(report[name], name)
    _validate_report_execution(report)
    _validate_report_numbers(report)
    parsed = _validate_outcomes(report)
    header = report["header"]
    origin = str(header["data_origin"]) if type(header) is dict else ""
    if origin not in ("provider_result", "synthetic_classical"):
        raise ValueError("an evaluation derives from provider results or a fixture")
    receipt_digests = report["receipt_digests"]
    parents = (str(report["protocol_digest"]), str(report["split_digest"]))
    _validate_artifact_header(report, (*parents, *map(str, receipt_digests or [])), origin)
    _cross_check_report(report, protocol, split, parsed)
    _check_report_receipts(report, request["receipts"])
    if (origin == "provider_result") != (report["execution_class"] == "hardware"):
        raise ValueError("only a hardware evaluation derives from provider results")
    _check_report_statistics(report, parsed, protocol)
    wire = _canonical_bytes(report)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "report": json.loads(wire),
        "report_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_STUDENT_TEXT = ("student_id", "architecture", "optimizer", "device", "runtime", "runtime_version")
_STUDENT_DIGESTS = (
    "protocol_digest",
    "model_digest",
    "compressor_digest",
    "teacher_digest",
    "weights_digest",
    "config_digest",
)
_STUDENT_FIELDS = {
    "schema",
    "object_kind",
    *_STUDENT_TEXT,
    *_STUDENT_DIGESTS,
    "teacher_kind",
    "targets",
    "training_groups",
    "learning_rate",
    "epochs",
    "seed",
    "heldout_split",
    "fidelity_metric",
    "fidelity",
    "utility_metric",
    "utility",
    "output_class",
    "header",
}


def _validate_student_fields(student: object) -> dict[str, object]:
    if type(student) is not dict or set(student) != _STUDENT_FIELDS:
        raise ValueError("StudentArtifact fields mismatch")
    if student["schema"] != _STUDENT_SCHEMA or student["object_kind"] != "student_artifact":
        raise ValueError("unknown StudentArtifact schema")
    if student["output_class"] != "classical_offline":
        raise ValueError("a student runs classically; its output is never quantum or hardware")
    for name in _STUDENT_TEXT:
        _target_text(student[name], name)
    for name in _STUDENT_DIGESTS:
        _worker_digest(student[name], name)
    if student["teacher_kind"] not in _TEACHER_ORIGINS:
        raise ValueError("a teacher is hardware or simulated quantum features")
    for name in ("targets", "training_groups"):
        values = student[name]
        if type(values) is not list or not values or len(set(map(str, values))) != len(values):
            raise ValueError(f"{name} must be a nonempty unique list")
    groups = student["training_groups"]
    if groups != sorted(map(str, groups if type(groups) is list else [])):
        raise ValueError("training groups must be sorted")
    epochs, seed = student["epochs"], student["seed"]
    if _f64_token(student["learning_rate"]) <= 0.0:
        raise ValueError("learning rate must be positive")
    if type(epochs) is not int or not 0 < epochs <= 1_000_000:
        raise ValueError("epochs must be a bounded positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative 63-bit integer")
    if student["heldout_split"] not in ("dev", "test"):
        raise ValueError("held-out scores come from the dev or the test split")
    if student["fidelity_metric"] not in ("feature_mse", "feature_mae"):
        raise ValueError("unknown fidelity metric")
    if _f64_token(student["fidelity"]) < 0.0:
        raise ValueError("a feature error cannot be negative")
    metric, utility = student["utility_metric"], _f64_token(student["utility"])
    if metric not in ("accuracy", "mean_log_loss", "mse", "mae"):
        raise ValueError("unknown utility metric")
    if utility < 0.0 or (metric == "accuracy" and utility > 1.0):
        raise ValueError("utility must be non-negative, and at most 1 for accuracy")
    return student


def _cross_check_student(
    student: dict[str, object],
    protocol: dict[str, object],
    split: dict[str, object],
    teacher: dict[str, object],
) -> None:
    if student["protocol_digest"] != hashlib.sha256(_canonical_bytes(protocol)).hexdigest():
        raise ValueError("the student names a different frozen protocol")
    if protocol["split_digest"] != hashlib.sha256(_canonical_bytes(split)).hexdigest():
        raise ValueError("the protocol binds a different split")
    names = ("model_digest", "compressor_digest")
    if [student[name] for name in names] != [protocol[name] for name in names]:
        raise ValueError("the student names another model or compressor than the protocol")
    if student["teacher_digest"] != hashlib.sha256(_canonical_bytes(teacher)).hexdigest():
        raise ValueError("the student names a different teacher feature batch")
    header = teacher["header"]
    origin = header["data_origin"] if type(header) is dict else None
    if _TEACHER_ORIGINS[str(student["teacher_kind"])] != origin:
        raise ValueError("teacher kind does not match the teacher's origin; no relabelling")
    targets, columns = student["targets"], teacher["observable_ids"]
    if type(targets) is not list or type(columns) is not list:
        raise ValueError("targets and observables must be lists")
    if targets != [name for name in columns if name in targets]:
        raise ValueError("student targets must be teacher observables, in teacher order")
    train = split["train_groups"]
    groups = student["training_groups"]
    if type(train) is not list or type(groups) is not list or not set(groups) <= set(train):
        raise ValueError("a student trains on train groups only")
    if student["utility_metric"] != protocol["primary_metric"]:
        raise ValueError("student utility must use the protocol's primary metric")


def _roundtrip_student_artifact(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "task", "split", "protocol", "teacher", "student"}:
        raise ValueError("student request fields mismatch")
    _roundtrip_task_split(
        {"op": "roundtrip_task_split", "task": request["task"], "split": request["split"]}
    )
    _roundtrip_experiment_protocol(
        {"op": "roundtrip_experiment_protocol", "protocol": request["protocol"]}
    )
    protocol, split = request["protocol"], request["split"]
    if type(protocol) is not dict or type(split) is not dict:
        raise ValueError("protocol and split must be objects")
    teacher = _validate_feature_fields(request["teacher"])
    teacher_header = teacher["header"]
    teacher_origin = str(teacher_header["data_origin"]) if type(teacher_header) is dict else ""
    if teacher_origin not in ("provider_result", "synthetic_classical"):
        raise ValueError("a feature batch derives from provider results or a fixture")
    _validate_artifact_header(teacher, (str(teacher["raw_result_set_digest"]),), teacher_origin)
    student = _validate_student_fields(request["student"])
    origin = _TEACHER_ORIGINS[str(student["teacher_kind"])]
    parents = (str(student["protocol_digest"]), str(student["teacher_digest"]))
    _validate_artifact_header(student, parents, origin)
    _cross_check_student(student, protocol, split, teacher)
    wire = _canonical_bytes(student)
    return {
        "schema": _SCHEMA,
        "status": "validated_roundtrip_no_compute",
        "student": json.loads(wire),
        "student_sha256": hashlib.sha256(wire).hexdigest(),
        "hardware_submission_enabled": False,
    }


_ROUNDTRIPS: dict[str, tuple[Any, tuple[type[Exception], ...]]] = {
    "roundtrip_contract": (_roundtrip_contract, (ValueError, TypeError, KeyError)),
    "roundtrip_task_split": (_roundtrip_task_split, (ValueError, TypeError, KeyError)),
    "roundtrip_model_descriptor": (_roundtrip_model_descriptor, (ValueError, TypeError, KeyError)),
    "roundtrip_latent_batch": (_roundtrip_latent_batch, (ValueError, TypeError, KeyError)),
    "roundtrip_compression": (
        _roundtrip_compression,
        (ValueError, TypeError, KeyError, OverflowError),
    ),
    "roundtrip_experiment_protocol": (
        _roundtrip_experiment_protocol,
        (ValueError, TypeError, KeyError),
    ),
    "roundtrip_target_snapshot": (_roundtrip_target_snapshot, (ValueError, TypeError, KeyError)),
    "roundtrip_compiled_bundle": (_roundtrip_compiled_bundle, (ValueError, TypeError, KeyError)),
    "roundtrip_cost_quote": (_roundtrip_cost_quote, (ValueError, TypeError, KeyError)),
    "roundtrip_owner_approval": (_roundtrip_owner_approval, (ValueError, TypeError, KeyError)),
    "roundtrip_provider_batch_receipt": (
        _roundtrip_provider_batch_receipt,
        (ValueError, TypeError, KeyError),
    ),
    "roundtrip_raw_result_set": (_roundtrip_raw_result_set, (ValueError, TypeError, KeyError)),
    "roundtrip_feature_batch": (_roundtrip_feature_batch, (ValueError, TypeError, KeyError)),
    "roundtrip_evaluation_report": (
        _roundtrip_evaluation_report,
        (ValueError, TypeError, KeyError),
    ),
    "roundtrip_student_artifact": (_roundtrip_student_artifact, (ValueError, TypeError, KeyError)),
}


def main() -> int:
    """Read one bounded request and refuse any operation except discovery.

    Returns
    -------
    int
        Zero for a boundary description or two for a refused request.

    """
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "request too large"})
        return 2

    def unique_fields(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

    try:
        request = json.loads(raw, object_pairs_hook=unique_fields)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "invalid JSON"})
        return 2
    op = request.get("op") if type(request) is dict else None
    if type(op) is str and op in _ROUNDTRIPS:
        handler, refusals = _ROUNDTRIPS[op]
        try:
            _emit(handler(request))
        except refusals as exc:
            _emit({"schema": _SCHEMA, "status": "refused", "reason": str(exc)})
            return 2
        return 0
    if type(request) is not dict or set(request) != {"op"} or request["op"] != "describe":
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "unsupported operation"})
        return 2
    _emit(
        {
            "schema": _SCHEMA,
            "status": "experimental_no_compute",
            "supported_operations": ["describe", *_ROUNDTRIPS],
            "hardware_submission_enabled": False,
            "provider_credentials_required": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
