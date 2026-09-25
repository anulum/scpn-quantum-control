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
import re
import struct
import sys
import unicodedata

_MAX_REQUEST_BYTES = 65_536
_SCHEMA = "scpn.experimental.llm_qpu.worker_boundary.v1"
_CELL_SCHEMA = "scpn.experimental.llm_qpu.planned_cell.v1"
_ARRAY_SCHEMA = "scpn.experimental.llm_qpu.array_descriptor.v1"
_TASK_SCHEMA = "scpn.experimental.llm_qpu.task_spec.v2"
_SPLIT_SCHEMA = "scpn.experimental.llm_qpu.split_manifest.v2"
_HEADER_SCHEMA = "scpn.experimental.llm_qpu.artifact_header.v1"
_MODEL_SCHEMA = "scpn.experimental.llm_qpu.model_descriptor.v1"
_LATENT_SCHEMA = "scpn.experimental.llm_qpu.latent_batch.v1"
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


def _roundtrip_contract(request: dict[str, object]) -> dict[str, object]:
    if set(request) != {"op", "cell", "array", "array_base64"}:
        raise ValueError("contract request fields mismatch")
    cell = request["cell"]
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
    cell_wire = _canonical_bytes(cell)
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
    if task["primary_metric"] not in ("accuracy", "balanced_accuracy", "f1", "mse", "mae"):
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
    if model["quantization"] not in ("none", "gguf_q6_k", "gguf_q4_k_m"):
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
    task = request["task"]
    split = request["split"]
    model = request["model"]
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
        if layout == "contextual" and length != 1:
            raise ValueError("contextual latent length must be one")
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
    if type(request) is dict and request.get("op") == "roundtrip_contract":
        try:
            _emit(_roundtrip_contract(request))
        except (ValueError, TypeError, KeyError) as exc:
            _emit({"schema": _SCHEMA, "status": "refused", "reason": str(exc)})
            return 2
        return 0
    if type(request) is dict and request.get("op") == "roundtrip_task_split":
        try:
            _emit(_roundtrip_task_split(request))
        except (ValueError, TypeError, KeyError) as exc:
            _emit({"schema": _SCHEMA, "status": "refused", "reason": str(exc)})
            return 2
        return 0
    if type(request) is dict and request.get("op") == "roundtrip_model_descriptor":
        try:
            _emit(_roundtrip_model_descriptor(request))
        except (ValueError, TypeError, KeyError) as exc:
            _emit({"schema": _SCHEMA, "status": "refused", "reason": str(exc)})
            return 2
        return 0
    if type(request) is dict and request.get("op") == "roundtrip_latent_batch":
        try:
            _emit(_roundtrip_latent_batch(request))
        except (ValueError, TypeError, KeyError) as exc:
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
            "supported_operations": [
                "describe",
                "roundtrip_contract",
                "roundtrip_task_split",
                "roundtrip_model_descriptor",
                "roundtrip_latent_batch",
            ],
            "hardware_submission_enabled": False,
            "provider_credentials_required": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
