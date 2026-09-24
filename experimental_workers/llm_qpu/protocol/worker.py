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
    if type(request) is not dict or set(request) != {"op"} or request["op"] != "describe":
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "unsupported operation"})
        return 2
    _emit(
        {
            "schema": _SCHEMA,
            "status": "experimental_no_compute",
            "supported_operations": ["describe", "roundtrip_contract"],
            "hardware_submission_enabled": False,
            "provider_credentials_required": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
