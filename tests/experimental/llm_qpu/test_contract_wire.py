# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU canonical wire refusals
"""Refuse every malformed wire value, array descriptor and provenance header."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import CellKey, decode_contract, wire
from scpn_quantum_control.experimental.llm_qpu.contracts.wire import (
    ARRAY_SCHEMA,
    HEADER_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    canonical_bytes,
    f64,
)

_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _header(**changes: Any) -> ArtifactHeader:
    """Build a valid design header, then apply the requested field changes."""
    fields: dict[str, Any] = {
        "object_kind": "task_spec",
        "content_digest": _DIGEST_A,
        "parents": (),
        "base_repo_commit": _BASE,
        "implementation_revision": _REVISION,
        "execution_origin": "offline_design",
        "data_origin": "synthetic_classical",
        "claim_scope": "design_only",
    }
    fields.update(changes)
    return ArtifactHeader(**fields)


def _nested(depth: int) -> object:
    """Return a list nested exactly ``depth`` levels below the root."""
    value: object = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_nested(17), "wire nesting exceeds limit"),
        (2**63, "wire integer exceeds signed 64-bit range"),
        (-(2**63) - 1, "wire integer exceeds signed 64-bit range"),
        ("", "wire string must be bounded nonempty text"),
        pytest.param("x" * 257, "wire string must be bounded nonempty text", id="long-text"),
        ("tab\there", "wire string must be bounded nonempty text"),
        (1.5, "raw JSON floats are forbidden; use f64"),
        (list(range(4097)), "wire list exceeds item limit"),
        ({f"k{index}": index for index in range(4097)}, "wire object exceeds item limit"),
        ({"$f64": 1}, "f64 tag must contain hex text"),
        ({"$f64": "not-hex"}, "invalid f64 hex"),
        ({"$f64": "0x1p+0"}, "noncanonical or nonfinite f64 hex"),
        ({"$f64": "inf"}, "noncanonical or nonfinite f64 hex"),
        ({"1st": 1}, "wire field names must be ASCII identifiers"),
        ({7: 1}, "wire field names must be ASCII identifiers"),
        (b"bytes", "unsupported wire value"),
        ({1, 2}, "unsupported wire value"),
    ],
)
def test_canonical_bytes_refuses_values_outside_the_wire(value: object, message: str) -> None:
    """Encoding refuses each unbounded, lossy or untyped value with its reason."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        canonical_bytes(value)


def test_canonical_bytes_keeps_the_admitted_limits_and_canonical_f64() -> None:
    """The deepest list, extreme integers and canonical hex floats still encode."""
    assert canonical_bytes(_nested(16)) == b"[" * 16 + b"[]" + b"]" * 16
    assert canonical_bytes([2**63 - 1, -(2**63), None, True]) == (
        b"[9223372036854775807,-9223372036854775808,null,true]"
    )
    token = f64(1.0)["$f64"]
    assert canonical_bytes({"x": {"$f64": token}}) == b'{"x":{"$f64":"0x1.0000000000000p+0"}}'
    assert f64(-0.0) == {"$f64": "0x0.0p+0"}


@pytest.mark.parametrize("value", [1, True, float("inf"), float("nan"), "1.0"])
def test_f64_refuses_non_floats_and_nonfinite_values(value: Any) -> None:
    """Only a finite value of exact type float can be tagged as scientific f64."""
    with pytest.raises(ValueError, match="^scientific f64 must be finite and exactly float$"):
        f64(value)


def test_canonical_bytes_refuses_a_wire_above_one_mebibyte() -> None:
    """A wire of admitted items still refuses once its bytes exceed the ceiling."""
    with pytest.raises(ValueError, match="^canonical wire exceeds byte limit$"):
        canonical_bytes(["x" * 256] * 4096)


def _key(**changes: Any) -> CellKey:
    """Build a valid cell identity, then apply the requested field changes."""
    fields: dict[str, Any] = {
        "experiment_id": "exp",
        "source_sample_id": "sample",
        "sequence_id": "seq",
        "arm_id": "arm",
        "basis_id": "z",
        "replicate_id": "r0",
        "planned_epoch_id": "epoch",
        "parameter_binding_id": "binding",
    }
    fields.update(changes)
    return CellKey(**fields)


def test_contract_text_fields_refuse_non_text_and_normalise_to_nfc() -> None:
    """A contract identity refuses a non-text field and stores text in NFC form."""
    with pytest.raises(ValueError, match="^experiment_id must be text$"):
        _key(experiment_id=7)
    with pytest.raises(ValueError, match="^arm_id must be bounded nonempty text$"):
        _key(arm_id="")
    assert _key(arm_id="e\u0301").arm_id == "\u00e9"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{}", "wire must be bounded bytes"),
        pytest.param(b" " * 1_048_577, "wire must be bounded bytes", id="oversize-bytes"),
        (b'{"a":1,"a":2}', "duplicate JSON field"),
        (b"[NaN]", "nonfinite JSON"),
        (b"\xff", "invalid UTF-8 JSON"),
        (b"{", "invalid UTF-8 JSON"),
        (b'{"b":1, "a":2}', "noncanonical JSON wire"),
        (b"[1]", "contract wire must be an object"),
        (b'{"schema":"unknown"}', "unsupported contract schema"),
    ],
)
def test_decode_contract_refuses_ambiguous_or_noncanonical_bytes(raw: Any, message: str) -> None:
    """Decoding refuses text, oversize, duplicate, nonfinite, broken and reordered wires."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        decode_contract(raw)


def test_decode_contract_returns_the_record_its_canonical_wire_encodes() -> None:
    """A descriptor's canonical bytes decode back to an equal descriptor."""
    descriptor = ArrayDescriptor("<i4", (4,), 16, _DIGEST_B)
    raw = canonical_bytes(descriptor.to_wire())
    assert decode_contract(raw) == descriptor
    with pytest.raises(ValueError, match="^noncanonical JSON wire$"):
        decode_contract(raw.replace(b",", b", ", 1))


@pytest.mark.parametrize(
    ("dtype", "shape", "byte_length", "digest", "message"),
    [
        ("<f2", (1,), 2, _DIGEST_A, "unsupported array dtype or shape"),
        (8, (1,), 8, _DIGEST_A, "unsupported array dtype or shape"),
        ("<f8", [1], 8, _DIGEST_A, "unsupported array dtype or shape"),
        ("<f8", (), 0, _DIGEST_A, "array rank out of bounds"),
        ("<f8", (1, 1, 1, 1, 1), 8, _DIGEST_A, "array rank out of bounds"),
        ("<f8", (0,), 0, _DIGEST_A, "array axis must be a bounded positive integer"),
        ("<f8", (1000, 1001), 8_008_000, _DIGEST_A, "array element limit exceeded"),
        ("<f8", (2,), 8, _DIGEST_A, "array byte length mismatch"),
        ("<f8", (2,), 16.0, _DIGEST_A, "array byte length mismatch"),
        ("<f8", (2,), 16, "A" * 64, "array payload must be a lowercase SHA-256 digest"),
    ],
)
def test_array_descriptor_refuses_unsupported_layouts(
    dtype: Any, shape: Any, byte_length: Any, digest: Any, message: str
) -> None:
    """Each malformed dtype, rank, axis, size or digest is refused at construction."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        ArrayDescriptor(dtype, shape, byte_length, digest)


def test_array_element_ceiling_keeps_every_payload_within_sixteen_mebibytes() -> None:
    """The element limit binds first, so no admitted dtype can pass 16 MiB."""
    assert wire._MAX_ARRAY_ELEMENTS * max(wire._DTYPE_SIZE.values()) <= 16 * 1024 * 1024
    largest = ArrayDescriptor("<f8", (1000, 1000), 8_000_000, _DIGEST_A)
    assert largest.byte_length == 8_000_000


def test_array_payload_validation_checks_bytes_digest_and_finiteness() -> None:
    """Payloads must match length and digest, and float payloads must be finite."""
    finite = struct.pack("<2d", 1.0, -2.5)
    descriptor = ArrayDescriptor("<f8", (2,), 16, hashlib.sha256(finite).hexdigest())
    descriptor.validate_payload(finite)
    with pytest.raises(ValueError, match="^array payload byte length mismatch$"):
        descriptor.validate_payload(finite[:8])
    mutable: Any = bytearray(finite)
    with pytest.raises(ValueError, match="^array payload byte length mismatch$"):
        descriptor.validate_payload(mutable)
    with pytest.raises(ValueError, match="^array payload digest mismatch$"):
        descriptor.validate_payload(struct.pack("<2d", 1.0, 2.5))
    nonfinite = struct.pack("<2f", 1.0, float("nan"))
    single = ArrayDescriptor("<f4", (2,), 8, hashlib.sha256(nonfinite).hexdigest())
    with pytest.raises(ValueError, match="^array contains nonfinite float$"):
        single.validate_payload(nonfinite)
    integers = struct.pack("<2q", -1, 2**62)
    ArrayDescriptor("<i8", (2,), 16, hashlib.sha256(integers).hexdigest()).validate_payload(
        integers
    )


def test_array_descriptor_wire_roundtrip_and_refusals() -> None:
    """The descriptor round-trips and refuses fields, schema, kind and types."""
    descriptor = ArrayDescriptor("<u4", (3, 2), 24, _DIGEST_A)
    encoded = descriptor.to_wire()
    assert ArrayDescriptor.from_wire(encoded) == descriptor
    cases: list[tuple[object, str]] = [
        ([], "array descriptor fields mismatch"),
        ({**encoded, "extra": 1}, "array descriptor fields mismatch"),
        ({**encoded, "schema": ARRAY_SCHEMA + "x"}, "unknown array descriptor schema"),
        ({**encoded, "object_kind": "latent_batch"}, "unknown array descriptor schema"),
        ({**encoded, "dtype": 4}, "array descriptor dtype and digest must be text"),
        ({**encoded, "sha256": None}, "array descriptor dtype and digest must be text"),
        ({**encoded, "shape": (3, 2)}, "array shape must be a list"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            ArrayDescriptor.from_wire(value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"object_kind": "reservoir_spec"}, "unsupported artifact object kind"),
        ({"content_digest": "z" * 64}, "artifact content must be a lowercase SHA-256 digest"),
        ({"parents": [_DIGEST_A]}, "artifact parents must be a bounded tuple"),
        ({"parents": tuple(f"{i:064x}" for i in range(33))}, "artifact parents must be"),
        ({"parents": ("bad",)}, "artifact parent must be a lowercase SHA-256 digest"),
        ({"parents": (_DIGEST_B, _DIGEST_A)}, "artifact parents must be sorted and unique"),
        ({"parents": (_DIGEST_A, _DIGEST_A)}, "artifact parents must be sorted and unique"),
        ({"base_repo_commit": _BASE[:39]}, "base_repo_commit must be a full lowercase"),
        ({"implementation_revision": 1}, "implementation_revision must be a full lowercase"),
        ({"implementation_revision": _BASE}, "implementation revision must differ"),
        ({"execution_origin": "provider_run"}, "design artifact cannot claim a live"),
        ({"data_origin": "web_scrape"}, "unknown artifact data origin"),
        ({"claim_scope": "confirmed"}, "design artifact cannot claim confirmation"),
    ],
)
def test_artifact_header_refuses_unbound_provenance(changes: dict[str, Any], message: str) -> None:
    """A header refuses unknown kinds, bad digests, lineage, commits and claims."""
    with pytest.raises(ValueError, match=f"^{message}"):
        _header(**changes)


def test_artifact_header_wire_roundtrip_and_refusals() -> None:
    """The header round-trips and refuses fields, schema, lane and parent shape."""
    header = _header(parents=(_DIGEST_A, _DIGEST_B), data_origin="owner_dataset")
    encoded = header.to_wire()
    assert encoded["schema"] == HEADER_SCHEMA
    assert ArtifactHeader.from_wire(encoded) == header
    cases: list[tuple[object, str]] = [
        ("header", "ArtifactHeader fields mismatch"),
        ({**encoded, "extra": 1}, "ArtifactHeader fields mismatch"),
        ({**encoded, "schema": "v0"}, "unknown ArtifactHeader schema or lane"),
        ({**encoded, "lane_id": "other"}, "unknown ArtifactHeader schema or lane"),
        ({**encoded, "parents": (_DIGEST_A,)}, "artifact parents must be a list on wire"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}$"):
            ArtifactHeader.from_wire(value)


def test_artifact_header_binds_exact_content_and_lineage() -> None:
    """Content must be a dict with the declared digest and the declared parents."""
    content: dict[str, object] = {"task": "temporal"}
    digest = hashlib.sha256(canonical_bytes(content)).hexdigest()
    header = _header(content_digest=digest, parents=(_DIGEST_A, _DIGEST_B))
    header.validate_content(content, parents=(_DIGEST_B, _DIGEST_A))
    with pytest.raises(ValueError, match="^artifact content or parent lineage mismatch$"):
        header.validate_content(content, parents=(_DIGEST_A,))
    listed: Any = [content]
    with pytest.raises(ValueError, match="^artifact content or parent lineage mismatch$"):
        header.validate_content(listed, parents=(_DIGEST_A, _DIGEST_B))
    with pytest.raises(ValueError, match="^artifact content digest mismatch$"):
        header.validate_content({"task": "other"}, parents=(_DIGEST_A, _DIGEST_B))
