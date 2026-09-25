# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — LLM-QPU private shard lineage
"""Immutable latent-shard manifests with explicit incomplete-evidence states."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ArrayDescriptor, canonical_bytes
from ..contracts.wire import _digest, _strict_json, _text
from .artifact_store import ArtifactStore, MissingArtifactError

_SCHEMA = "scpn.experimental.llm_qpu.shard_manifest.v1"


@dataclass(frozen=True, slots=True)
class ShardRef:
    """One contiguous raw array shard and its row interval."""

    descriptor: ArrayDescriptor
    row_start: int
    row_count: int

    def __post_init__(self) -> None:
        """Keep row shape and interval exact before any array allocation."""
        if type(self.descriptor) is not ArrayDescriptor:
            raise ValueError("shard requires a bounded array descriptor")
        if type(self.row_start) is not int or not 0 <= self.row_start <= 1_000_000:
            raise ValueError("shard row start is invalid")
        if (
            type(self.row_count) is not int
            or not 0 < self.row_count <= 1_000_000
            or self.descriptor.shape[0] != self.row_count
        ):
            raise ValueError("shard row count differs from array shape")

    def to_wire(self) -> dict[str, object]:
        """Return descriptor and exact row interval."""
        return {
            "descriptor": self.descriptor.to_wire(),
            "row_start": self.row_start,
            "row_count": self.row_count,
        }

    @classmethod
    def from_wire(cls, value: object) -> ShardRef:
        """Reject missing or unreviewed shard fields."""
        if type(value) is not dict or set(value) != {"descriptor", "row_start", "row_count"}:
            raise ValueError("shard reference fields mismatch")
        return cls(
            descriptor=ArrayDescriptor.from_wire(value["descriptor"]),
            row_start=value["row_start"],
            row_count=value["row_count"],
        )


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """Exact private latent or compressed-angle shard set and source identity."""

    artifact_kind: str
    model_digest: str
    tokenizer_digest: str
    tap_id: str
    context_mode: str
    compressor_digest: str | None
    sample_ids_digest: str
    group_ids_digest: str
    shards: tuple[ShardRef, ...]
    row_count: int
    data_origin: str
    egress: str

    def __post_init__(self) -> None:
        """Refuse gaps, overlaps, mixed modalities, and public latent egress."""
        if self.artifact_kind not in ("latent_batch", "compressed_latent_batch"):
            raise ValueError("unsupported shard manifest kind")
        for name in ("model_digest", "tokenizer_digest", "sample_ids_digest", "group_ids_digest"):
            _digest(getattr(self, name), name=name)
        _text(self.tap_id, name="tap ID")
        if self.context_mode not in ("contextual", "chunk_isolated"):
            raise ValueError("unsupported latent context mode")
        if self.artifact_kind == "compressed_latent_batch":
            _digest(self.compressor_digest, name="compressor")
        elif self.compressor_digest is not None:
            raise ValueError("uncompressed latents cannot claim a compressor")
        if type(self.shards) is not tuple or not 1 <= len(self.shards) <= 1024:
            raise ValueError("shard inventory out of bounds")
        if type(self.shards[0]) is not ShardRef:
            raise ValueError("manifest shard must be a ShardRef")
        cursor = 0
        dtype = self.shards[0].descriptor.dtype
        trailing_shape = self.shards[0].descriptor.shape[1:]
        for shard in self.shards:
            if type(shard) is not ShardRef or shard.row_start != cursor:
                raise ValueError("shard row intervals must be contiguous and ordered")
            if shard.descriptor.dtype != dtype or shard.descriptor.shape[1:] != trailing_shape:
                raise ValueError("shard dtype or trailing shape drift")
            cursor += shard.row_count
        if type(self.row_count) is not int or cursor != self.row_count:
            raise ValueError("manifest row count mismatch")
        if self.data_origin not in ("synthetic_classical", "owner_dataset", "external_dataset"):
            raise ValueError("unknown manifest data origin")
        if self.egress != "private_derived":
            raise ValueError("derived latent shards must remain private")

    def to_wire(self) -> dict[str, object]:
        """Return deterministic, value-free shard references."""
        return {
            "schema": _SCHEMA,
            "object_kind": "shard_manifest",
            "artifact_kind": self.artifact_kind,
            "model_digest": self.model_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "tap_id": self.tap_id,
            "context_mode": self.context_mode,
            "compressor_digest": self.compressor_digest,
            "sample_ids_digest": self.sample_ids_digest,
            "group_ids_digest": self.group_ids_digest,
            "shards": [shard.to_wire() for shard in self.shards],
            "row_count": self.row_count,
            "data_origin": self.data_origin,
            "egress": self.egress,
        }

    @classmethod
    def from_wire(cls, value: object) -> ShardManifest:
        """Decode only the reviewed manifest field inventory."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("shard manifest fields mismatch")
        if value["schema"] != _SCHEMA or value["object_kind"] != "shard_manifest":
            raise ValueError("unsupported shard manifest schema")
        if type(value["shards"]) is not list:
            raise ValueError("manifest shards must be a list")
        return cls(
            artifact_kind=value["artifact_kind"],
            model_digest=value["model_digest"],
            tokenizer_digest=value["tokenizer_digest"],
            tap_id=value["tap_id"],
            context_mode=value["context_mode"],
            compressor_digest=value["compressor_digest"],
            sample_ids_digest=value["sample_ids_digest"],
            group_ids_digest=value["group_ids_digest"],
            shards=tuple(ShardRef.from_wire(shard) for shard in value["shards"]),
            row_count=value["row_count"],
            data_origin=value["data_origin"],
            egress=value["egress"],
        )


def publish_manifest(store: ArtifactStore, manifest: ShardManifest) -> str:
    """Admit only a complete verified shard set to the durable index."""
    if type(store) is not ArtifactStore or type(manifest) is not ShardManifest:
        raise ValueError("manifest publication requires reviewed store and manifest")
    digests: list[str] = []
    for shard in manifest.shards:
        record, payload = store.get(shard.descriptor.sha256)
        if record.kind != "array_shard" or record.egress != "private_derived":
            raise ValueError("manifest shard custody label mismatch")
        shard.descriptor.validate_payload(payload)
        digests.append(record.sha256)
    encoded = canonical_bytes(manifest.to_wire())
    record = store.put(encoded, kind="manifest", retention="pinned", egress="private_derived")
    with store._connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "INSERT OR IGNORE INTO manifest_shards VALUES (?, ?)",
            ((record.sha256, digest) for digest in digests),
        )
        connection.execute("INSERT OR IGNORE INTO admitted_manifests VALUES (?)", (record.sha256,))
    return record.sha256


def load_manifest(store: ArtifactStore, digest: str) -> ShardManifest:
    """Return only admitted manifests whose exact shards remain readable."""
    if type(store) is not ArtifactStore:
        raise ValueError("manifest loading requires an ArtifactStore")
    with store._connection() as connection:
        admitted = connection.execute(
            "SELECT 1 FROM admitted_manifests WHERE sha256=?", (digest,)
        ).fetchone()
    if admitted is None:
        raise MissingArtifactError(f"manifest not admitted: {digest}")
    record, encoded = store.get(digest)
    if record.kind != "manifest" or record.egress != "private_derived":
        raise ValueError("manifest custody label mismatch")
    manifest = ShardManifest.from_wire(_strict_json(encoded))
    with store._connection() as connection:
        indexed = {
            row[0]
            for row in connection.execute(
                "SELECT shard_sha256 FROM manifest_shards WHERE manifest_sha256=?", (digest,)
            )
        }
    expected = {shard.descriptor.sha256 for shard in manifest.shards}
    if indexed != expected:
        raise MissingArtifactError("manifest shard index incomplete")
    for shard in manifest.shards:
        shard_record, payload = store.get(shard.descriptor.sha256)
        if shard_record.kind != "array_shard" or shard_record.egress != "private_derived":
            raise ValueError("manifest shard custody label mismatch")
        shard.descriptor.validate_payload(payload)
    return manifest
