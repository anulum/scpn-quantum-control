# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — manifest and cache acceptance
"""Exercise real manifest admission and revision-specific cache lookup."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import ArrayDescriptor
from scpn_quantum_control.experimental.llm_qpu.data.artifact_store import (
    ArtifactStore,
    MissingArtifactError,
)
from scpn_quantum_control.experimental.llm_qpu.data.cache import (
    ArtifactCache,
    CacheKey,
    StaleCacheError,
)
from scpn_quantum_control.experimental.llm_qpu.data.lineage import (
    ShardManifest,
    ShardRef,
    load_manifest,
    publish_manifest,
)


def _manifest(store: ArtifactStore, *, compressed: bool = False) -> tuple[str, ArrayDescriptor]:
    """Store one two-row shard and publish its latent or compressed manifest."""
    payload = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    descriptor = ArrayDescriptor("<f4", (2, 2), len(payload), hashlib.sha256(payload).hexdigest())
    store.put_array(descriptor, payload)
    manifest = ShardManifest(
        artifact_kind="compressed_latent_batch" if compressed else "latent_batch",
        model_digest="a" * 64,
        tokenizer_digest="b" * 64,
        tap_id="block.1.post_attention",
        context_mode="contextual",
        compressor_digest="c" * 64 if compressed else None,
        sample_ids_digest="d" * 64,
        group_ids_digest="e" * 64,
        shards=(ShardRef(descriptor, 0, 2),),
        row_count=2,
        data_origin="synthetic_classical",
        egress="private_derived",
    )
    return publish_manifest(store, manifest), descriptor


def _key(stage: str) -> CacheKey:
    """Return the cache key of one pipeline stage with its revision inputs."""
    return CacheKey(
        source_digest="f" * 64,
        stage=stage,
        model_digest="a" * 64,
        tokenizer_digest="b" * 64,
        tap_id="block.1.post_attention",
        context_mode="contextual",
        compressor_digest="c" * 64 if stage != "latent" else None,
        sdk_lock_digest="d" * 64 if stage in ("compiled", "features") else None,
        circuit_map_digest="e" * 64 if stage in ("compiled", "features") else None,
    )


def test_t03e_missing_shard_is_incomplete_evidence_after_reopen(tmp_path: Path) -> None:
    """An admitted manifest remains valid only while every shard is readable."""
    root = tmp_path / "private"
    store = ArtifactStore(root)
    manifest_digest, descriptor = _manifest(store)
    assert load_manifest(ArtifactStore(root), manifest_digest).row_count == 2

    (store.objects / descriptor.sha256[:2] / descriptor.sha256).unlink()
    with pytest.raises(MissingArtifactError):
        load_manifest(ArtifactStore(root), manifest_digest)
    with pytest.raises(MissingArtifactError):
        ArtifactCache(ArtifactStore(root)).bind(_key("latent"), manifest_digest)


def test_t03c_model_compressor_and_sdk_changes_refuse_stale_cache(tmp_path: Path) -> None:
    """Each revision input participates in exact cache identity."""
    store = ArtifactStore(tmp_path / "private")
    cache = ArtifactCache(store)
    latent_digest, _ = _manifest(store)
    latent_key = _key("latent")
    cache.bind(latent_key, latent_digest)
    assert cache.lookup(latent_key) == latent_digest
    with pytest.raises(ValueError, match="disagrees"):
        cache.bind(replace(latent_key, model_digest="1" * 64), latent_digest)
    with pytest.raises(StaleCacheError):
        cache.lookup(replace(latent_key, model_digest="1" * 64))

    compressed_digest, _ = _manifest(store, compressed=True)
    compressed_key = _key("compressed")
    cache.bind(compressed_key, compressed_digest)
    assert cache.lookup(compressed_key) == compressed_digest
    with pytest.raises(StaleCacheError):
        cache.lookup(replace(compressed_key, compressor_digest="2" * 64))

    evidence = store.put(
        b"exact compiled circuit evidence",
        kind="evidence",
        retention="pinned",
        egress="private_derived",
    )
    compiled_key = _key("compiled")
    cache.bind(compiled_key, evidence.sha256)
    assert cache.lookup(compiled_key) == evidence.sha256
    with pytest.raises(StaleCacheError):
        cache.lookup(replace(compiled_key, sdk_lock_digest="3" * 64))


def test_manifest_rejects_noncontiguous_or_public_shards(tmp_path: Path) -> None:
    """A lineage gap or public derived-data label never reaches the index."""
    store = ArtifactStore(tmp_path / "private")
    digest, descriptor = _manifest(store)
    admitted = load_manifest(store, digest)
    with pytest.raises(ValueError, match="contiguous"):
        replace(admitted, shards=(ShardRef(descriptor, 1, 2),))
    with pytest.raises(ValueError, match="private"):
        replace(admitted, egress="value_free")


def _descriptor(shape: tuple[int, ...], dtype: str = "<f4") -> ArrayDescriptor:
    """Return a descriptor of the given shape over zero-valued bytes."""
    size = 4 if dtype == "<f4" else 8
    length = size
    for axis in shape:
        length *= axis
    return ArrayDescriptor(dtype, shape, length, hashlib.sha256(bytes(length)).hexdigest())


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (("descriptor", 0, 2), "shard requires a bounded array descriptor"),
        ((None, -1, 2), "shard row start is invalid"),
        ((None, 1_000_001, 2), "shard row start is invalid"),
        ((None, True, 2), "shard row start is invalid"),
        ((None, 0, 0), "shard row count differs from array shape"),
        ((None, 0, 3), "shard row count differs from array shape"),
        ((None, 0, 2.0), "shard row count differs from array shape"),
    ],
)
def test_shard_reference_refuses_bad_descriptors_and_intervals(
    fields: tuple[Any, int, Any], message: str
) -> None:
    """A shard needs a real descriptor, a bounded start and a count equal to its rows."""
    descriptor: Any = fields[0] if fields[0] is not None else _descriptor((2, 2))
    with pytest.raises(ValueError, match=f"^{message}$"):
        ShardRef(descriptor, fields[1], fields[2])


def test_shard_reference_wire_round_trip_and_refusals() -> None:
    """A shard reference round-trips and refuses any other field set."""
    reference = ShardRef(_descriptor((2, 2)), 0, 2)
    assert ShardRef.from_wire(reference.to_wire()) == reference
    refused: tuple[object, ...] = (
        [],
        {**reference.to_wire(), "extra": 1},
        {"descriptor": {}, "row_start": 0},
    )
    for value in refused:
        with pytest.raises(ValueError, match="^shard reference fields mismatch$"):
            ShardRef.from_wire(value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"artifact_kind": "raw_tokens"}, "unsupported shard manifest kind"),
        ({"tokenizer_digest": "B" * 64}, "tokenizer_digest must be a lowercase SHA-256 digest"),
        ({"tap_id": ""}, "tap ID must be bounded nonempty text"),
        ({"context_mode": "streaming"}, "unsupported latent context mode"),
        ({"compressor_digest": "c" * 64}, "uncompressed latents cannot claim a compressor"),
        ({"shards": ()}, "shard inventory out of bounds"),
        ({"shards": []}, "shard inventory out of bounds"),
        ({"shards": ("shard",)}, "manifest shard must be a ShardRef"),
        ({"row_count": 3}, "manifest row count mismatch"),
        ({"row_count": True}, "manifest row count mismatch"),
        ({"data_origin": "web_scrape"}, "unknown manifest data origin"),
    ],
)
def test_manifest_refuses_unbound_identity_and_inventory(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    """A manifest refuses unknown kinds, digests, modes, claims, inventories, counts and origins."""
    store = ArtifactStore(tmp_path / "private")
    admitted = load_manifest(store, _manifest(store)[0])
    with pytest.raises(ValueError, match=f"^{message}$"):
        replace(admitted, **changes)


def test_manifest_refuses_mixed_shards_and_a_compressed_batch_without_compressor(
    tmp_path: Path,
) -> None:
    """All shards share dtype and trailing shape, and a compressed batch names its compressor."""
    store = ArtifactStore(tmp_path / "private")
    admitted = load_manifest(store, _manifest(store)[0])
    first = admitted.shards[0]
    for other in (_descriptor((1, 3)), _descriptor((1, 2), "<f8")):
        with pytest.raises(ValueError, match="^shard dtype or trailing shape drift$"):
            replace(admitted, shards=(first, ShardRef(other, 2, 1)), row_count=3)
    with pytest.raises(ValueError, match="^compressor must be a lowercase SHA-256 digest$"):
        replace(admitted, artifact_kind="compressed_latent_batch")


def test_manifest_wire_round_trip_and_refusals(tmp_path: Path) -> None:
    """A manifest round-trips and refuses other fields, another schema and non-list shards."""
    store = ArtifactStore(tmp_path / "private")
    admitted = load_manifest(store, _manifest(store)[0])
    encoded = admitted.to_wire()
    assert ShardManifest.from_wire(encoded) == admitted
    cases: list[tuple[object, str]] = [
        ("manifest", "shard manifest fields mismatch"),
        ({**encoded, "extra": 1}, "shard manifest fields mismatch"),
        ({**encoded, "schema": "v0"}, "unsupported shard manifest schema"),
        ({**encoded, "object_kind": "latent_batch"}, "unsupported shard manifest schema"),
        ({**encoded, "shards": {"first": encoded["shards"]}}, "manifest shards must be a list"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}$"):
            ShardManifest.from_wire(value)


def test_publication_refuses_unreviewed_inputs_and_mislabelled_shards(tmp_path: Path) -> None:
    """Only a reviewed store and manifest publish, and only private raw shards are admitted."""
    store = ArtifactStore(tmp_path / "private")
    admitted = load_manifest(store, _manifest(store)[0])
    loose_store: Any = tmp_path / "private"
    loose_manifest: Any = admitted.to_wire()
    for bad_store, bad_manifest in ((loose_store, admitted), (store, loose_manifest)):
        with pytest.raises(ValueError, match="^manifest publication requires reviewed store"):
            publish_manifest(bad_store, bad_manifest)
    synthetic = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    evidence = b"\x00" * 15 + b"\x01"
    store.put_array(
        ArrayDescriptor("<f4", (2, 2), 16, hashlib.sha256(synthetic).hexdigest()),
        synthetic,
        egress="reviewed_synthetic",
    )
    store.put(evidence, kind="evidence", retention="pinned", egress="private_derived")
    for payload in (synthetic, evidence):
        reference = ShardRef(
            ArrayDescriptor("<f4", (2, 2), 16, hashlib.sha256(payload).hexdigest()), 0, 2
        )
        with pytest.raises(ValueError, match="^manifest shard custody label mismatch$"):
            publish_manifest(store, replace(admitted, shards=(reference,)))


def _execute(store: ArtifactStore, statement: str, parameters: tuple[str, ...]) -> None:
    """Apply one statement to the store index, as a damaged index would hold it."""
    with sqlite3.connect(store.database) as connection:
        connection.execute(statement, parameters)
    connection.close()


def test_loading_refuses_unadmitted_and_damaged_index_state(tmp_path: Path) -> None:
    """Loading needs a store, an admitted manifest record, a full shard index and raw shards."""
    store = ArtifactStore(tmp_path / "private")
    digest, descriptor = _manifest(store)
    loose_store: Any = tmp_path / "private"
    with pytest.raises(ValueError, match="^manifest loading requires an ArtifactStore$"):
        load_manifest(loose_store, digest)
    with pytest.raises(MissingArtifactError, match="manifest not admitted"):
        load_manifest(store, "0" * 64)
    other = store.put(
        b"evidence bytes", kind="evidence", retention="pinned", egress="private_derived"
    )
    _execute(store, "INSERT INTO admitted_manifests VALUES (?)", (other.sha256,))
    with pytest.raises(ValueError, match="^manifest custody label mismatch$"):
        load_manifest(store, other.sha256)
    _execute(store, "DELETE FROM manifest_shards WHERE manifest_sha256=?", (digest,))
    with pytest.raises(MissingArtifactError, match="manifest shard index incomplete"):
        load_manifest(store, digest)
    _execute(store, "INSERT INTO manifest_shards VALUES (?, ?)", (digest, descriptor.sha256))
    assert load_manifest(store, digest).row_count == 2
    _execute(store, "UPDATE objects SET kind='evidence' WHERE sha256=?", (descriptor.sha256,))
    with pytest.raises(ValueError, match="^manifest shard custody label mismatch$"):
        load_manifest(store, digest)


@pytest.mark.parametrize(
    ("stage", "changes", "message"),
    [
        ("latent", {"stage": "raw_tokens"}, "unsupported cache stage"),
        ("latent", {"context_mode": "streaming"}, "unsupported cache context mode"),
        ("latent", {"compressor_digest": "c" * 64}, "latent cache cannot claim downstream"),
        ("latent", {"sdk_lock_digest": "d" * 64}, "latent cache cannot claim downstream"),
        ("compressed", {"compressor_digest": None}, "derived cache requires a compressor digest"),
        ("compiled", {"sdk_lock_digest": None}, "provider-derived cache requires SDK and circuit"),
        ("features", {"circuit_map_digest": None}, "provider-derived cache requires SDK"),
    ],
)
def test_cache_key_refuses_unversioned_identities(
    stage: str, changes: dict[str, Any], message: str
) -> None:
    """A key refuses unknown stages and modes and any revision set its stage cannot carry."""
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(_key(stage), **changes)


def test_cache_refuses_unreviewed_inputs_public_objects_and_rebinding(tmp_path: Path) -> None:
    """The cache needs the private store and exact keys, and never rebinds a key."""
    loose_store: Any = tmp_path / "private"
    with pytest.raises(ValueError, match="^cache requires the private artifact store$"):
        ArtifactCache(loose_store)
    store = ArtifactStore(tmp_path / "private")
    cache = ArtifactCache(store)
    key = _key("compiled")
    loose_key: Any = key.digest
    first = store.put(
        b"first evidence", kind="evidence", retention="pinned", egress="private_derived"
    )
    with pytest.raises(ValueError, match="^cache bind requires a CacheKey$"):
        cache.bind(loose_key, first.sha256)
    with pytest.raises(ValueError, match="^cache lookup requires a CacheKey$"):
        cache.lookup(loose_key)
    with pytest.raises(MissingArtifactError, match="cache source has no entry"):
        cache.lookup(key)
    public = store.put(
        b"public evidence", kind="evidence", retention="pinned", egress="value_free"
    )
    with pytest.raises(ValueError, match="^cache object must preserve private-derived egress$"):
        cache.bind(key, public.sha256)
    cache.bind(key, first.sha256)
    cache.bind(key, first.sha256)
    second = store.put(
        b"second evidence", kind="evidence", retention="pinned", egress="private_derived"
    )
    with pytest.raises(StaleCacheError, match="^exact cache key already binds different content$"):
        cache.bind(key, second.sha256)
    assert cache.lookup(key) == first.sha256


def test_cache_lookup_refuses_a_damaged_index_digest(tmp_path: Path) -> None:
    """An index row whose object digest is not text is refused, never returned."""
    store = ArtifactStore(tmp_path / "private")
    cache = ArtifactCache(store)
    key = _key("compiled")
    evidence = store.put(
        b"evidence", kind="evidence", retention="pinned", egress="private_derived"
    )
    cache.bind(key, evidence.sha256)
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE cache_entries SET object_sha256=? WHERE key_sha256=?", (b"\x00", key.digest)
        )
    connection.close()
    with pytest.raises(ValueError, match="^indexed cache digest is invalid$"):
        cache.lookup(key)
