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
import struct
from dataclasses import replace
from pathlib import Path

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
