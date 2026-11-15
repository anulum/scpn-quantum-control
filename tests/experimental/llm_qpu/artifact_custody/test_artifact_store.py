# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — private artifact custody acceptance
"""Exercise immutable raw-shard custody against real disk failures."""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import ArrayDescriptor
from scpn_quantum_control.experimental.llm_qpu.data.artifact_store import (
    ArtifactStore,
    CorruptArtifactError,
    MissingArtifactError,
    open_repository_store,
)


def _array(values: tuple[float, ...]) -> tuple[ArrayDescriptor, bytes]:
    """Return a one-column float32 descriptor and its exact payload."""
    payload = struct.pack(f"<{len(values)}f", *values)
    return (
        ArrayDescriptor(
            "<f4", (len(values), 1), len(payload), hashlib.sha256(payload).hexdigest()
        ),
        payload,
    )


def test_t03a_conflicting_disk_bytes_are_quarantined_without_overwrite(tmp_path: Path) -> None:
    """A content address cannot silently replace conflicting bytes."""
    store = ArtifactStore(tmp_path / "private")
    descriptor, payload = _array((1.0, 2.0))
    record = store.put_array(descriptor, payload)
    object_path = store.objects / record.sha256[:2] / record.sha256
    object_path.write_bytes(b"conflicting bytes")

    with pytest.raises(CorruptArtifactError, match="digest mismatch"):
        store.put_array(descriptor, payload)

    assert not object_path.exists()
    assert [path.read_bytes() for path in store.quarantine.iterdir()] == [b"conflicting bytes"]
    with pytest.raises(MissingArtifactError):
        store.get(record.sha256)


def test_t03b_interrupted_write_leaves_no_valid_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial filesystem write followed by disk failure admits no object."""
    root = tmp_path / "private"
    store = ArtifactStore(root)
    payload = b"x" * 8192
    digest = hashlib.sha256(payload).hexdigest()
    original_write = os.write
    writes = 0

    def fail_after_partial_write(descriptor: int, chunk: bytes | memoryview) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return original_write(descriptor, chunk[:1024])
        raise OSError("simulated disk full after partial write")

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", fail_after_partial_write)
        with pytest.raises(OSError, match="simulated disk full"):
            store.put(payload, kind="evidence", retention="pinned", egress="private_derived")
    assert writes == 2
    with pytest.raises(MissingArtifactError):
        store.get(digest)
    assert list(store.objects.rglob("*")) == [store.objects / digest[:2]]
    assert not list(store.objects.rglob("*.tmp"))


def test_t03d_rejects_escape_symlink_and_archive_before_decode(tmp_path: Path) -> None:
    """Only digest paths and raw descriptor bytes enter the shard store."""
    store = ArtifactStore(tmp_path / "private")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        store.get("../outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    (store.objects / "ff").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.get("f" * 64)

    payload = b"PK\x03\x04" + b"\0" * 4
    descriptor = ArrayDescriptor("<u4", (2, 1), 8, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="container formats"):
        store.put_array(descriptor, payload)
    assert not list(store.objects.rglob("*.tmp"))


def test_private_repo_root_and_index_permissions(tmp_path: Path) -> None:
    """Derived data lands under the dedicated results root with private index."""
    store = open_repository_store(tmp_path)
    assert store.root == tmp_path / "results" / "experimental" / "llm_qpu" / "private"
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.database.stat().st_mode & 0o777 == 0o600

    store.root.chmod(0o755)
    with pytest.raises(ValueError, match="owned and private"):
        open_repository_store(tmp_path)


def _concurrent_writer(monkeypatch: pytest.MonkeyPatch, published: bytes) -> None:
    """Make the publish step meet a target that another writer created first."""

    def link_after_other_writer(source: str, target: str, *, follow_symlinks: bool) -> None:
        Path(target).write_bytes(published)
        raise FileExistsError(target)

    monkeypatch.setattr(os, "link", link_after_other_writer)


def test_concurrent_conflicting_publish_is_quarantined_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes another writer placed at the same content address are quarantined, never indexed."""
    store = ArtifactStore(tmp_path / "private")
    payload = b"payload bytes"
    digest = hashlib.sha256(payload).hexdigest()
    _concurrent_writer(monkeypatch, b"other writer bytes")
    with pytest.raises(CorruptArtifactError, match="^concurrent content ID collision$"):
        store.put(payload, kind="evidence", retention="pinned", egress="private_derived")
    monkeypatch.undo()
    assert not (store.objects / digest[:2] / digest).exists()
    assert [path.read_bytes() for path in store.quarantine.iterdir()] == [b"other writer bytes"]
    assert not list(store.objects.rglob("*.tmp"))
    assert not store.has(digest)


def test_concurrent_identical_publish_is_admitted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identical concurrent publish is verified and indexed like a fresh one."""
    store = ArtifactStore(tmp_path / "private")
    payload = b"payload bytes"
    _concurrent_writer(monkeypatch, payload)
    record = store.put(payload, kind="evidence", retention="cache", egress="value_free")
    monkeypatch.undo()
    assert store.get(record.sha256) == (record, payload)
    assert not list(store.quarantine.iterdir())
    assert not list(store.objects.rglob("*.tmp"))
