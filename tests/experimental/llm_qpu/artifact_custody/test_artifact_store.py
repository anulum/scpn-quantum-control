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
import sqlite3
import struct
from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize(
    ("total", "per_object", "message"),
    [
        (0, 1, "invalid total-byte capacity"),
        (2**42 + 1, 1, "invalid total-byte capacity"),
        (True, 1, "invalid total-byte capacity"),
        (16, 0, "invalid object-byte capacity"),
        (16, 17, "invalid object-byte capacity"),
        (16, 8.0, "invalid object-byte capacity"),
    ],
)
def test_store_refuses_invalid_capacities(
    tmp_path: Path, total: Any, per_object: Any, message: str
) -> None:
    """Capacities must be bounded positive integers with the object limit inside the total."""
    with pytest.raises(ValueError, match=f"^{message}$"):
        ArtifactStore(tmp_path / "private", max_total_bytes=total, max_object_bytes=per_object)
    assert not (tmp_path / "private").exists()


def test_store_refuses_text_roots_shared_indexes_and_missing_repositories(tmp_path: Path) -> None:
    """The root must be a Path, the index private, and a repository root an existing directory."""
    text_root: Any = str(tmp_path / "private")
    with pytest.raises(ValueError, match="^artifact root must be a Path$"):
        ArtifactStore(text_root)
    for mode in (0o640, 0o604):
        root = tmp_path / f"shared-index-{mode:o}"
        root.mkdir(mode=0o700)
        index = root / "index.sqlite3"
        index.touch(mode=0o600)
        index.chmod(mode)
        with pytest.raises(
            ValueError, match=r"^artifact index must be owned and private \(0600\)$"
        ):
            ArtifactStore(root)
    text_repository: Any = str(tmp_path)
    for repository in (tmp_path / "missing", text_repository):
        with pytest.raises(ValueError, match="^repository root must be an existing directory$"):
            open_repository_store(repository)


def test_put_refuses_unbounded_payloads_unknown_labels_and_overflow(tmp_path: Path) -> None:
    """Payloads are bounded bytes with known labels, and the store never passes its capacity."""
    store = ArtifactStore(tmp_path / "private", max_total_bytes=16, max_object_bytes=10)
    mutable: Any = bytearray(b"abc")
    for payload in (b"", mutable, b"x" * 11):
        with pytest.raises(ValueError, match="^artifact payload must be bounded nonempty bytes$"):
            store.put(payload, kind="evidence", retention="pinned", egress="value_free")
    labels: list[dict[str, Any]] = [
        {"kind": "archive", "retention": "pinned", "egress": "value_free"},
        {"kind": "evidence", "retention": "forever", "egress": "value_free"},
        {"kind": "evidence", "retention": "pinned", "egress": "public"},
    ]
    for label in labels:
        with pytest.raises(ValueError, match="^unknown artifact custody label$"):
            store.put(b"abc", **label)
    first = store.put(b"0123456789", kind="evidence", retention="pinned", egress="value_free")
    with pytest.raises(ValueError, match="^artifact store capacity exceeded$"):
        store.put(b"abcdefghij", kind="evidence", retention="pinned", egress="value_free")
    again = store.put(b"0123456789", kind="evidence", retention="pinned", egress="value_free")
    assert again == first
    assert store.has(first.sha256)
    assert not store.has(hashlib.sha256(b"abcdefghij").hexdigest())


def test_put_refuses_a_content_id_under_different_custody(tmp_path: Path) -> None:
    """One content address keeps one custody record; a relabelled publish is refused."""
    store = ArtifactStore(tmp_path / "private")
    record = store.put(b"evidence", kind="evidence", retention="pinned", egress="value_free")
    with pytest.raises(
        CorruptArtifactError, match="^existing content ID has different custody metadata$"
    ):
        store.put(b"evidence", kind="manifest", retention="pinned", egress="value_free")
    assert store.get(record.sha256) == (record, b"evidence")


def test_put_refuses_a_zero_byte_write_and_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that makes no progress is a disk failure, not an empty success."""
    store = ArtifactStore(tmp_path / "private")
    payload = b"payload bytes"

    def stalled_write(descriptor: int, chunk: bytes | memoryview) -> int:
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", stalled_write)
        with pytest.raises(OSError, match="^short artifact write$"):
            store.put(payload, kind="evidence", retention="pinned", egress="value_free")
    assert not store.has(hashlib.sha256(payload).hexdigest())
    assert not list(store.objects.rglob("*.tmp"))


def test_put_array_refuses_an_unreviewed_descriptor(tmp_path: Path) -> None:
    """Array storage needs the frozen descriptor contract, not its wire mapping."""
    store = ArtifactStore(tmp_path / "private")
    descriptor, payload = _array((1.0, 2.0))
    loose: Any = descriptor.to_wire()
    with pytest.raises(ValueError, match="^array storage requires an ArrayDescriptor$"):
        store.put_array(loose, payload)
    assert store.put_array(descriptor, payload, retention="cache").retention == "cache"


def _set_indexed_length(store: ArtifactStore, digest: str, length: int) -> None:
    """Rewrite one index row's byte length, as a damaged index would hold it."""
    with sqlite3.connect(store.database) as connection:
        connection.execute("UPDATE objects SET byte_length=? WHERE sha256=?", (length, digest))
    connection.close()


def test_get_refuses_index_rows_that_disagree_with_the_object(tmp_path: Path) -> None:
    """A damaged index length is refused before or after reading, never trusted."""
    store = ArtifactStore(tmp_path / "private")
    record = store.put(b"0123456789", kind="evidence", retention="pinned", egress="value_free")
    for length, message in (
        (0, "indexed artifact length is invalid"),
        (store.max_object_bytes + 1, "indexed artifact length is invalid"),
        (9, "stored artifact exceeds declared limit"),
        (11, "indexed artifact length mismatch"),
    ):
        _set_indexed_length(store, record.sha256, length)
        with pytest.raises(CorruptArtifactError, match=f"^{message}$"):
            store.get(record.sha256)
    _set_indexed_length(store, record.sha256, 10)
    assert store.get(record.sha256) == (record, b"0123456789")


def test_get_refuses_an_object_that_grows_while_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes beyond the indexed length are refused even if the size check saw a smaller file."""
    store = ArtifactStore(tmp_path / "private")
    record = store.put(b"0123456789", kind="evidence", retention="pinned", egress="value_free")
    _set_indexed_length(store, record.sha256, 9)
    real_fstat = os.fstat

    def stale_fstat(descriptor: int) -> os.stat_result:
        info = real_fstat(descriptor)
        fields = list(info[:10])
        fields[6] = 0
        return os.stat_result(fields)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", stale_fstat)
        with pytest.raises(CorruptArtifactError, match="^stored artifact exceeds declared limit$"):
            store.get(record.sha256)
    object_path = store.objects / record.sha256[:2] / record.sha256
    object_path.unlink()
    assert not store.has(record.sha256)
