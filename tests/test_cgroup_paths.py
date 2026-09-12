# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Process cgroup memory path contract tests
"""Exercise proc-to-controller mapping through the real default budget APIs.

All files are tiny injected snapshots. No cgroups, containers or allocations
are created, so this does not replace hosted container acceptance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scpn_quantum_control.dense_budget as budget
from scpn_quantum_control._cgroup_paths import memory_cgroup_paths


def _controls(directory: Path, version: int, limit: str, usage: str = "0") -> None:
    """Write a tiny finite or unlimited memory-controller snapshot."""
    directory.mkdir(parents=True, exist_ok=True)
    names = (
        ("memory.max", "memory.current")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    (directory / names[0]).write_text(limit, encoding="utf-8")
    (directory / names[1]).write_text(usage, encoding="utf-8")


def _mount(directory: Path, version: int, root: str = "/") -> str:
    """Build a kernel-shaped mountinfo line with encoded whitespace/backslashes."""
    encoded = str(directory).replace("\\", r"\134").replace(" ", r"\040")
    encoded = encoded.replace("\t", r"\011").replace("\n", r"\012")
    fs = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,cpu,memory"
    return f"37 28 0:31 {root} {encoded} rw shared:9 - {fs}\n"


def _proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, membership: str, mounts: str) -> Path:
    """Bind the public default guard to real test-owned proc metadata files."""
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    (proc / "cgroup").write_text(membership, encoding="utf-8")
    (proc / "mountinfo").write_text(mounts, encoding="utf-8")
    monkeypatch.setattr(budget, "DEFAULT_PROC_ROOT", proc)
    monkeypatch.setattr(budget, "host_available_memory_bytes", lambda: 10**9)
    monkeypatch.delenv(budget.DEFAULT_DENSE_BUDGET_ENV, raising=False)
    return proc


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("leaf_limit", ["max", "1000"])
def test_parent_headroom_accounts_for_sibling_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int, leaf_limit: str
) -> None:
    """Parent use, not only this child's use, constrains the public default guard."""
    mount = tmp_path / "controller"
    _controls(mount, version, "max")
    _controls(mount / "parent", version, "1000", "900")
    _controls(mount / "parent/child", version, leaf_limit, "10")
    membership = "0::/parent/child\n" if version == 2 else "5:cpu,memory:/parent/child\n"
    proc = _proc(tmp_path, monkeypatch, membership, _mount(mount, version))
    paths = memory_cgroup_paths(proc)
    assert paths is not None
    assert [(entry.directory, entry.leaf) for entry in paths] == [
        (mount / "parent/child", True),
        (mount / "parent", False),
        (mount, False),
    ]
    assert budget.available_memory_bytes() == 100
    assert budget.dense_budget_bytes() == 30
    with pytest.raises(budget.DenseAllocationError, match="above the active"):
        budget.require_dense_allocation(1)
    assert budget.dense_budget_bytes(max_gib=1.0) == budget.GIB
    monkeypatch.setenv(budget.DEFAULT_DENSE_BUDGET_ENV, "2")
    assert budget.dense_budget_bytes() == 2 * budget.GIB


@pytest.mark.parametrize(
    "mount_name", ["custom", "space here", "back\\slash", "tab\there", "line\nhere"]
)
@pytest.mark.parametrize("version", [1, 2])
def test_subtree_mount_mapping_and_escaped_mountpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount_name: str, version: int
) -> None:
    """Map a non-root hierarchy mount without traversing into its host parent."""
    mount = tmp_path / mount_name
    _controls(mount, version, "500", "100")
    _controls(mount / "child", version, "300", "100")
    _controls(tmp_path, version, "1", "1")
    member = "0::/tenant/child\n" if version == 2 else "5:memory:/tenant/child\n"
    _proc(tmp_path, monkeypatch, member, _mount(mount, version, "/tenant"))
    assert budget.cgroup_headroom_bytes() == 200
    assert budget.require_dense_allocation(1, rank=1).bytes_required == 32


@pytest.mark.parametrize("version", [1, 2])
def test_namespace_root_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    """A namespace exposing its process cgroup as slash maps directly to the mount."""
    mount = tmp_path / "namespace"
    _controls(mount, version, "256", "16")
    _proc(
        tmp_path, monkeypatch, "0::/\n" if version == 2 else "5:memory:/\n", _mount(mount, version)
    )
    assert budget.cgroup_headroom_bytes() == 240


@pytest.mark.parametrize(
    "membership",
    [
        "broken",
        "0::relative",
        "0::/../escape",
        "0::/other\n",
        "0::/a\n0::/b",
        "0::/bad\x00",
        "x:memory:/tenant",
        "1::/tenant",
    ],
)
def test_unmappable_membership_refuses_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, membership: str
) -> None:
    """Malformed or invisible membership must not fall back to free host memory."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "max")
    proc = _proc(tmp_path, monkeypatch, membership, _mount(mount, 2, "/tenant"))
    with pytest.raises(ValueError):
        memory_cgroup_paths(proc)
    assert budget.available_memory_bytes() == 0


def test_absent_leaf_refuses_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A disappearing process cgroup is not an unrestricted process."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "max")
    _proc(tmp_path, monkeypatch, "0::/gone", _mount(mount, 2))
    assert budget.dense_budget_bytes() == 0


def test_unreadable_parent_usage_refuses_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unknown-usage refusal applies to ancestor limits as well as leaf limits."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "500", "invalid")
    _controls(mount / "child", 2, "max")
    _proc(tmp_path, monkeypatch, "0::/child", _mount(mount, 2))
    assert budget.dense_budget_bytes() == 0


def test_legacy_nonhierarchical_parent_does_not_constrain_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 parent explicitly disabling hierarchical accounting is not inherited."""
    mount = tmp_path / "legacy"
    _controls(mount, 1, "10", "10")
    (mount / "memory.use_hierarchy").write_text("0", encoding="utf-8")
    _controls(mount / "child", 1, "200", "50")
    _proc(tmp_path, monkeypatch, "5:memory:/child", _mount(mount, 1))
    assert budget.cgroup_headroom_bytes() == 150


def test_irrelevant_and_duplicate_mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore other controllers/filesystems, tolerate unrelated malformed lines and deduplicate."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "500", "10")
    mounts = "bad line\n1 0 0:1 / /cpu rw - cgroup cgroup rw,cpu\n"
    mounts += "2 0 0:2 / /tmp rw - tmpfs tmpfs rw\n"
    mounts += _mount(mount, 1) + _mount(mount, 2) * 2
    proc = _proc(tmp_path, monkeypatch, "1:cpu:/elsewhere\n0::/\n0::/", mounts)
    paths = memory_cgroup_paths(proc)
    assert paths is not None and len(paths) == 1
    assert budget.cgroup_headroom_bytes() == 490


def test_no_memory_membership_uses_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-memory memberships do not invent a memory-controller constraint."""
    _proc(tmp_path, monkeypatch, "1:cpu:/worker", "")
    assert budget.available_memory_bytes() == 10**9


def test_unlimited_hierarchy_uses_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An entirely unlimited visible hierarchy contributes no finite allowance."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "max")
    _proc(tmp_path, monkeypatch, "0::/", _mount(mount, 2))
    assert budget.available_memory_bytes() == 10**9


@pytest.mark.parametrize("contents", [None, b"\xff"])
def test_unavailable_proc_retains_flat_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes | None
) -> None:
    """Absent proc uses the flat fallback; invalid encoded metadata refuses admission."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "500", "100")
    proc = tmp_path / "proc"
    proc.mkdir()
    if contents is not None:
        (proc / "cgroup").write_bytes(contents)
    monkeypatch.setattr(budget, "DEFAULT_PROC_ROOT", proc)
    monkeypatch.setattr(budget, "DEFAULT_CGROUP_ROOT", mount)
    assert budget.cgroup_headroom_bytes() == (400 if contents is None else 0)


def test_explicit_flat_root_ignores_process_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing explicit-root callers keep their flat-controller snapshot contract."""
    mount = tmp_path / "controller"
    _controls(mount, 2, "500", "100")
    _proc(tmp_path, monkeypatch, "malformed", "")
    assert budget.cgroup_headroom_bytes(mount) == 400
