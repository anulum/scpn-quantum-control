# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Container-aware dense budget tests
"""The default budget must respect the allowance, not the host's free memory.

``available_memory_bytes`` read ``SC_AVPHYS_PAGES`` and stopped there. Inside a
memory-limited container that is the machine's free memory, not the process's,
so the default budget could exceed what the process may actually use. The
explicit override existed but did not correct the default, which is what
unattended callers get.

Every cgroup here is an injected directory tree. Nothing allocates the budget
and no container is started. The separate Docker workflow checks live default
admission in memory-limited containers; these fixtures do not prove that run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import scpn_quantum_control.dense_budget as dense_budget
from scpn_quantum_control.dense_budget import (
    CGROUP_UNLIMITED_THRESHOLD,
    DEFAULT_DENSE_BUDGET_CAP_GIB,
    DEFAULT_DENSE_BUDGET_ENV,
    DEFAULT_DENSE_RAM_FRACTION,
    GIB,
    available_memory_bytes,
    cgroup_headroom_bytes,
    dense_budget_bytes,
    host_available_memory_bytes,
)

MIB = 1024**2
"""One mebibyte, the scale these fixtures work at."""


@pytest.fixture(autouse=True)
def flat_controller_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these flat-root fixtures independent of the host's proc hierarchy.

    Parameters
    ----------
    tmp_path
        Test-owned root; the proc subdirectory is intentionally absent.
    monkeypatch
        Select the documented unavailable-proc fallback. Process discovery
        itself is exercised by the dedicated cgroup path tests.
    """
    monkeypatch.setattr(dense_budget, "DEFAULT_PROC_ROOT", tmp_path / "absent-proc")


def _write_v2(root: Path, *, limit: str | None, current: str | None = None) -> Path:
    """Lay out a cgroup v2 memory controller.

    Parameters
    ----------
    root
        Directory to write into.
    limit
        Contents of ``memory.max``, or ``None`` to leave the file absent.
    current
        Contents of ``memory.current``, or ``None`` to leave it absent.

    Returns
    -------
    pathlib.Path
        The root, for chaining.

    """
    if limit is not None:
        (root / "memory.max").write_text(limit, encoding="utf-8")
    if current is not None:
        (root / "memory.current").write_text(current, encoding="utf-8")
    return root


def _write_v1(root: Path, *, limit: str | None, usage: str | None = None) -> Path:
    """Lay out a cgroup v1 memory controller.

    Parameters
    ----------
    root
        Directory to write into.
    limit
        Contents of ``memory/memory.limit_in_bytes``, or ``None`` for absent.
    usage
        Contents of ``memory/memory.usage_in_bytes``, or ``None`` for absent.

    Returns
    -------
    pathlib.Path
        The root, for chaining.

    """
    (root / "memory").mkdir(exist_ok=True)
    if limit is not None:
        (root / "memory" / "memory.limit_in_bytes").write_text(limit, encoding="utf-8")
    if usage is not None:
        (root / "memory" / "memory.usage_in_bytes").write_text(usage, encoding="utf-8")
    return root


class TestCgroupV2:
    """The layout a modern container host presents."""

    def test_headroom_is_the_limit_minus_current_use(self, tmp_path: Path) -> None:
        """A container that has already used part of its allowance says so.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(512 * MIB), current=str(200 * MIB))

        assert cgroup_headroom_bytes(tmp_path) == 312 * MIB

    def test_absent_usage_has_no_verified_headroom(self, tmp_path: Path) -> None:
        """A known limit cannot justify allocation when current usage is unknown.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(256 * MIB))

        assert cgroup_headroom_bytes(tmp_path) == 0

    def test_use_beyond_the_limit_reports_no_headroom(self, tmp_path: Path) -> None:
        """Headroom is clamped at zero rather than going negative.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(100 * MIB), current=str(150 * MIB))

        assert cgroup_headroom_bytes(tmp_path) == 0

    def test_the_literal_max_means_unlimited(self, tmp_path: Path) -> None:
        """Cgroup v2 spells "no limit" as the word ``max``.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit="max", current=str(10 * MIB))

        assert cgroup_headroom_bytes(tmp_path) is None


class TestCgroupV1:
    """The older layout, and its sentinel for "unlimited"."""

    def test_headroom_is_the_limit_minus_usage(self, tmp_path: Path) -> None:
        """v1 names the same two quantities differently.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v1(tmp_path, limit=str(400 * MIB), usage=str(150 * MIB))

        assert cgroup_headroom_bytes(tmp_path) == 250 * MIB

    def test_the_sentinel_limit_means_unlimited(self, tmp_path: Path) -> None:
        """v1 has no ``max`` spelling and writes a near-maximum integer.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v1(tmp_path, limit="9223372036854771712", usage=str(10 * MIB))

        assert cgroup_headroom_bytes(tmp_path) is None

    def test_the_unlimited_threshold_is_the_boundary(self, tmp_path: Path) -> None:
        """Just below the threshold is a limit; at it and above is not.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v1(tmp_path, limit=str(CGROUP_UNLIMITED_THRESHOLD - 1), usage="0")
        assert cgroup_headroom_bytes(tmp_path) == CGROUP_UNLIMITED_THRESHOLD - 1

        _write_v1(tmp_path, limit=str(CGROUP_UNLIMITED_THRESHOLD), usage="0")
        assert cgroup_headroom_bytes(tmp_path) is None

    def test_version_two_is_preferred_when_both_are_present(self, tmp_path: Path) -> None:
        """A host can mount both; the modern controller is authoritative.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(64 * MIB), current="0")
        _write_v1(tmp_path, limit=str(999 * MIB), usage="0")

        assert cgroup_headroom_bytes(tmp_path) == 64 * MIB


class TestUnreadableAndMalformed:
    """Anything that cannot be understood must not widen the budget."""

    def test_a_missing_mount_reports_nothing(self, tmp_path: Path) -> None:
        """No cgroup files at all is the ordinary unrestricted host.

        Parameters
        ----------
        tmp_path
            Empty injected cgroup root.

        """
        assert cgroup_headroom_bytes(tmp_path) is None

    @pytest.mark.parametrize("content", ["", "   ", "abc", "12.5", "1e9", "-1", "1 2"])
    def test_malformed_values_report_nothing(self, tmp_path: Path, content: str) -> None:
        """A control file that is not a plain integer is not a number.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        content
            Contents written to ``memory.max``.

        """
        _write_v2(tmp_path, limit=content, current="0")

        assert cgroup_headroom_bytes(tmp_path) is None

    def test_a_malformed_usage_has_no_verified_headroom(self, tmp_path: Path) -> None:
        """Malformed usage cannot be interpreted as a completely unused allowance.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(128 * MIB), current="not a number")

        assert cgroup_headroom_bytes(tmp_path) == 0

    @pytest.mark.parametrize("version", [1, 2])
    @pytest.mark.parametrize(
        "usage", [None, b"", b"abc", b"-1", b"+1", b"1_0", b"\xff", "１２".encode()]
    )
    def test_unknown_usage_refuses_default_allocation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        version: int,
        usage: bytes | None,
    ) -> None:
        """Refuse the public allocation request without allocating any buffers.

        Parameters
        ----------
        tmp_path
            Injected controller filesystem.
        monkeypatch
            Bind the default controller root and remove operator overrides.
        version
            Controller layout, v1 or v2.
        usage
            Missing or invalid current-byte value.

        """
        monkeypatch.delenv(DEFAULT_DENSE_BUDGET_ENV, raising=False)
        monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
        if version == 2:
            _write_v2(tmp_path, limit=str(128 * MIB))
            usage_path = tmp_path / "memory.current"
        else:
            _write_v1(tmp_path, limit=str(128 * MIB))
            usage_path = tmp_path / "memory/memory.usage_in_bytes"
        if usage is not None:
            usage_path.write_bytes(usage)
        assert available_memory_bytes(tmp_path) == 0
        assert dense_budget_bytes() == 0
        with pytest.raises(dense_budget.DenseAllocationError, match="above the active"):
            dense_budget.require_dense_allocation(1)
        assert dense_budget_bytes(max_gib=1.0) == GIB
        monkeypatch.setenv(DEFAULT_DENSE_BUDGET_ENV, "2")
        assert dense_budget_bytes() == 2 * GIB

    @pytest.mark.parametrize("content", [b"+1", b"1_0", b"\xff", "１２".encode()])
    def test_non_kernel_integer_limit_is_unknown(
        self,
        tmp_path: Path,
        content: bytes,
    ) -> None:
        """Reject non-ASCII and Python-only integer syntax without decoding errors.

        Parameters
        ----------
        tmp_path
            Injected v2 controller directory.
        content
            Invalid limit bytes; an unknown limit remains distinct from zero.

        """
        (tmp_path / "memory.max").write_bytes(content)
        (tmp_path / "memory.current").write_text("0", encoding="utf-8")
        assert cgroup_headroom_bytes(tmp_path) is None

    def test_an_unreadable_file_reports_nothing(self, tmp_path: Path) -> None:
        """Permission denied is a failure to read, not a limit of zero.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        if os.geteuid() == 0:
            pytest.skip("root can read a mode-000 file, so this cannot be exercised")
        path = _write_v2(tmp_path, limit=str(64 * MIB)) / "memory.max"
        path.chmod(0o000)
        try:
            assert cgroup_headroom_bytes(tmp_path) is None
        finally:
            path.chmod(0o644)

    def test_a_directory_where_a_file_belongs_reports_nothing(self, tmp_path: Path) -> None:
        """A mount can be shaped unexpectedly; reading it must not raise.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        (tmp_path / "memory.max").mkdir()

        assert cgroup_headroom_bytes(tmp_path) is None


class TestEffectiveAvailability:
    """Host memory and cgroup headroom combined."""

    def test_the_smaller_of_the_two_wins(self, tmp_path: Path) -> None:
        """A container smaller than the host's free memory bounds the process.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        _write_v2(tmp_path, limit=str(16 * MIB), current="0")

        assert available_memory_bytes(tmp_path) == 16 * MIB

    def test_an_unrestricted_host_is_unchanged(self, tmp_path: Path) -> None:
        """With no cgroup limit the answer is the host figure, exactly.

        Parameters
        ----------
        tmp_path
            Empty injected cgroup root.

        """
        assert available_memory_bytes(tmp_path) == host_available_memory_bytes()

    def test_a_limit_larger_than_the_host_does_not_inflate_it(self, tmp_path: Path) -> None:
        """A generous container cannot conjure memory the host lacks.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.

        """
        host = host_available_memory_bytes()
        assert host is not None
        _write_v2(tmp_path, limit=str(host + 64 * GIB), current="0")

        assert available_memory_bytes(tmp_path) == host

    def test_headroom_alone_is_used_when_the_host_reports_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A platform without ``sysconf`` still gets the container's allowance.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        monkeypatch
            Used to silence the host source.

        """
        monkeypatch.setattr(dense_budget, "host_available_memory_bytes", lambda: None)
        _write_v2(tmp_path, limit=str(32 * MIB), current="0")

        assert available_memory_bytes(tmp_path) == 32 * MIB

    def test_neither_source_reporting_gives_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller must be able to tell "unknown" from "zero".

        Parameters
        ----------
        tmp_path
            Empty injected cgroup root.
        monkeypatch
            Used to silence the host source.

        """
        monkeypatch.setattr(dense_budget, "host_available_memory_bytes", lambda: None)

        assert available_memory_bytes(tmp_path) is None


class TestBudgetHonoursTheAllowance:
    """The default budget, which is what unattended callers get."""

    def test_a_small_container_bounds_the_default_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect itself: the cap used to apply to host memory.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        monkeypatch
            Used to install the injected root and clear the override.

        """
        monkeypatch.delenv(DEFAULT_DENSE_BUDGET_ENV, raising=False)
        monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
        _write_v2(tmp_path, limit=str(256 * MIB), current="0")

        budget = dense_budget_bytes()

        assert budget == int(256 * MIB * DEFAULT_DENSE_RAM_FRACTION)
        assert budget < int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)

    def test_the_cap_still_applies_on_a_large_allowance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Container awareness lowers the budget; it never raises it.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        monkeypatch
            Used to install the injected root and a large host figure.

        """
        monkeypatch.delenv(DEFAULT_DENSE_BUDGET_ENV, raising=False)
        monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(dense_budget, "host_available_memory_bytes", lambda: 512 * GIB)
        _write_v2(tmp_path, limit=str(256 * GIB), current="0")

        assert dense_budget_bytes() == int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)

    def test_the_explicit_argument_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operator overrides keep their meaning, as the card requires.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        monkeypatch
            Used to install the injected root.

        """
        monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
        _write_v2(tmp_path, limit=str(16 * MIB), current="0")

        assert dense_budget_bytes(max_gib=2.0) == int(2.0 * GIB)

    def test_the_environment_override_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator who sets the variable means it, container or not.

        Parameters
        ----------
        tmp_path
            Injected cgroup root.
        monkeypatch
            Used to install the injected root and the override.

        """
        monkeypatch.setattr(dense_budget, "DEFAULT_CGROUP_ROOT", tmp_path)
        monkeypatch.setenv(DEFAULT_DENSE_BUDGET_ENV, "3")
        _write_v2(tmp_path, limit=str(16 * MIB), current="0")

        assert dense_budget_bytes() == int(3 * GIB)
