# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the tier-benchmark measurement core
"""Tests for ``accel.tier_benchmark``: statistics, provenance, and artefacts.

The module is benchmark instrumentation, so the tests pin the deterministic
contract: percentile reduction over a known sample list, warm-up discard and
repeat collection, per-(operation, backend) row construction including the
explicit unavailable path, every provenance helper's success and failure
branch, and the schema / digest shape of the assembled artefact and manifest.
"""

from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path

import pytest

from oscillatools.accel import tier_benchmark as tb

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_compute_stats_orders_percentiles_and_throughput() -> None:
    stats = tb.compute_stats([5.0, 1.0, 3.0, 2.0, 4.0])
    assert stats.min_us == 1.0
    assert stats.max_us == 5.0
    assert stats.p50_us <= stats.p95_us <= stats.p99_us
    assert stats.mean_us == pytest.approx(3.0)
    assert stats.throughput_ops_s == pytest.approx(1.0e6 / 3.0)
    assert stats.samples == 5


def test_compute_stats_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty sample list"):
        tb.compute_stats([])


def test_compute_stats_zero_mean_gives_zero_throughput() -> None:
    stats = tb.compute_stats([0.0, 0.0])
    assert stats.throughput_ops_s == 0.0


def test_tier_stats_to_dict_round_trips_fields() -> None:
    stats = tb.compute_stats([1.0, 2.0, 3.0])
    payload = stats.to_dict()
    assert payload["min_us"] == 1.0
    assert payload["max_us"] == 3.0
    assert set(payload) == {
        "p50_us",
        "p95_us",
        "p99_us",
        "mean_us",
        "min_us",
        "max_us",
        "throughput_ops_s",
        "samples",
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_measure_discards_warmup_then_collects_repeats() -> None:
    calls: list[int] = []
    tb.measure(lambda: calls.append(1), warmup=3, repeats=5, inner=2)
    # 3 warm-up calls + 5 repeats * 2 inner = 13 invocations.
    assert len(calls) == 3 + 5 * 2


@pytest.mark.parametrize(
    ("warmup", "repeats", "inner"),
    [(-1, 5, 1), (0, 0, 1), (0, 5, 0)],
)
def test_measure_rejects_degenerate_parameters(warmup: int, repeats: int, inner: int) -> None:
    with pytest.raises(ValueError, match="warmup >= 0"):
        tb.measure(lambda: None, warmup=warmup, repeats=repeats, inner=inner)


def test_measure_returns_positive_statistics() -> None:
    stats = tb.measure(lambda: sum(range(64)), warmup=1, repeats=4, inner=4)
    assert stats.mean_us >= 0.0
    assert stats.samples == 4


# ---------------------------------------------------------------------------
# Rows and primitive results
# ---------------------------------------------------------------------------


def test_measured_and_unavailable_rows_carry_expected_shape() -> None:
    stats = tb.compute_stats([1.0, 2.0])
    measured = tb.measured_row("rust", stats)
    absent = tb.unavailable_row("julia", "juliacall not installed")

    assert measured.to_dict() == {
        "backend": "rust",
        "status": tb.STATUS_MEASURED,
        "stats": stats.to_dict(),
        "reason": None,
    }
    assert absent.to_dict() == {
        "backend": "julia",
        "status": tb.STATUS_UNAVAILABLE,
        "stats": None,
        "reason": "juliacall not installed",
    }


def test_primitive_result_picks_lowest_p50_backend() -> None:
    fast = tb.measured_row("rust", tb.compute_stats([1.0, 1.0]))
    slow = tb.measured_row("python", tb.compute_stats([9.0, 9.0]))
    result = tb.PrimitiveResult(
        operation="order_parameter",
        size=64,
        rows=(fast, slow, tb.unavailable_row("julia", "excluded")),
        parity_max_abs_diff=1e-15,
        extra={"cost_class": "scalar"},
    )
    assert result.fastest_backend() == "rust"
    payload = result.to_dict()
    assert payload["fastest_backend"] == "rust"
    assert payload["operation"] == "order_parameter"
    assert payload["parity_max_abs_diff"] == 1e-15
    assert payload["extra"] == {"cost_class": "scalar"}
    assert len(payload["rows"]) == 3


def test_primitive_result_without_measured_rows_has_no_fastest() -> None:
    result = tb.PrimitiveResult(
        operation="mean_phase",
        size=8,
        rows=(tb.unavailable_row("rust", "absent"), tb.unavailable_row("python", "absent")),
    )
    assert result.fastest_backend() is None
    assert result.to_dict()["fastest_backend"] is None


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o755)


def test_cpu_model_reads_proc_or_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # The live path returns a non-empty string on the test host.
    assert isinstance(tb._cpu_model(), str)

    class _FakePath:
        def __init__(self, *_: object) -> None:
            pass

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(tb, "Path", _FakePath)
    monkeypatch.setattr(platform, "processor", lambda: "fallback-cpu")
    assert tb._cpu_model() == "fallback-cpu"
    monkeypatch.setattr(platform, "processor", lambda: "")
    assert tb._cpu_model() == "unknown"


def test_cpu_model_handles_cpuinfo_without_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    fake = type(
        "P",
        (),
        {
            "exists": lambda self: True,
            "read_text": lambda self, encoding="utf-8": "processor\t: 0\nflags\t: fpu\n",
        },
    )
    monkeypatch.setattr(tb, "Path", lambda *_: fake())
    monkeypatch.setattr(platform, "processor", lambda: "x86-fallback")
    assert tb._cpu_model() == "x86-fallback"


def test_affinity_returns_sorted_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    affinity = tb._affinity()
    assert affinity is None or affinity == sorted(affinity)

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert tb._affinity() is None


def test_loadavg_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (1.234, 2.0, 3.0))
    assert tb._loadavg() == [1.23, 2.0, 3.0]

    def _raise() -> object:
        raise OSError("no load average")

    monkeypatch.setattr(os, "getloadavg", _raise)
    assert tb._loadavg() is None


def test_git_commit_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commit = tb._git_commit()
    assert isinstance(commit, str) and commit

    git = tmp_path / "git"
    _write_executable(
        git,
        'test "$1" = "rev-parse" && test "$2" = "HEAD" && printf "  deadbeef  \\n"',
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    assert tb._git_commit() == "deadbeef"

    _write_executable(git, 'test "$1" = "rev-parse" && test "$2" = "HEAD"')
    assert tb._git_commit() == "unknown"

    git.chmod(0o644)
    assert tb._git_commit() == "unknown"


def test_rustc_version_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rustc = tmp_path / "rustc"
    _write_executable(rustc, 'test "$1" = "--version" && printf "rustc 1.96.0\\n"')
    monkeypatch.setenv("PATH", str(tmp_path))
    assert tb._rustc_version() == "rustc 1.96.0"

    _write_executable(rustc, 'test "$1" = "--version"')
    assert tb._rustc_version() == "absent"

    rustc.chmod(0o644)
    assert tb._rustc_version() == "absent"


def test_engine_label_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb, "optional_rust_engine", lambda: None)
    assert tb._engine_label() == "absent"

    monkeypatch.setattr(tb, "optional_rust_engine", lambda: type("E", (), {"__version__": "9.9"}))
    assert tb._engine_label() == "9.9"

    monkeypatch.setattr(tb, "optional_rust_engine", lambda: type("E", (), {}))
    assert tb._engine_label() == "installed"


def test_distribution_version_found_and_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "version", lambda name: "1.2.3")
    assert tb._distribution_version("numpy") == "1.2.3"

    def _raise(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)
    assert tb._distribution_version("ghost") == "absent"


def test_capture_provenance_populates_every_field() -> None:
    provenance = tb.capture_provenance()
    payload = provenance.to_dict()
    assert set(payload) == {
        "cpu_model",
        "cpu_count",
        "python",
        "numpy",
        "engine",
        "juliacall",
        "rustc",
        "platform",
        "machine",
        "commit",
        "cpu_affinity",
        "loadavg",
    }
    assert payload["python"].count(".") == 2


# ---------------------------------------------------------------------------
# Artefact assembly
# ---------------------------------------------------------------------------


def test_payload_digest_excludes_existing_digest() -> None:
    body = {"a": 1, "b": [2, 3]}
    digest = tb.payload_digest(body)
    stamped = dict(body, payload_sha256=digest)
    assert tb.payload_digest(stamped) == digest


def _example_results() -> list[tb.PrimitiveResult]:
    rows = (
        tb.measured_row("rust", tb.compute_stats([1.0, 1.1])),
        tb.measured_row("python", tb.compute_stats([4.0, 4.2])),
        tb.unavailable_row("julia", "excluded by --tiers"),
    )
    return [
        tb.PrimitiveResult("order_parameter", 8, rows, parity_max_abs_diff=2e-16),
        tb.PrimitiveResult("mean_phase", 8, rows, parity_max_abs_diff=None),
    ]


def test_build_primitive_artifact_has_schema_and_digest() -> None:
    provenance = tb.capture_provenance()
    artifact = tb.build_primitive_artifact(
        environment="local",
        generated_utc="2026-06-25T00:00:00Z",
        provenance=provenance,
        parameters={"sizes": [8], "seed": 7},
        results=_example_results(),
    )
    assert artifact["schema_version"] == tb.PRIMITIVE_SCHEMA
    assert artifact["environment"] == "local"
    assert artifact["production_claim_allowed"] is False
    assert len(artifact["results"]) == 2
    assert artifact["payload_sha256"] == tb.payload_digest(artifact)


def test_build_manifest_indexes_primitives() -> None:
    provenance = tb.capture_provenance()
    manifest = tb.build_manifest(
        generated_utc="2026-06-25T00:00:00Z",
        provenance=provenance,
        parameters={"sizes": [8], "seed": 7},
        results=_example_results(),
        tier_availability={
            "rust": "available",
            "julia": "unavailable: excluded",
            "python": "available",
        },
    )
    assert manifest["schema_version"] == tb.MANIFEST_SCHEMA
    assert manifest["primitive_count"] == 2
    assert manifest["primitives"][0]["operation"] == "order_parameter"
    assert manifest["primitives"][0]["fastest_backend"] == "rust"
    assert manifest["primitives"][0]["backends"] == ["rust", "python", "julia"]
    assert manifest["tier_availability"]["julia"].startswith("unavailable")
    assert manifest["payload_sha256"] == tb.payload_digest(manifest)


def test_resolve_executable_resolution_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``which`` hit whose path resolution raises ``OSError`` resolves to ``None``."""
    monkeypatch.setattr(tb.shutil, "which", lambda _command: str(tmp_path / "ghost"))

    class _ResolveRaises:
        def __init__(self, *_args: object) -> None:
            pass

        def resolve(self, strict: bool = False) -> Path:
            raise OSError("resolution failed")

    monkeypatch.setattr(tb, "Path", _ResolveRaises)
    assert tb._resolve_executable("ghost") is None


def test_resolve_executable_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``which`` hit that is a plain, non-executable file resolves to ``None``."""
    plain = tmp_path / "note.txt"
    plain.write_text("not executable", encoding="utf-8")
    plain.chmod(0o644)
    monkeypatch.setattr(tb.shutil, "which", lambda _command: str(plain))
    assert tb._resolve_executable("note") is None


def test_run_admitted_command_unresolvable_and_subprocess_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty command is rejected; unresolvable and subprocess ``OSError`` yield ``None``."""
    with pytest.raises(ValueError, match="command must contain an executable"):
        tb._run_admitted_command(())

    monkeypatch.setattr(tb, "_resolve_executable", lambda _command: None)
    assert tb._run_admitted_command(("ghost-command",)) is None

    monkeypatch.setattr(tb, "_resolve_executable", lambda _command: "/usr/bin/true")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("exec failed")

    monkeypatch.setattr(tb.subprocess, "run", _raise)
    assert tb._run_admitted_command(("true",)) is None
