# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI shard admission integration tests
"""Exercise actual timing, artifact, branch-data and aggregate CLI boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import coverage
import pytest
from _ci_shard_execution import IDENTITY, execute, read_worker, run_workers, verification_command

from tools.ci_test_shards import load_timings, partition_tests, verify_shards, write_json


@pytest.fixture(scope="module")
def real_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Produce actual indexed data by executing maintained public logging tests."""
    root = tmp_path_factory.mktemp("ci-shard-admission")
    artifacts, outcomes = run_workers(root / "run")
    assert all(result.returncode == 0 for result in outcomes), [
        result.stdout + result.stderr for result in outcomes
    ]
    return artifacts


def test_real_cli_combines_all_branch_data_and_retains_inputs(
    real_artifacts: Path, tmp_path: Path
) -> None:
    """Admit every actual test once and retain original branch data for diagnosis."""
    output = tmp_path / ".coverage"
    hints = tmp_path / "timings.json"
    result = execute(
        [
            *verification_command(real_artifacts),
            "--coverage-output",
            str(output),
            "--timings-output",
            str(hints),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(result.stdout)
    assert proof["tests"] == len(read_worker(real_artifacts)["inventory"])
    assert sum(proof["per_worker"]) == proof["tests"]
    data = coverage.CoverageData(basename=str(output))
    data.read()
    assert data.has_arcs()
    assert any(path.endswith("/logging_setup.py") for path in data.measured_files())
    assert len(load_timings(hints)) == proof["tests"]
    assert all(
        (real_artifacts / IDENTITY.directory(index) / f".coverage.shard-{index}").is_file()
        for index in (1, 2)
    )
    repeated = execute([*verification_command(real_artifacts), "--coverage-output", str(output)])
    assert repeated.returncode == 1 and "already exists" in repeated.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "absent",
        "extra",
        "identity",
        "index",
        "count",
        "unsealed",
        "failed",
        "collect",
        "incomplete",
        "discovery",
        "timing-snapshot",
        "missing-snapshot",
        "foreign-selected",
        "duplicate",
        "omitted",
        "foreign-time",
        "empty-coverage",
        "corrupt-coverage",
        "tampered-digest",
        "bad-schema",
        "not-object",
        "invalid-test-id",
        "wrong-run",
        "wrong-attempt",
        "wrong-python",
        "wrong-suite",
    ],
)
def test_real_cli_refuses_damaged_actual_artifacts(
    real_artifacts: Path, tmp_path: Path, mutation: str
) -> None:
    """Fault-inject real completed evidence and refuse incomplete or wrong unions."""
    artifacts = tmp_path / "artifacts"
    shutil.copytree(real_artifacts, artifacts)
    directory = artifacts / IDENTITY.directory(1)
    result: dict[str, Any] = read_worker(artifacts)
    if mutation == "absent":
        shutil.rmtree(directory)
    elif mutation == "extra":
        (artifacts / "unexpected-worker").mkdir()
    elif mutation == "identity":
        result["identity"]["source_sha"] = "f" * 40
    elif mutation in {"wrong-run", "wrong-attempt", "wrong-python", "wrong-suite"}:
        field = {
            "wrong-run": "run_id",
            "wrong-attempt": "attempt",
            "wrong-python": "python",
            "wrong-suite": "suite",
        }[mutation]
        result["identity"][field] += "-different"
    elif mutation == "invalid-test-id":
        result["inventory"][0] = None
    elif mutation == "index":
        result["shard"] = 2
    elif mutation == "count":
        result["shards"] = 3
    elif mutation == "unsealed":
        result["sealed"] = False
    elif mutation == "failed":
        result["exitstatus"] = 1
    elif mutation == "collect":
        result["collect_only"] = True
    elif mutation == "incomplete":
        result["completed"] = result["completed"][1:]
    elif mutation == "discovery":
        result["inventory"] = result["inventory"][1:]
    elif mutation == "timing-snapshot":
        result["timing_snapshot_sha256"] = "f" * 64
    elif mutation == "missing-snapshot":
        result.pop("timing_snapshot_sha256")
    elif mutation == "foreign-selected":
        result["selected"].append(
            "tests/test_ci_test_shards.py::test_real_cli_refuses_damaged_actual_artifacts"
        )
        result["completed"] = list(result["selected"])
    elif mutation == "duplicate":
        other = read_worker(artifacts, 2)
        result["selected"].append(other["selected"][0])
        result["completed"] = list(result["selected"])
    elif mutation == "omitted":
        removed = result["selected"].pop()
        result["completed"].remove(removed)
        result["seconds"].pop(removed)
    elif mutation == "foreign-time":
        result["seconds"][
            "tests/test_ci_test_shards.py::test_real_cli_combines_all_branch_data_and_retains_inputs"
        ] = 1.0
    elif mutation == "empty-coverage":
        (directory / ".coverage.shard-1").unlink()
    elif mutation == "corrupt-coverage":
        (directory / ".coverage.shard-1").write_bytes(b"not a coverage SQLite database")
    elif mutation == "tampered-digest":
        result["coverage_sha256"] = "f" * 64
    elif mutation == "bad-schema":
        result["schema_version"] = 2
    if directory.exists():
        write_json(directory / "result.json", [] if mutation == "not-object" else result)
    combined = tmp_path / ".coverage"
    rejected = execute([*verification_command(artifacts), "--coverage-output", str(combined)])
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "CI shard admission failed" in rejected.stderr
    assert not combined.exists()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"schema_version": 2, "seconds": {}},
        {"schema_version": 1, "seconds": []},
        {"schema_version": 1, "seconds": {"": 1}},
        {"schema_version": 1, "seconds": {"node": True}},
        {"schema_version": 1, "seconds": {"node": "1"}},
        {"schema_version": 1, "seconds": {"node": 0}},
        {"schema_version": 1, "seconds": {"node": -1}},
        {"schema_version": 1, "seconds": {"node": float("nan")}},
        {"schema_version": 1, "seconds": {"node": float("inf")}},
    ],
)
def test_snapshot_cli_refuses_corrupt_cache_before_writing_output(
    tmp_path: Path, payload: object
) -> None:
    """Reject cache damage through the same preparation CLI used before pytest."""
    source = tmp_path / "cache.json"
    write_json(source, payload)
    output = tmp_path / "snapshot.json"
    result = execute(
        [
            "-m",
            "tools.ci_test_shards",
            "snapshot",
            "--source",
            str(source),
            "--output",
            str(output),
        ]
    )
    assert result.returncode == 1
    assert not output.exists()


def test_scheduler_uses_complete_real_inventory_in_cold_and_weighted_modes(
    real_artifacts: Path,
) -> None:
    """Retain actual discovery despite stale hints, changed order or new admission."""
    result: dict[str, Any] = read_worker(real_artifacts)
    inventory: list[str] = result["inventory"]
    cold = partition_tests(inventory, 3, {})
    assert sorted(node for group in cold for node in group) == inventory
    assert cold == partition_tests(list(reversed(inventory)), 3, {})
    hints: dict[str, float] = result["seconds"]
    weighted = partition_tests(inventory, 3, hints)
    assert sorted(node for group in weighted for node in group) == inventory
    stale = dict(hints, **{"tests/deleted_module.py::deleted_test": 999.0})
    assert partition_tests(inventory, 3, stale) == weighted
    with pytest.raises(ValueError, match="duplicate"):
        partition_tests([*inventory, inventory[0]], 3, {})
    with pytest.raises(ValueError, match="count"):
        partition_tests(inventory, 0, {})
    with pytest.raises(ValueError, match="nonempty"):
        partition_tests([], 1, {})
    with pytest.raises(ValueError, match="positive"):
        verify_shards(real_artifacts, IDENTITY, 0)


def test_cli_rejects_unreadable_json_and_non_branch_data(
    real_artifacts: Path, tmp_path: Path
) -> None:
    """Reject malformed JSON and line-only data instead of accepting a false union."""
    bad = tmp_path / "broken.json"
    bad.write_text("{", encoding="utf-8")
    result = execute(
        [
            "-m",
            "tools.ci_test_shards",
            "snapshot",
            "--source",
            str(bad),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert result.returncode == 1
    directory = tmp_path / "line-only"
    directory.mkdir()
    write_json(directory / "result.json", read_worker(real_artifacts))
    data = coverage.CoverageData(basename=str(directory / ".coverage.shard-1"))
    data.add_lines({str(Path("src/scpn_quantum_control/logging_setup.py").resolve()): {1}})
    data.write()
    rejected = execute(["-m", "tools.ci_test_shards", "seal", str(directory), "--coverage"])
    assert rejected.returncode == 1 and "branch data" in rejected.stderr
