# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded corpus replay regression tests
"""Prove that numerical replay never excuses corrupted custody or input drift."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _contract_custody_replay_helpers import (
    FISHER_FIXTURE,
    assert_corpus_replay,
    recorded_successors,
    successor_bytes,
)

from scpn_quantum_control import stable_core_product as codec

FROZEN = Path(__file__).parent / "data" / "contract_custody_corpus"
COMPILE_PLAN = "studio_compile_default_preserves_plan.json"


def _reseal_studio_row(root: Path, name: str, digest: str) -> None:
    """Record ``digest`` for one fixture in the copied Studio manifest."""
    path = root / "manifest_studio.json"
    manifest = json.loads(path.read_text())
    for row in manifest["cases"]:
        if row["fixture"] == name:
            row["fixture_sha256"] = digest
    path.write_text(json.dumps(manifest))


def _write_recorded_successors(root: Path) -> None:
    """Replace the copied frozen plans by their recorded successors and reseal them."""
    for name, row in recorded_successors().items():
        frozen = json.loads((FROZEN / name).read_text())
        (root / name).write_bytes(successor_bytes(frozen, row))
        _reseal_studio_row(root, name, row["fixture_sha256"])


def _mutate_replay(root: Path, fault: str, steps: int = 0) -> None:
    """Alter one copied fixture and reseal honest changes, except deliberate corruption."""
    shutil.copytree(FROZEN, root, dirs_exist_ok=True)
    _write_recorded_successors(root)
    path = root / FISHER_FIXTURE
    payload: dict[str, Any] = json.loads(path.read_text())
    route = payload["routes"]["expected" if fault == "expected" else "observed"]
    result = route["result"]
    if fault in ("ulp", "expected"):
        for field in ("fisher_standard_error", "fisher_confidence_radius"):
            value = result[field][0][0]
            for _ in range(abs(steps)):
                value = float(np.nextafter(value, np.inf if steps >= 0 else -np.inf))
            result[field] = [[value]]
    elif fault == "shape":
        result["fisher_standard_error"] = result["fisher_standard_error"][0]
    elif fault == "count":
        route["inputs"]["observed_counts"]["0"] += 1
    elif fault == "missing":
        del result["confidence_level"]
    elif fault == "extra":
        result["unexpected"] = True
    elif fault == "nonfinite":
        result["fisher_standard_error"] = [[float("inf")]]
    route["result_sha256"] = (
        codec.digest_stable_core_payload(result) if fault != "nonfinite" else "invalid"
    )
    if fault == "result_digest":
        route["result_sha256"] = "0" * 64
    if fault == "nonfinite":
        # Invalid JSON numbers must be refused, not normalized into valid evidence.
        path.write_text(json.dumps(payload))
        return
    path.write_bytes(codec.canonical_json_bytes(payload))
    for name in ("manifest.json", "manifest_studio.json"):
        manifest = json.loads((root / name).read_text())
        for row in manifest["cases"]:
            if row["fixture"] == FISHER_FIXTURE:
                row["fixture_sha256"] = codec.digest_stable_core_payload(payload)
                if fault == "fixture_digest":
                    row["fixture_sha256"] = "0" * 64
        if name == "manifest_studio.json":
            base = json.loads((root / "manifest.json").read_text())
            manifest["base_manifest_sha256"] = codec.digest_stable_core_payload(base)
            if fault == "anchor":
                manifest["base_manifest_sha256"] = "0" * 64
        (root / name).write_text(json.dumps(manifest))


@pytest.mark.parametrize("steps", (-8, -6, 0, 6, 8))
@pytest.mark.parametrize("manifest", ("manifest.json", "manifest_studio.json"))
def test_replay_accepts_only_bounded_observed_uncertainty(
    tmp_path: Path, steps: int, manifest: str
) -> None:
    """Both profiles accept resealed binary64 roundoff up to the explicit boundary."""
    _mutate_replay(tmp_path, "ulp", steps)
    assert_corpus_replay(tmp_path, FROZEN, manifest)


@pytest.mark.parametrize(
    ("fault", "steps"),
    (
        ("ulp", 9),
        ("ulp", -9),
        ("expected", 1),
        ("shape", 0),
        ("count", 0),
        ("missing", 0),
        ("extra", 0),
        ("result_digest", 0),
        ("fixture_digest", 0),
        ("anchor", 0),
    ),
)
def test_replay_rejects_scientific_or_custody_drift(
    tmp_path: Path, fault: str, steps: int
) -> None:
    """Reject just-outside roundoff, changed contracts, corrupt digests and base anchors."""
    _mutate_replay(tmp_path, fault, steps)
    with pytest.raises(AssertionError):
        assert_corpus_replay(tmp_path, FROZEN, "manifest_studio.json")


def test_replay_rejects_nonfinite_fixture(tmp_path: Path) -> None:
    """Nonfinite numerical evidence cannot pass even before checksum comparison."""
    _mutate_replay(tmp_path, "nonfinite")
    with pytest.raises((AssertionError, ValueError)):
        assert_corpus_replay(tmp_path, FROZEN, "manifest.json")


def test_recorded_successors_change_only_produced_families_and_the_plan_digest() -> None:
    """Each successor is its frozen plan plus the two families the compile verb gained."""
    successors = recorded_successors()
    assert sorted(successors) == [
        COMPILE_PLAN,
        "studio_compile_requested_rust_preserves_plan.json",
    ]
    for name, row in successors.items():
        frozen = json.loads((FROZEN / name).read_text())
        assert codec.digest_stable_core_payload(frozen["plan"]) == frozen["plan_sha256"]
        successor = json.loads(successor_bytes(frozen, row))
        kept = len(frozen["plan"]["contract"]["produces"])
        assert successor["plan"]["contract"]["produces"][kept:] == [
            "studio.program-source.v1",
            "studio.compiler-trace.v1",
        ]
        assert successor["plan_sha256"] == codec.digest_stable_core_payload(successor["plan"])
        assert successor["plan_sha256"] != frozen["plan_sha256"]
        restored = copy.deepcopy(successor)
        restored["plan"]["contract"]["produces"] = frozen["plan"]["contract"]["produces"]
        restored["plan_sha256"] = frozen["plan_sha256"]
        assert restored == frozen


@pytest.mark.parametrize("fault", ("original", "step", "family", "digest"))
def test_replay_rejects_plan_drift_other_than_the_recorded_successor(
    tmp_path: Path, fault: str
) -> None:
    """Reject the superseded original, any further plan change and a wrong digest.

    Parameters
    ----------
    tmp_path
        Directory that receives the copied corpus.
    fault
        ``original`` leaves the frozen plan in place of its successor; ``step``
        and ``family`` change the successor and reseal it honestly; ``digest``
        records a wrong digest for an unchanged successor.

    """
    shutil.copytree(FROZEN, tmp_path, dirs_exist_ok=True)
    if fault != "original":
        _write_recorded_successors(tmp_path)
        payload = json.loads((tmp_path / COMPILE_PLAN).read_text())
        if fault == "step":
            payload["plan"]["steps"][0] = "validate another network"
        elif fault == "family":
            payload["plan"]["contract"]["produces"].append("studio.unrecorded.v1")
        payload["plan_sha256"] = codec.digest_stable_core_payload(payload["plan"])
        (tmp_path / COMPILE_PLAN).write_bytes(codec.canonical_json_bytes(payload))
        _reseal_studio_row(
            tmp_path,
            COMPILE_PLAN,
            "0" * 64 if fault == "digest" else codec.digest_stable_core_payload(payload),
        )
    with pytest.raises(AssertionError):
        assert_corpus_replay(tmp_path, FROZEN, "manifest_studio.json")


@pytest.mark.parametrize("fault", ("reordered", "removed", "repeated", "unchanged", "digest"))
def test_successor_record_must_grow_the_frozen_families_and_match_its_digest(
    fault: str,
) -> None:
    """A record that reorders, removes, repeats or adds nothing, or misstates its digest, fails.

    Parameters
    ----------
    fault
        The defect introduced into a copy of the recorded successor row.

    """
    frozen = json.loads((FROZEN / COMPILE_PLAN).read_text())
    row = dict(recorded_successors()[COMPILE_PLAN])
    produces = list(row["produces"])
    if fault == "reordered":
        produces[0], produces[1] = produces[1], produces[0]
    elif fault == "removed":
        produces = produces[1:]
    elif fault == "repeated":
        produces.append(produces[-1])
    elif fault == "unchanged":
        produces = list(frozen["plan"]["contract"]["produces"])
    else:
        row["fixture_sha256"] = "0" * 64
    row["produces"] = produces
    with pytest.raises(AssertionError):
        successor_bytes(frozen, row)


@pytest.mark.parametrize("fault", ("schema", "repeated"))
def test_successor_record_rejects_another_schema_and_a_repeated_fixture(
    tmp_path: Path, fault: str
) -> None:
    """A record with an unknown schema or two rows for one fixture is refused.

    Parameters
    ----------
    tmp_path
        Directory that receives the altered record.
    fault
        The defect written into the copied record.

    """
    rows = list(recorded_successors().values())
    body = {
        "schema": "another.v1" if fault == "schema" else "contract_custody_successors.v1",
        "successors": rows + rows[:1] if fault == "repeated" else rows,
    }
    record = tmp_path / "successors.json"
    record.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(AssertionError):
        recorded_successors(record)
