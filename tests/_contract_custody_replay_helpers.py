# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — frozen custody and numerical replay assertions
"""Separate exact stored-byte custody from bounded floating-point recomputation.

The observed-count Fisher replay differed by six binary64 ULPs across the
host and Docker numerical environments. Permit at most eight ULPs for its two
1x1 uncertainty outputs only; all other results, inputs and metadata stay exact.
This is a corpus regression budget, not a scientific accuracy guarantee.

A frozen fixture is never rewritten. When a producer changes on purpose, the
change is recorded in ``data/contract_custody_successors.json`` and the replay
derives the expected successor from the frozen fixture and that record, then
compares it with the fresh output byte for byte. The only recorded kind of
change is growth of the produced families in a plan's verb contract.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control import stable_core_product as codec

FISHER_FIXTURE = "fisher_observed_and_expected_routes_stay_distinct.json"
SUCCESSOR_RECORD = Path(__file__).parent / "data" / "contract_custody_successors.json"
SUCCESSOR_SCHEMA = "contract_custody_successors.v1"


def recorded_successors(record: Path = SUCCESSOR_RECORD) -> dict[str, dict[str, Any]]:
    """Return the recorded successors of frozen fixtures, keyed by fixture name.

    Parameters
    ----------
    record
        Successor record to read.

    Returns
    -------
    dict
        One row per frozen fixture that has a recorded successor.

    """
    body = json.loads(record.read_text(encoding="utf-8"))
    assert body["schema"] == SUCCESSOR_SCHEMA
    rows: list[dict[str, Any]] = body["successors"]
    names = [row["fixture"] for row in rows]
    assert len(names) == len(set(names))
    return dict(zip(names, rows, strict=True))


def successor_bytes(frozen: Mapping[str, Any], row: Mapping[str, Any]) -> bytes:
    """Derive the successor of a frozen plan fixture from its recorded contract growth.

    Parameters
    ----------
    frozen
        Original captured plan fixture, never modified.
    row
        Recorded successor: the grown list of produced families and the digest
        of the whole successor fixture.

    Returns
    -------
    bytes
        Canonical bytes of the successor. It equals the frozen fixture except
        for the produced families, which must keep the frozen ones first and
        add at least one distinct family, and the plan digest recomputed over
        the changed plan. Its digest must be the recorded one.

    """
    original: list[str] = frozen["plan"]["contract"]["produces"]
    grown: list[str] = row["produces"]
    assert len(grown) > len(original)
    assert grown[: len(original)] == original
    assert len(set(grown)) == len(grown)
    successor = copy.deepcopy(dict(frozen))
    successor["plan"]["contract"]["produces"] = list(grown)
    successor["plan_sha256"] = codec.digest_stable_core_payload(successor["plan"])
    assert codec.digest_stable_core_payload(successor) == row["fixture_sha256"]
    return codec.canonical_json_bytes(successor)


def assert_fisher_result(actual: dict[str, Any], frozen: dict[str, Any], route: str) -> None:
    """Compare complete Fisher results with narrowly bounded observed uncertainty.

    Parameters
    ----------
    actual
        Fresh public producer result.
    frozen
        Original captured result, never modified.
    route
        Only ``observed`` permits the two documented scalar differences.

    """
    normalized = copy.deepcopy(actual)
    if route == "observed":
        for field in ("fisher_standard_error", "fisher_confidence_radius"):
            left, right = np.asarray(actual[field]), np.asarray(frozen[field])
            assert left.shape == right.shape == (1, 1)
            assert left.dtype == right.dtype == np.dtype("float64")
            assert np.isfinite(left).all() and np.isfinite(right).all()
            np.testing.assert_array_max_ulp(left, right, maxulp=8)
            normalized[field] = frozen[field]
    assert normalized == frozen


def _assert_fisher_payload(actual: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Verify each route's own digest before comparing its complete result."""
    normalized = copy.deepcopy(actual)
    for route in ("expected", "observed"):
        left, right = actual["routes"][route], frozen["routes"][route]
        for capture in (left, right):
            assert codec.digest_stable_core_payload(capture["result"]) == capture["result_sha256"]
        assert_fisher_result(left["result"], right["result"], route)
        normalized["routes"][route]["result"] = right["result"]
        normalized["routes"][route]["result_sha256"] = right["result_sha256"]
    assert normalized == frozen


def assert_corpus_replay(actual_root: Path, frozen_root: Path, manifest_name: str) -> None:
    """Verify every fixture digest and metadata field before numerical replay.

    Parameters
    ----------
    actual_root
        Freshly generated corpus directory.
    frozen_root
        Original committed corpus directory.
    manifest_name
        Base or Studio manifest filename. Studio must bind its own base bytes.

    """
    actual = json.loads((actual_root / manifest_name).read_text())
    frozen = json.loads((frozen_root / manifest_name).read_text())
    normalized = copy.deepcopy(actual)
    successors = recorded_successors()
    assert len(actual["cases"]) == len(frozen["cases"])
    for index, (left, right) in enumerate(zip(actual["cases"], frozen["cases"], strict=True)):
        assert left["fixture"] == right["fixture"]
        filename = left["fixture"]
        if filename is None:
            assert left == right
            continue
        left_bytes = (actual_root / filename).read_bytes()
        right_bytes = (frozen_root / filename).read_bytes()
        left_payload, right_payload = json.loads(left_bytes), json.loads(right_bytes)
        for payload, row, raw in (
            (left_payload, left, left_bytes),
            (right_payload, right, right_bytes),
        ):
            assert codec.canonical_json_bytes(payload) == raw
            assert codec.digest_stable_core_payload(payload) == row["fixture_sha256"]
        if filename == FISHER_FIXTURE:
            _assert_fisher_payload(left_payload, right_payload)
            normalized["cases"][index]["fixture_sha256"] = right["fixture_sha256"]
        elif filename in successors:
            assert left_bytes == successor_bytes(right_payload, successors[filename])
            normalized["cases"][index]["fixture_sha256"] = right["fixture_sha256"]
        else:
            assert left_bytes == right_bytes
    if manifest_name == "manifest_studio.json":
        assert set(successors) <= {row["fixture"] for row in frozen["cases"]}
        assert_corpus_replay(actual_root, frozen_root, "manifest.json")
        for root, manifest in ((actual_root, actual), (frozen_root, frozen)):
            base = json.loads((root / "manifest.json").read_text())
            assert manifest["base_manifest_sha256"] == codec.digest_stable_core_payload(base)
            assert manifest["cases"][: len(base["cases"])] == base["cases"]
        normalized["base_manifest_sha256"] = frozen["base_manifest_sha256"]
    assert normalized == frozen
