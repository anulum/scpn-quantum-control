# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded corpus replay regression tests
"""Prove that numerical replay never excuses corrupted custody or input drift."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _contract_custody_replay_helpers import FISHER_FIXTURE, assert_corpus_replay

from scpn_quantum_control import stable_core_product as codec

FROZEN = Path(__file__).parent / "data" / "contract_custody_corpus"


def _mutate_replay(root: Path, fault: str, steps: int = 0) -> None:
    """Alter one copied fixture and reseal honest changes, except deliberate corruption."""
    shutil.copytree(FROZEN, root, dirs_exist_ok=True)
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
