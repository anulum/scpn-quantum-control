# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-owned sensing metadata and private export boundaries
"""Exercise public generation, analytic scores, metadata export and CLI drift refusal."""

from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.sensing.domain_evidence import EntropyEvidence
from scpn_quantum_control.studio import sensing_instrument_artifact as artifact


def test_generated_scores_match_the_independent_gaussian_update() -> None:
    """Source-owned scores and uncertainty match the analytic conjugate update."""
    payload = artifact.build_sensing_instrument_artifact()
    rows: Any = payload["candidates"]
    for row in rows:
        snr = 32 * row["prior_variance"] * row["sensitivity"] ** 2 / row["noise_variance"]
        assert row["expected_information_gain_nats"] == pytest.approx(math.log1p(snr) / 2)
        assert row["posterior_variance"] == pytest.approx(row["prior_variance"] / (1 + snr))
    entropy: Any = payload["entropy"]
    assert entropy[0]["source_kind"] == "synthetic"
    assert entropy[0]["diagnostics"]["signature_valid"] is True
    assert entropy[1]["source_kind"] == "device" and entropy[1]["diagnostics"] is None
    assert entropy[1]["source"] is None
    assert all(row["source_entropy_qualified"] is False for row in entropy)


def test_public_entropy_export_excludes_private_material_and_claim_promotion() -> None:
    """Passing diagnostics cannot promote source entropy or export unlisted private data."""
    exported = artifact.project_entropy_metadata(
        {
            "source_kind": "device",
            "case_id": "external-observation",
            "seed": "PRIVATE_SEED",
            "private_key": "PRIVATE_KEY",
            "samples": "PRIVATE_SAMPLES",
            "extra": "PRIVATE_EXTRA",
        },
        EntropyEvidence(True, True, True),
    )
    assert exported["source_entropy_qualified"] is False
    assert "PRIVATE_" not in json.dumps(exported)
    assert exported["source"] is None and exported["extractor"] is None
    assert artifact.project_entropy_metadata({}, None)["source_kind"] == "unknown"


@pytest.mark.parametrize(
    "metadata", [{"source_kind": "certified"}, {"source": 7}, {"source": " "}]
)
def test_invalid_public_provenance_refuses(metadata: dict[str, object]) -> None:
    """Reject malformed public fields without invoking any provider.

    Parameters
    ----------
    metadata
        Invalid source declarations.

    """
    with pytest.raises(ValueError):
        artifact.project_entropy_metadata(metadata, None)


def test_committed_metadata_and_public_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse changed metadata through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert artifact.main(["--check"]) == 0
    target = tmp_path / "metadata.json"
    assert artifact.main(["--write", "--json-path", str(target)]) == 0
    assert artifact.main(["--check", "--json-path", str(target)]) == 0
    target.write_text("{}")
    assert artifact.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["sensing_instrument_artifact", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(artifact.__name__, run_name="__main__")
    assert stopped.value.code == 0
