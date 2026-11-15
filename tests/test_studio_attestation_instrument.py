# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — attestation instrument data against independent oracles
"""Check the attestation instrument data, its oracles, its refusals and its CLI."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import studio_attestation_instrument as instrument

EXPECTED_STATUSES = {
    "intact": "VERIFIED",
    "changed-value": "FORGED",
    "rebound-axes": "FORGED",
    "stripped-axes": "STRIPPED",
    "missing-digest": "UNGRADED",
    "unverified-signature": "UNGRADED",
    "signature-required": "UNGRADED",
}


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Build the instrument once from its owner and the pinned summary.

    Returns
    -------
    dict
        The fresh artefact payload.

    """
    return instrument.build_attestation_instrument_artifact()


def _bound_digest(axes: object, content: object) -> str:
    """Digest of the documented canonical form, written out independently."""
    text = json.dumps(
        {"content": content, "claim_axes": axes},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_attestation_instrument_01_cases_carry_the_owner_verdicts(built: dict[str, Any]) -> None:
    """Each case has the expected status, and the sealed digest covers the pinned bytes.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    cases = {case["id"]: case for case in built["cases"]}
    assert {
        key: case["python_verdict"]["status"] for key, case in cases.items()
    } == EXPECTED_STATUSES
    source = json.loads((instrument.ROOT / instrument.SOURCE_PATH).read_bytes())
    sealed = cases["intact"]["envelope"]
    assert sealed["content"] == source
    assert sealed["content_digest"] == _bound_digest(sealed["claim_axes"], source)
    assert cases["changed-value"]["envelope"]["content_digest"] == sealed["content_digest"]
    assert cases["changed-value"]["envelope"]["content"]["A_even"]["mean"] == 0.9285
    assert cases["rebound-axes"]["python_verdict"]["expected_digest"] == _bound_digest(
        cases["rebound-axes"]["envelope"]["claim_axes"], source
    )
    assert cases["stripped-axes"]["envelope"]["claim_axes"]["honesty_axes"] == []
    assert cases["missing-digest"]["envelope"]["content_digest"] == ""
    assert cases["unverified-signature"]["envelope"]["signature"] != ""
    assert cases["signature-required"]["require_signature"] is True
    assert "aggregate_summary_not_raw_counts" in sealed["claim_axes"]["honesty_axes"]


def test_attestation_instrument_02_oracle_text_is_the_owner_digest_input(
    built: dict[str, Any],
) -> None:
    """Every recorded canonical text hashes to the owner's digest of its content.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    labels = [row["label"] for row in built["canonical_oracle"]]
    assert labels == [
        "edge floats",
        "seeded random floats",
        "integers",
        "strings",
        "key order",
        "nesting and literals",
    ]
    for row in built["canonical_oracle"]:
        assert hashlib.sha256(row["python_text"].encode("utf-8")).hexdigest() == row["digest"]
        assert row["digest"] == _bound_digest({}, row["content"])
    floats = built["canonical_oracle"][1]["content"]["value"]
    assert len(floats) == 64 and all(math.isfinite(value) for value in floats)
    assert '"\uffff":1' in built["canonical_oracle"][4]["python_text"]
    assert built["canonical_oracle"][4]["python_text"].index("￿") < built["canonical_oracle"][4][
        "python_text"
    ].index("\U0001d538")


def test_attestation_instrument_03_whitespace_set_is_pythons(built: dict[str, Any]) -> None:
    """The recorded set is exactly the code points Python strips.

    Parameters
    ----------
    built
        Fresh artefact payload.

    """
    recorded = built["python_whitespace"]
    assert recorded == sorted(set(recorded))
    assert all(chr(code).strip() == "" for code in recorded)
    assert {0x1C, 0x85, 0x3000} <= set(recorded)
    assert sum(1 for code in range(0x110000) if chr(code).isspace()) == len(recorded)


def test_attestation_instrument_04_changed_source_is_refused(tmp_path: Path) -> None:
    """A summary whose bytes differ from the pin is refused before any envelope.

    Parameters
    ----------
    tmp_path
        Owned copy of the repository path holding the summary.

    """
    target = tmp_path / instrument.SOURCE_PATH
    target.parent.mkdir(parents=True)
    shutil.copyfile(instrument.ROOT / instrument.SOURCE_PATH, target)
    assert instrument.load_source(tmp_path) == json.loads(target.read_bytes())
    target.write_bytes(target.read_bytes().replace(b"0.9185", b"0.9186", 1))
    with pytest.raises(RuntimeError, match="changed: sha256"):
        instrument.build_attestation_instrument_artifact(tmp_path)


def test_committed_data_and_command_line_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, check and refuse changed data through the public command line.

    Parameters
    ----------
    tmp_path
        Owned scratch output.
    monkeypatch
        Supplies the real command's argument vector.

    """
    assert instrument.main(["--check"]) == 0
    target = tmp_path / "attestation.json"
    assert instrument.main(["--write", "--json-path", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["cases"][0]["python_verdict"]["status"] = "FORGED"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert instrument.main(["--check", "--json-path", str(target)]) == 1
    monkeypatch.setattr(sys, "argv", ["attestation", "--check"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(
            str(instrument.ROOT / "tools/studio_attestation_instrument.py"), run_name="__main__"
        )
    assert exit_status.value.code == 0
