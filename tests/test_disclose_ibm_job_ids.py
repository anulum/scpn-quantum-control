# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the March job-id disclosure ceremony
"""Tests for scripts/disclose_ibm_job_ids.py.

Covers the commitment digest, the label->raw-id projection (skipping non-job-id
kinds), the commitment-opening happy path and its three fail-closed branches
(missing raw id, missing nonce, digest mismatch), the CLI, and — against the
committed private records — that all 24 March commitments genuinely open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import disclose_ibm_job_ids as script

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_A, RAW_B = "rawjoba00000000", "rawjobb00000000"
NONCE_A, NONCE_B = "aa" * 16, "bb" * 16
LABEL_A, LABEL_B = "ibm-run-aaaa", "ibm-run-bbbb"


def _commitments() -> dict[str, Any]:
    return {
        "commitments": [
            {
                "public_label": LABEL_A,
                "commitment_sha256": script.commitment_digest(RAW_A, NONCE_A),
                "artefact_references": ["results/a.json"],
            },
            {
                "public_label": LABEL_B,
                "commitment_sha256": script.commitment_digest(RAW_B, NONCE_B),
                "artefact_references": [],
            },
        ]
    }


def _mapping() -> dict[str, Any]:
    return {
        "entries": [
            {"kind": "raw_ibm_job_id", "public_label": LABEL_A, "raw_value": RAW_A},
            {"kind": "raw_ibm_job_id", "public_label": LABEL_B, "raw_value": RAW_B},
            {"kind": "other_secret", "public_label": "ignore", "raw_value": "zzz"},
        ]
    }


def _nonces() -> dict[str, Any]:
    return {"nonces": {RAW_A: NONCE_A, RAW_B: NONCE_B}}


class TestCommitmentDigest:
    def test_matches_sha256_of_pair(self) -> None:
        import hashlib

        expected = hashlib.sha256(f"{RAW_A}:{NONCE_A}".encode()).hexdigest()
        assert script.commitment_digest(RAW_A, NONCE_A) == expected


class TestRawIdByLabel:
    def test_skips_non_job_id_kinds(self) -> None:
        out = script.raw_id_by_label(_mapping())
        assert out == {LABEL_A: RAW_A, LABEL_B: RAW_B}
        assert "ignore" not in out

    def test_skips_job_id_entry_with_missing_raw(self) -> None:
        mapping = {
            "entries": [
                {"kind": "raw_ibm_job_id", "public_label": LABEL_A, "raw_value": RAW_A},
                {"kind": "raw_ibm_job_id", "public_label": "ibm-run-empty", "raw_value": ""},
            ]
        }
        assert script.raw_id_by_label(mapping) == {LABEL_A: RAW_A}


class TestOpenCommitments:
    def test_opens_and_verifies_all(self) -> None:
        result = script.open_commitments(_commitments(), _mapping(), _nonces())
        assert result["opened_count"] == 2
        assert result["all_verified"] is True
        assert result["disclosures"][0]["public_label"] == LABEL_A
        assert result["disclosures"][0]["nonce"] == NONCE_A

    def test_missing_raw_id_fails(self) -> None:
        mapping = {
            "entries": [{"kind": "raw_ibm_job_id", "public_label": LABEL_A, "raw_value": RAW_A}]
        }
        with pytest.raises(ValueError, match="no raw id"):
            script.open_commitments(_commitments(), mapping, _nonces())

    def test_missing_nonce_fails(self) -> None:
        with pytest.raises(ValueError, match="no nonce"):
            script.open_commitments(_commitments(), _mapping(), {"nonces": {RAW_A: NONCE_A}})

    def test_digest_mismatch_fails(self) -> None:
        bad = {"nonces": {RAW_A: NONCE_A, RAW_B: "ff" * 16}}
        with pytest.raises(ValueError, match="does not open"):
            script.open_commitments(_commitments(), _mapping(), bad)


class TestCli:
    def test_writes_disclosure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / "commitments.json").write_text(json.dumps(_commitments()), encoding="utf-8")
        (tmp_path / "mapping.json").write_text(json.dumps(_mapping()), encoding="utf-8")
        (tmp_path / "nonces.json").write_text(json.dumps(_nonces()), encoding="utf-8")
        out = tmp_path / "disclosure.json"
        rc = script.main(
            [
                "--commitments",
                str(tmp_path / "commitments.json"),
                "--mapping",
                str(tmp_path / "mapping.json"),
                "--nonces",
                str(tmp_path / "nonces.json"),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["opened_count"] == 2
        assert "opened 2 commitments" in capsys.readouterr().out


class TestCommittedDisclosure:
    """Referee re-verification of the committed public disclosure.

    Reads only public tracked files (the disclosure + the commitments), never
    the git-ignored private mapping/nonce sidecar, so it runs in CI. This is
    exactly the check an external referee performs.
    """

    def test_disclosure_opens_every_committed_commitment(self) -> None:
        base = REPO_ROOT / "data" / "march_flagship_verifiability"
        disclosure = json.loads(
            (base / "march_job_id_disclosure_2026-07-18.json").read_text(encoding="utf-8")
        )
        commitments = json.loads(
            (base / "march_job_id_commitments.json").read_text(encoding="utf-8")
        )
        committed_digests = {
            c["public_label"]: c["commitment_sha256"] for c in commitments["commitments"]
        }
        assert disclosure["opened_count"] == len(committed_digests) == 24
        assert disclosure["all_verified"] is True
        for row in disclosure["disclosures"]:
            # recompute the commitment from the disclosed (raw id, nonce) pair
            recomputed = script.commitment_digest(row["raw_ibm_job_id"], row["nonce"])
            assert recomputed == row["commitment_sha256"]
            # and confirm it matches the independently-committed commitment file
            assert recomputed == committed_digests[row["public_label"]]
