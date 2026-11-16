# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the March retrieval-receipts script
"""Tests for scripts/retrieve_march_flagship_receipts.py.

Exercises commitment-file validation, digest cross-checks before any
retrieval, service authentication kwargs, usage-metric extraction, the
error-class-only failure receipts, the fail-closed leak check (raw ids,
nonces, API token), and the CLI against fully stubbed services — no
network or real identifiers anywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts import build_march_job_id_commitments as builder
from scripts import retrieve_march_flagship_receipts as script

RAW_A = "fakejobid00000000aaa"
RAW_B = "fakejobid00000000bbb"
LABEL_A = "ibm-run-00000000000000a1"
LABEL_B = "ibm-run-00000000000000b2"
NONCES = {RAW_A: "aa" * 16, RAW_B: "bb" * 16}


def make_commitments() -> dict[str, Any]:
    """Return a valid commitments payload for the two test labels."""
    return {
        "schema": builder.COMMITMENT_SCHEMA,
        "generated_utc": "2026-07-17T00:00:00Z",
        "commitments": [
            {
                "public_label": LABEL_A,
                "commitment_sha256": builder.commitment_digest(RAW_A, NONCES[RAW_A]),
                "artefact_references": [],
            },
            {
                "public_label": LABEL_B,
                "commitment_sha256": builder.commitment_digest(RAW_B, NONCES[RAW_B]),
                "artefact_references": [],
            },
        ],
    }


def make_manifest() -> dict[str, Any]:
    """Return a valid private-mapping manifest for the two test labels."""
    return {
        "schema": builder.MAPPING_SCHEMA,
        "label_prefix": "ibm-run",
        "salt_sha256": "ab" * 32,
        "entries": [
            {
                "kind": "raw_ibm_job_id",
                "raw_value": raw,
                "public_label": label,
                "path": "results/ibm_hardware_2026-03-28/x.json",
                "json_pointer": "/job_id",
                "field": "job_id",
            }
            for raw, label in ((RAW_A, LABEL_A), (RAW_B, LABEL_B))
        ],
    }


class FakeBackend:
    """A provider backend that names itself."""

    name = "ibm_fez"


class FakeJob:
    """A provider job with a set status and metrics."""

    def __init__(
        self,
        *,
        status: str = "DONE",
        creation: datetime | None = datetime(2026, 3, 28, 12, 0, tzinfo=UTC),
        metrics: Any = None,
        metrics_error: bool = False,
    ) -> None:
        self._status = status
        self.creation_date = creation
        self._metrics = {"usage": {"seconds": 33.5}} if metrics is None else metrics
        self._metrics_error = metrics_error

    def backend(self) -> FakeBackend:
        """Return the job's backend."""
        return FakeBackend()

    def status(self) -> str:
        """Return the job's status."""
        return self._status

    def metrics(self) -> Any:
        """Return the job's metrics, or raise when set to fail."""
        if self._metrics_error:
            raise RuntimeError(f"metrics unavailable for {RAW_A}")
        return self._metrics


class FakeService:
    """A runtime service holding known jobs."""

    def __init__(self, jobs: dict[str, FakeJob]) -> None:
        self._jobs = jobs

    def job(self, raw_id: str) -> FakeJob:
        """Return the job for a raw id, or raise KeyError for an unknown one."""
        if raw_id not in self._jobs:
            raise KeyError(f"no job {raw_id}")
        return self._jobs[raw_id]


class TestLoadCommitments:
    """Loading the public commitments."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """Valid commitments load."""
        path = tmp_path / "commitments.json"
        path.write_text(json.dumps(make_commitments()), encoding="utf-8")
        assert script.load_commitments(path)["schema"] == builder.COMMITMENT_SCHEMA

    def test_rejects_wrong_schema(self, tmp_path: Path) -> None:
        """Commitments with the wrong schema are refused."""
        path = tmp_path / "commitments.json"
        path.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
        with pytest.raises(ValueError, match="is not a"):
            script.load_commitments(path)

    def test_rejects_empty_commitments(self, tmp_path: Path) -> None:
        """Commitments without entries are refused."""
        payload = make_commitments()
        payload["commitments"] = []
        path = tmp_path / "commitments.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="carries no commitments"):
            script.load_commitments(path)


class TestLoadNonces:
    """Loading the nonce sidecar."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """A valid sidecar loads its nonces."""
        path = tmp_path / "nonces.json"
        path.write_text(
            json.dumps({"schema": builder.NONCE_SIDECAR_SCHEMA, "nonces": NONCES}),
            encoding="utf-8",
        )
        assert script.load_nonces(path) == NONCES

    def test_malformed_fails_closed(self, tmp_path: Path) -> None:
        """A malformed sidecar fails closed."""
        path = tmp_path / "nonces.json"
        path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            script.load_nonces(path)


class TestResolveRawIds:
    """Resolving labels to raw job ids."""

    def test_happy_path_verifies_every_digest(self) -> None:
        """Every label resolves after its digest verifies."""
        resolved = script.resolve_raw_ids(make_commitments(), make_manifest(), NONCES)
        assert resolved == {LABEL_A: RAW_A, LABEL_B: RAW_B}

    def test_label_absent_from_mapping(self) -> None:
        """A label missing from the mapping is refused."""
        manifest = make_manifest()
        manifest["entries"] = manifest["entries"][:1]
        with pytest.raises(ValueError, match=f"{LABEL_B} is absent"):
            script.resolve_raw_ids(make_commitments(), manifest, NONCES)

    def test_missing_nonce(self) -> None:
        """A label without a nonce is refused."""
        with pytest.raises(ValueError, match=f"{LABEL_B} has no nonce"):
            script.resolve_raw_ids(make_commitments(), make_manifest(), {RAW_A: NONCES[RAW_A]})

    def test_digest_mismatch_refuses_retrieval(self) -> None:
        """A digest mismatch refuses retrieval."""
        commitments = make_commitments()
        commitments["commitments"][0]["commitment_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="digest mismatch"):
            script.resolve_raw_ids(commitments, make_manifest(), NONCES)


class TestLoadRuntimeService:
    """Choosing the runtime-service credentials."""

    def install_fake_runtime(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Install a fake runtime service and return the keyword arguments it receives."""
        captured: dict[str, Any] = {}

        class FakeRuntimeService:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        module = types.ModuleType("qiskit_ibm_runtime")
        module.QiskitRuntimeService = FakeRuntimeService  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", module)
        return captured

    def test_vault_credentials_select_ibm_cloud_channel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Vault credentials select the cloud channel with token and instance."""
        captured = self.install_fake_runtime(monkeypatch)
        from scripts import prepare_s1_ibm_live_readiness as readiness

        monkeypatch.setattr(readiness, "_parse_vault", lambda path: ("tok", "crn-instance"))
        script.load_runtime_service(None, tmp_path / "vault.md")
        assert captured == {"channel": "ibm_cloud", "token": "tok", "instance": "crn-instance"}

    def test_explicit_instance_overrides_vault(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An explicit instance overrides the vault's."""
        captured = self.install_fake_runtime(monkeypatch)
        from scripts import prepare_s1_ibm_live_readiness as readiness

        monkeypatch.setattr(readiness, "_parse_vault", lambda path: ("tok", "crn-instance"))
        script.load_runtime_service("crn-other", tmp_path / "vault.md")
        assert captured["instance"] == "crn-other"

    def test_no_vault_uses_saved_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without vault credentials the saved account is used."""
        captured = self.install_fake_runtime(monkeypatch)
        script.load_runtime_service(None, None)
        assert captured == {}


class TestUsageSeconds:
    """Reading the usage seconds of a job."""

    def test_seconds_extracted(self) -> None:
        """Usage seconds are read from the metrics."""
        assert script._usage_seconds(FakeJob()) == 33.5

    def test_quantum_seconds_fallback(self) -> None:
        """Quantum seconds are used when plain seconds are missing."""
        job = FakeJob(metrics={"usage": {"quantum_seconds": 12}})
        assert script._usage_seconds(job) == 12.0

    def test_metrics_error_returns_none(self) -> None:
        """A metrics error gives no usage."""
        assert script._usage_seconds(FakeJob(metrics_error=True)) is None

    def test_non_mapping_metrics_returns_none(self) -> None:
        """Metrics that are not a mapping give no usage."""
        assert script._usage_seconds(FakeJob(metrics=["not-a-mapping"])) is None

    def test_non_numeric_usage_returns_none(self) -> None:
        """A non-numeric usage gives none."""
        assert script._usage_seconds(FakeJob(metrics={"usage": {"seconds": "n/a"}})) is None


class TestJobReceipt:
    """Receipts for single jobs."""

    def test_ok_receipt_carries_public_fields_only(self) -> None:
        """A good receipt carries only public fields."""
        service = FakeService({RAW_A: FakeJob()})
        receipt = script.job_receipt(service, LABEL_A, "c" * 64, RAW_A)
        assert receipt["retrieval"] == "ok"
        assert receipt["backend"] == "ibm_fez"
        assert receipt["status"] == "DONE"
        assert receipt["creation_date_utc"] == "2026-03-28T12:00:00+00:00"
        assert receipt["usage_seconds"] == 33.5
        assert RAW_A not in json.dumps(receipt)

    def test_missing_creation_date(self) -> None:
        """A job without a creation date gets none in its receipt."""
        service = FakeService({RAW_A: FakeJob(creation=None)})
        receipt = script.job_receipt(service, LABEL_A, "c" * 64, RAW_A)
        assert receipt["creation_date_utc"] is None

    def test_error_receipt_records_class_name_only(self) -> None:
        """An error receipt records the error class and never the raw id."""
        receipt = script.job_receipt(FakeService({}), LABEL_A, "c" * 64, RAW_A)
        assert receipt["retrieval"] == "error"
        assert receipt["error_type"] == "KeyError"
        assert RAW_A not in json.dumps(receipt)


class TestBuildReceipts:
    """Building the receipts payload."""

    def test_counts_and_ordering(self) -> None:
        """The payload has its schema, counts and label order."""
        service = FakeService({RAW_A: FakeJob(), RAW_B: FakeJob()})
        payload = script.build_receipts(
            service,
            make_commitments(),
            {LABEL_A: RAW_A, LABEL_B: RAW_B},
            "f" * 64,
            "2026-07-17T01:00:00Z",
        )
        assert payload["schema"] == script.RECEIPTS_SCHEMA
        assert payload["receipt_count"] == 2
        assert payload["ok_count"] == 2
        assert payload["commitments_sha256"] == "f" * 64
        labels = [r["public_label"] for r in payload["receipts"]]
        assert labels == sorted(labels)

    def test_partial_failure_counted(self) -> None:
        """A failed retrieval lowers the good count."""
        service = FakeService({RAW_A: FakeJob()})
        payload = script.build_receipts(
            service,
            make_commitments(),
            {LABEL_A: RAW_A, LABEL_B: RAW_B},
            "f" * 64,
            "2026-07-17T01:00:00Z",
        )
        assert payload["ok_count"] == 1


class TestLeakCheck:
    """Checking the payload for leaks."""

    def test_clean_payload_passes(self) -> None:
        """A clean payload passes."""
        script.assert_no_private_leak("labels only", {LABEL_A: RAW_A}, NONCES, "tok")

    def test_raw_id_leak_is_fatal(self) -> None:
        """A raw job id in the payload is fatal."""
        with pytest.raises(RuntimeError, match=f"raw job id for {LABEL_A}"):
            script.assert_no_private_leak(RAW_A, {LABEL_A: RAW_A}, {}, None)

    def test_nonce_leak_is_fatal(self) -> None:
        """A nonce in the payload is fatal."""
        with pytest.raises(RuntimeError, match="nonce leaked"):
            script.assert_no_private_leak(NONCES[RAW_A], {}, NONCES, None)

    def test_token_leak_is_fatal(self) -> None:
        """The API token in the payload is fatal."""
        with pytest.raises(RuntimeError, match="API token leaked"):
            script.assert_no_private_leak("text secret-token text", {}, {}, "secret-token")

    def test_absent_token_is_skipped(self) -> None:
        """Without a token there is nothing to check for it."""
        script.assert_no_private_leak("anything", {}, {}, None)


class TestMain:
    """The retrieval command."""

    def write_inputs(self, tmp_path: Path) -> dict[str, Path]:
        """Write the commitments, mapping and nonce inputs and return their paths."""
        paths = {
            "commitments": tmp_path / "commitments.json",
            "mapping": tmp_path / "mapping.json",
            "nonces": tmp_path / "nonces.json",
            "vault": tmp_path / "vault.md",
            "outdir": tmp_path / "receipts",
        }
        paths["commitments"].write_text(json.dumps(make_commitments()), encoding="utf-8")
        paths["mapping"].write_text(json.dumps(make_manifest()), encoding="utf-8")
        paths["nonces"].write_text(
            json.dumps({"schema": builder.NONCE_SIDECAR_SCHEMA, "nonces": NONCES}),
            encoding="utf-8",
        )
        paths["vault"].write_text("### IBM Quantum\n", encoding="utf-8")
        return paths

    def run_main(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: dict[str, FakeJob]
    ) -> tuple[int, Path]:
        """Run the command against fake jobs and return its exit code and output directory."""
        paths = self.write_inputs(tmp_path)
        monkeypatch.setattr(
            script, "load_runtime_service", lambda instance, vault: FakeService(jobs)
        )
        code = script.main(
            [
                "--commitments",
                str(paths["commitments"]),
                "--private-mapping",
                str(paths["mapping"]),
                "--nonces",
                str(paths["nonces"]),
                "--credentials-vault",
                str(paths["vault"]),
                "--output-dir",
                str(paths["outdir"]),
            ]
        )
        return code, paths["outdir"]

    def test_happy_path_writes_receipts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A normal run writes one receipts file with every job good."""
        code, outdir = self.run_main(tmp_path, monkeypatch, {RAW_A: FakeJob(), RAW_B: FakeJob()})
        assert code == 0
        outputs = list(outdir.glob("march_retrieval_receipts_*.json"))
        assert len(outputs) == 1
        payload = json.loads(outputs[0].read_text(encoding="utf-8"))
        assert payload["ok_count"] == 2
        assert (
            payload["commitments_sha256"]
            == hashlib.sha256((tmp_path / "commitments.json").read_bytes()).hexdigest()
        )
        text = outputs[0].read_text(encoding="utf-8")
        assert RAW_A not in text and RAW_B not in text
        assert "receipts: 2/2 retrieved ok" in capsys.readouterr().out

    def test_partial_failure_returns_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed retrieval makes the run exit 1 and marks that label."""
        code, outdir = self.run_main(tmp_path, monkeypatch, {RAW_A: FakeJob()})
        assert code == 1
        payload = json.loads(
            next(iter(outdir.glob("march_retrieval_receipts_*.json"))).read_text(encoding="utf-8")
        )
        assert payload["ok_count"] == 1
        by_label = {r["public_label"]: r for r in payload["receipts"]}
        assert by_label[LABEL_B]["retrieval"] == "error"
