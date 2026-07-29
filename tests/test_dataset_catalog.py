# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the application benchmark catalogue
"""Tests for the packaged application-benchmark dataset catalogue.

Covers descriptor lookup (valid and unknown identifiers) and the
load-and-validate contract that rejects artifacts whose source name or domain
disagrees with the packaged descriptor.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import scpn_quantum_control.applications.dataset_catalog as catalog
from scpn_quantum_control.applications.dataset_catalog import (
    ApplicationBenchmarkPrivacyAudit,
    audit_application_benchmark_privacy,
    get_application_benchmark_descriptor,
    list_application_benchmark_descriptors,
    load_application_benchmark_artifact,
)


def test_descriptors_are_packaged() -> None:
    """The catalogue exposes the packaged benchmark descriptors."""
    descriptors = list_application_benchmark_descriptors()
    ids = {d.dataset_id for d in descriptors}
    assert "eeg_alpha_plv_8ch" in ids
    assert all(d.path.name == d.filename for d in descriptors)


def test_get_descriptor_by_id() -> None:
    """A known dataset id resolves to its descriptor."""
    descriptor = get_application_benchmark_descriptor("eeg_alpha_plv_8ch")
    assert descriptor.domain == "eeg"


def test_get_descriptor_rejects_unknown_id() -> None:
    """An unknown dataset id fails closed and lists the known ids."""
    with pytest.raises(KeyError, match="unknown application benchmark dataset"):
        get_application_benchmark_descriptor("does_not_exist")


def test_load_rejects_source_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded artifact whose source_name disagrees with the descriptor is rejected."""

    def fake_read(_path: Any) -> Any:
        return SimpleNamespace(source_name="unexpected", domain="eeg")

    monkeypatch.setattr(catalog, "read_qpu_data_artifact", fake_read)
    with pytest.raises(ValueError, match="source_name"):
        load_application_benchmark_artifact("eeg_alpha_plv_8ch")


def test_load_rejects_domain_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded artifact whose domain disagrees with the descriptor is rejected."""

    def fake_read(_path: Any) -> Any:
        return SimpleNamespace(source_name="eeg_alpha_plv_8ch", domain="unexpected")

    monkeypatch.setattr(catalog, "read_qpu_data_artifact", fake_read)
    with pytest.raises(ValueError, match="domain"):
        load_application_benchmark_artifact("eeg_alpha_plv_8ch")


def _matching_fake_artifact(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "source_name": "eeg_alpha_plv_8ch",
        "domain": "eeg",
        "source_mode": "curated",
        "metadata": {
            "licence_note": (
                "small curated benchmark matrix for software validation; "
                "not a raw participant recording"
            )
        },
        "hashes": {"K_nm_sha256": "a" * 64},
        "require_publication_safe": lambda: None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_rejects_source_mode_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provenance-mode mismatch fails before publication-safe admission."""
    monkeypatch.setattr(
        catalog,
        "read_qpu_data_artifact",
        lambda _path: _matching_fake_artifact(source_mode="private"),
    )
    with pytest.raises(ValueError, match="source_mode"):
        load_application_benchmark_artifact("eeg_alpha_plv_8ch")


def test_load_rejects_personal_data_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built-in loader cannot admit a descriptor marked as personal data."""
    descriptor = get_application_benchmark_descriptor("eeg_alpha_plv_8ch")
    monkeypatch.setattr(
        catalog,
        "_DESCRIPTORS",
        (replace(descriptor, contains_personal_data=True),),
    )
    monkeypatch.setattr(catalog, "read_qpu_data_artifact", lambda _path: _matching_fake_artifact())
    with pytest.raises(ValueError, match="may not contain personal data"):
        load_application_benchmark_artifact("eeg_alpha_plv_8ch")


def test_load_rejects_privacy_boundary_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact packaged licence/privacy note is descriptor-bound."""
    monkeypatch.setattr(
        catalog,
        "read_qpu_data_artifact",
        lambda _path: _matching_fake_artifact(metadata={"licence_note": "unknown"}),
    )
    with pytest.raises(ValueError, match="privacy boundary"):
        load_application_benchmark_artifact("eeg_alpha_plv_8ch")


def test_privacy_audit_covers_every_packaged_descriptor() -> None:
    """The privacy audit validates all packaged rows and exposes defensive hashes."""
    rows = audit_application_benchmark_privacy()
    assert {row.dataset_id for row in rows} == {
        descriptor.dataset_id for descriptor in list_application_benchmark_descriptors()
    }
    assert all(row.passed and not row.contains_personal_data for row in rows)
    payload = rows[0].as_dict()
    payload["artifact_hashes"]["mutated"] = "yes"
    assert "mutated" not in rows[0].artifact_hashes


def test_privacy_audit_record_can_represent_a_failed_external_check() -> None:
    """The public record type retains an explicit result for aggregate reports."""
    row = ApplicationBenchmarkPrivacyAudit(
        dataset_id="external",
        source_mode="curated",
        privacy_classification="external_review",
        contains_personal_data=False,
        privacy_boundary="review required",
        artifact_hashes={},
        passed=False,
    )
    assert row.as_dict()["passed"] is False
