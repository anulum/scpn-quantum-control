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

from types import SimpleNamespace
from typing import Any

import pytest

import scpn_quantum_control.applications.dataset_catalog as catalog
from scpn_quantum_control.applications.dataset_catalog import (
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
