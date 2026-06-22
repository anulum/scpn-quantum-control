# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the application plugins
"""Guard and branch tests for the application benchmark plugin layer.

Covers the dataset domain-mismatch guard, the entry-point discovery wrapper, the
plugin protocol type guard, the finite-metric guard and the metadata-array
missing/shape/finiteness guards.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.applications.app_plugins import (
    EEGApplicationPlugin,
    _finite_metric,
    _metadata_array,
    _validate_plugin,
    discover_application_plugins,
)
from scpn_quantum_control.bridge import artifact_from_arrays
from scpn_quantum_control.bridge.qpu_data_artifact import QPUDataArtifact


def _artifact(metadata: dict[str, object] | None = None) -> QPUDataArtifact:
    return artifact_from_arrays(
        domain="unit",
        source_name="unit-source",
        source_mode="curated",
        K_nm=np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        normalization="unit",
        extraction_method="unit-test",
        metadata=metadata,
    )


class _WrongDomainPlugin(EEGApplicationPlugin):
    domain = "definitely_not_the_packaged_domain"


def test_load_dataset_rejects_domain_mismatch() -> None:
    """A plugin whose declared domain disagrees with the artifact is rejected."""
    with pytest.raises(ValueError, match="expected 'definitely_not_the_packaged_domain'"):
        _WrongDomainPlugin().load_dataset()


def test_discover_application_plugins_returns_names() -> None:
    """Entry-point discovery returns the registered plugin names."""
    names = discover_application_plugins()
    assert isinstance(names, list)


def test_validate_plugin_rejects_non_plugin() -> None:
    """An object that does not satisfy the plugin protocol is rejected."""
    with pytest.raises(TypeError, match="does not satisfy ApplicationPlugin"):
        _validate_plugin("bad", object())  # type: ignore[arg-type]


def test_finite_metric_rejects_non_finite() -> None:
    """A non-finite metric value is rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        _finite_metric("loss", float("inf"))


def test_metadata_array_rejects_missing_key() -> None:
    """A missing metadata key is rejected."""
    with pytest.raises(ValueError, match="is missing metadata 'beliefs'"):
        _metadata_array(_artifact(), "beliefs")


def test_metadata_array_rejects_wrong_shape() -> None:
    """A metadata array of the wrong shape is rejected."""
    artifact = _artifact(metadata={"beliefs": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="must have shape"):
        _metadata_array(artifact, "beliefs")


def test_metadata_array_rejects_non_finite() -> None:
    """A non-finite metadata array is rejected.

    The artifact constructor already forbids non-finite metadata, so a
    duck-typed stand-in exercises the helper's own finiteness guard.
    """
    fake_artifact = cast(
        Any,
        SimpleNamespace(
            metadata={"beliefs": [0.1, float("inf")]},
            source_name="unit-source",
            n_oscillators=2,
        ),
    )
    with pytest.raises(ValueError, match="metadata 'beliefs' must be finite"):
        _metadata_array(fake_artifact, "beliefs")
