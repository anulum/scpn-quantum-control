# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the QPU data artifact
"""Guard and validator tests for the QPU data artifact contract.

Covers the array-hash presence check, the positive-finite-float and
finite-float-array validators, the artifact source-mode/shape/layer guards and
the SC-NeuroCore datastream schema and n_layers guards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.bridge import artifact_from_arrays
from scpn_quantum_control.bridge.qpu_data_artifact import (
    SC_NEUROCORE_STREAM_SCHEMA,
    QPUDataArtifact,
    _finite_float_array,
    _positive_finite_float,
    _verify_or_set_array_hash,
)


def _artifact(**overrides: Any) -> QPUDataArtifact:
    kwargs: dict[str, Any] = {
        "domain": "unit",
        "source_name": "unit-source",
        "source_mode": "curated",
        "K_nm": np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        "omega": np.array([0.1, 0.2], dtype=np.float64),
        "normalization": "unit",
        "extraction_method": "unit-test",
    }
    kwargs.update(overrides)
    return artifact_from_arrays(**kwargs)


def test_artifact_reports_oscillator_count() -> None:
    """The artifact reports the oscillator count implied by the coupling matrix."""
    assert _artifact().n_oscillators == 2


def test_verify_array_hash_rejects_present_hash_without_array() -> None:
    """A recorded hash with no backing array is rejected."""
    with pytest.raises(ValueError, match="is present but the corresponding array is absent"):
        _verify_or_set_array_hash({"K_nm_sha256": "0" * 64}, "K_nm_sha256", None)


def test_positive_finite_float_rejects_string() -> None:
    """A string value is rejected by the positive-finite-float validator."""
    with pytest.raises(ValueError, match="must be positive finite"):
        _positive_finite_float("dt", "fast")


def test_positive_finite_float_rejects_complex() -> None:
    """A complex value is rejected."""
    with pytest.raises(ValueError, match="must be positive finite"):
        _positive_finite_float("dt", 1 + 2j)


def test_positive_finite_float_rejects_uncoercible() -> None:
    """A non-coercible object is rejected."""
    with pytest.raises(ValueError, match="must be positive finite"):
        _positive_finite_float("dt", object())


def test_finite_float_array_rejects_wrong_ndim() -> None:
    """An array of the wrong dimensionality is rejected."""
    with pytest.raises(ValueError, match="must be 1-D"):
        _finite_float_array("v", [[1.0, 2.0]], ndim=1)


def test_finite_float_array_rejects_non_finite() -> None:
    """A non-finite array is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _finite_float_array("v", [1.0, np.inf], ndim=1)


def test_artifact_rejects_unknown_source_mode() -> None:
    """An unknown source mode is rejected."""
    with pytest.raises(ValueError, match="source_mode must be one of"):
        _artifact(source_mode="bogus_mode")


def test_artifact_rejects_non_square_k() -> None:
    """A non-square coupling matrix is rejected."""
    with pytest.raises(ValueError, match="K_nm must be square"):
        _artifact(K_nm=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64))


def test_artifact_rejects_theta0_shape_mismatch() -> None:
    """A theta0 vector that does not match omega is rejected."""
    with pytest.raises(ValueError, match="theta0 shape must match omega shape"):
        _artifact(theta0=np.zeros(3, dtype=np.float64))


def test_artifact_rejects_layer_assignment_length() -> None:
    """A layer-assignment tuple of the wrong length is rejected."""
    with pytest.raises(ValueError, match="layer_assignments length must match"):
        _artifact(layer_assignments=("a", "b", "c"))


def test_datastream_rejects_unknown_schema() -> None:
    """An unknown SC-NeuroCore datastream schema version is rejected."""
    with pytest.raises(ValueError, match="unsupported SC-NeuroCore datastream schema version"):
        QPUDataArtifact.from_scpn_datastream_payload({"schema_version": "v0"})


def test_datastream_rejects_non_integer_n_layers() -> None:
    """A non-integer n_layers field is rejected."""
    payload = {
        "schema_version": SC_NEUROCORE_STREAM_SCHEMA,
        "seed": 1,
        "knm": [[0.0, 0.1], [0.1, 0.0]],
        "omega_rad_s": [0.1, 0.2],
        "layer_ids": ["L0", "L1"],
        "n_layers": "two",
    }
    with pytest.raises(ValueError, match="n_layers must be an integer"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)
