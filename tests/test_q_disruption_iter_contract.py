# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Contract and conversion tests for ITER disruption bridge
"""Dependency-contract, fusion-core conversion, and synthetic-data guard tests.

Covers every rejection branch of the SCPN-CONTROL bridge dependency-contract
validator, the fusion-core shot conversion (density proxy, missing-feature and
centre-default policy), the synthetic-data guards, the JSON normalisation of
numpy values, and the default feature specification.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.control.q_disruption_iter import (
    _jsonable,
    from_fusion_core_shot,
    generate_synthetic_iter_data,
    normalize_iter_features,
    scpn_control_bridge_dependency_contract,
    validate_scpn_control_bridge_dependency_contract,
)


def test_genuine_contract_validates_and_round_trips() -> None:
    """The generated contract passes validation and is returned unchanged."""
    contract = scpn_control_bridge_dependency_contract()
    assert validate_scpn_control_bridge_dependency_contract(contract) is contract


def test_non_object_contract_is_rejected() -> None:
    """A non-object payload is rejected outright."""
    with pytest.raises(ValueError, match="must be an object"):
        validate_scpn_control_bridge_dependency_contract(["not", "a", "dict"])  # type: ignore[arg-type]


_CONTRACT_MUTATIONS: list[tuple[tuple[str, ...], Any, str]] = [
    (("schema_version",), "x", "schema_version is unsupported"),
    (("control_facade_owner",), "x", "control_facade_owner is unsupported"),
    (("quantum_backend_owner",), "x", "quantum_backend_owner is unsupported"),
    (("quantum_module",), "x", "quantum_module is unsupported"),
    (("claim_boundary",), "x", "claim_boundary is unsupported"),
    (("admitted_for_control",), True, "admitted_for_control must be false"),
    (("publication_safe",), True, "publication_safe must be false"),
    (("report_schema_versions",), {}, "report_schema_versions are unsupported"),
    (("required_public_surface",), "x", "required_public_surface must be an object"),
    (("required_public_surface", "classifier_class"), "x", "classifier_class is unsupported"),
    (
        ("required_public_surface", "constructor_kwargs"),
        ["x"],
        "constructor_kwargs are unsupported",
    ),
    (("required_public_surface", "predict_method"), "x", "predict_method is unsupported"),
    (("required_public_surface", "predict_input"), "x", "predict_input must be an object"),
    (
        ("required_public_surface", "predict_input", "shape"),
        [99],
        "predict_input shape is unsupported",
    ),
    (
        ("required_public_surface", "predict_input", "normalised_range"),
        [0.0, 2.0],
        "normalised_range is unsupported",
    ),
    (
        ("required_public_surface", "predict_input", "dtype"),
        "x",
        "predict_input dtype is unsupported",
    ),
    (("required_public_surface", "predict_output"), "x", "predict_output must be an object"),
    (
        ("required_public_surface", "predict_output", "type"),
        "x",
        "predict_output type is unsupported",
    ),
    (
        ("required_public_surface", "predict_output", "range"),
        [0.0, 2.0],
        "predict_output range is unsupported",
    ),
    (("feature_contract",), "x", "feature_contract must be an object"),
    (
        ("feature_contract", "control_feature_names"),
        ["x"],
        "control_feature_names are unsupported",
    ),
    (("feature_contract", "iter_feature_names"), ["x"], "iter_feature_names are unsupported"),
    (("feature_contract", "extra_iter_features"), ["x"], "extra_iter_features are unsupported"),
    (
        ("feature_contract", "centre_defaults_allowed_only_when_declared"),
        False,
        "centre default policy is unsupported",
    ),
    (("dependency_groups",), "x", "dependency_groups must be an object"),
    (
        ("dependency_groups", "control_runtime"),
        ["x"],
        "control_runtime dependencies are unsupported",
    ),
    (("dependency_groups", "quantum_core"), ["x"], "quantum_core dependencies are unsupported"),
    (
        ("dependency_groups", "quantum_optional_providers"),
        ["x"],
        "optional provider dependencies are unsupported",
    ),
    (("required_downstream_policy",), [123], "required_downstream_policy must be strings"),
    (("contract_sha256",), "xyz", "must be a SHA-256 hex digest"),
]


@pytest.mark.parametrize("path,value,match", _CONTRACT_MUTATIONS)
def test_contract_field_mutation_is_rejected(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    """Each corrupted contract field is rejected with its specific message."""
    contract = deepcopy(scpn_control_bridge_dependency_contract())
    target: Any = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_scpn_control_bridge_dependency_contract(contract)


def test_contract_missing_required_policy_is_rejected() -> None:
    """Dropping a required downstream policy is rejected."""
    contract = deepcopy(scpn_control_bridge_dependency_contract())
    contract["required_downstream_policy"] = ["require_external_evidence"]
    with pytest.raises(ValueError, match="missing policy"):
        validate_scpn_control_bridge_dependency_contract(contract)


def test_contract_digest_mismatch_is_rejected() -> None:
    """A well-formed but incorrect digest is rejected."""
    contract = deepcopy(scpn_control_bridge_dependency_contract())
    contract["contract_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="does not match payload"):
        validate_scpn_control_bridge_dependency_contract(contract)


def test_from_fusion_core_shot_maps_features() -> None:
    """A shot with explicit Greenwald fraction maps its features and reads the label."""
    shot: dict[str, Any] = {
        "Ip_MA": [15.0, 15.2],
        "q95": 3.0,
        "beta_N": 1.8,
        "locked_mode_amp": 0.0001,
        "n_GW": 0.85,
        "is_disruption": 1,
    }
    features, label, warnings = from_fusion_core_shot(shot, allow_center_defaults=True)
    assert features.shape == (11,)
    assert label == 1
    assert warnings  # the unmapped features fall back to centre values


def test_from_fusion_core_shot_density_proxy_requires_opt_in() -> None:
    """Using ne_1e19 as the Greenwald proxy requires explicit opt-in."""
    with pytest.raises(ValueError, match="allow_density_proxy"):
        from_fusion_core_shot({"ne_1e19": 5.0})


def test_from_fusion_core_shot_skips_proxy_when_density_disallowed() -> None:
    """An explicit n_GW lets ne_1e19 be ignored without the proxy opt-in."""
    shot: dict[str, Any] = {"n_GW": 0.85, "ne_1e19": 5.0}
    features, _label, _warnings = from_fusion_core_shot(shot, allow_center_defaults=True)
    assert features.shape == (11,)


def test_from_fusion_core_shot_density_proxy_maps_when_allowed() -> None:
    """With the proxy opt-in, ne_1e19 populates the Greenwald slot."""
    shot: dict[str, Any] = {"ne_1e19": 5.0}
    features, _label, _warnings = from_fusion_core_shot(
        shot, allow_density_proxy=True, allow_center_defaults=True
    )
    assert features.shape == (11,)


def test_from_fusion_core_shot_missing_features_rejected() -> None:
    """Missing features without the centre-default opt-in are rejected."""
    with pytest.raises(ValueError, match="missing required ITER features"):
        from_fusion_core_shot({"Ip_MA": 15.0})


def test_generate_synthetic_requires_opt_in() -> None:
    """Synthetic data generation is refused without the explicit opt-in."""
    with pytest.raises(RuntimeError, match="allow_synthetic"):
        generate_synthetic_iter_data(10)


def test_generate_synthetic_rejects_non_positive_samples() -> None:
    """A non-positive sample count is rejected."""
    with pytest.raises(ValueError, match="n_samples must be positive"):
        generate_synthetic_iter_data(0, allow_synthetic=True)


def test_generate_synthetic_rejects_out_of_range_fraction() -> None:
    """A disruption fraction outside (0, 1) is rejected."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        generate_synthetic_iter_data(10, disruption_fraction=1.5, allow_synthetic=True)


def test_generate_synthetic_defaults_rng() -> None:
    """Omitting the generator falls back to a fresh default RNG."""
    features, labels = generate_synthetic_iter_data(10, allow_synthetic=True)
    assert features.shape == (10, 11)
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_jsonable_normalises_numpy_values() -> None:
    """Numpy arrays and scalars are reduced to plain JSON-compatible values."""
    assert _jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert _jsonable(np.float64(3.5)) == 3.5
    assert _jsonable(np.int64(7)) == 7


def test_normalize_iter_features_defaults_spec() -> None:
    """Omitting the feature spec uses the default ITER ranges."""
    raw = np.zeros(11, dtype=np.float64)
    normed = normalize_iter_features(raw)
    assert normed.shape == (11,)
    assert np.all((normed >= 0.0) & (normed <= 1.0))
