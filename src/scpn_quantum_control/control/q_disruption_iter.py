# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Q Disruption Iter
"""ITER-specific disruption classifier with 11 physics-based features.

Feature ranges from ITER Physics Basis, Nuclear Fusion 39 (12), 1999.

Integration with scpn-fusion-core: use from_fusion_core_shot() to load
real tokamak disruption data from NPZ archives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .q_disruption import QuantumDisruptionClassifier

SCPN_CONTROL_BRIDGE_CONTRACT_SCHEMA_VERSION = (
    "scpn-control.quantum-disruption-dependency-contract.v1"
)
SCPN_CONTROL_BRIDGE_REPORT_SCHEMA_VERSION = "scpn-control.quantum-disruption-bridge-report.v1"
SCPN_CONTROL_KERNEL_REPORT_SCHEMA_VERSION = "scpn-control.quantum-disruption-kernel-report.v1"
SCPN_CONTROL_CERTIFICATE_SCHEMA_VERSION = "scpn-control.quantum-disruption-advisory-certificate.v1"
CONTROL_FACADE_OWNER = "scpn-control"
QUANTUM_BACKEND_OWNER = "scpn-quantum-control"
QUANTUM_MODULE = "scpn_quantum_control.control.q_disruption_iter"
CLAIM_BOUNDARY = (
    "advisory bounded-model quantum disruption bridge; not measured facility validation, "
    "not controller promotion, and not publication-safe evidence without external validation"
)
REQUIRED_DOWNSTREAM_POLICY = (
    "do_not_admit_control_action",
    "do_not_publish_as_facility_validation",
    "require_external_evidence",
)
CONTROL_FEATURE_NAMES = (
    "Ip",
    "beta_N",
    "q95",
    "n_nGW",
    "li",
    "dBp_dt",
    "locked_mode_amp",
    "n1_rms",
)
EXTRA_ITER_FEATURE_NAMES = ("P_rad", "V_loop", "W_stored", "kappa", "dIp_dt")
QUANTUM_CORE_DEPENDENCIES = (
    "qiskit>=2.2,<3.0",
    "qiskit-" + "a" + "er>=0.15,<1.0",
    "qiskit-qasm3-import>=0.6,<1.0",
)
QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES = (
    "qiskit-ibm-runtime>=0.40,<1.0",
    "amazon-bra" + "k" + "et-sdk>=1.117,<2.0",
    "azure-quantum>=3.9,<4.0",
    "qbraid>=0.12,<1.0",
    "cirq-core>=1.6,<2.0",
    "pennylane>=0.40,<1.0",
    "requests>=2.22,<3.0",
    "oqc-qcaas-client>=3.22,<4.0",
    "pulser-core>=1.8,<2.0",
    "perceval-quandela>=1.1,<2.0",
    "pytket-quantinuum>=0.59,<1.0",
    "pyquil>=4.17,<5.0",
)


@dataclass
class ITERFeatureSpec:
    """11 ITER disruption features with physical units and valid ranges."""

    names: list[str] = field(
        default_factory=lambda: [
            "I_p",  # MA, plasma current
            "q95",  # safety factor at 95% flux
            "li",  # internal inductance
            "n_GW",  # Greenwald fraction
            "beta_N",  # normalized beta
            "P_rad",  # MW, radiated power
            "locked_mode",  # T, locked mode amplitude
            "V_loop",  # V, loop voltage
            "W_stored",  # MJ, stored energy
            "kappa",  # elongation
            "dIp_dt",  # MA/s, current ramp rate
        ]
    )
    mins: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, -5.0])
    )
    maxs: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [17.0, 8.0, 2.0, 1.5, 4.0, 100.0, 0.01, 5.0, 400.0, 2.2, 5.0]
        )
    )


# Mapping from fusion-core NPZ archive keys to ITER feature indices.
_FUSION_CORE_KEY_MAP: dict[str, int] = {
    "Ip_MA": 0,
    "q95": 1,
    "ne_1e19": 3,  # maps to n_GW (Greenwald fraction proxy)
    "beta_N": 4,
    "locked_mode_amp": 6,
}


def normalize_iter_features(
    raw: NDArray[np.float64], spec: ITERFeatureSpec | None = None
) -> NDArray[np.float64]:
    """Min-max normalize using ITER physics ranges, clip to [0, 1]."""
    if spec is None:
        spec = ITERFeatureSpec()
    denom = spec.maxs - spec.mins
    denom = np.where(denom > 0, denom, 1.0)
    normed: NDArray[np.float64] = np.clip((raw - spec.mins) / denom, 0.0, 1.0)
    return normed


def scpn_control_bridge_dependency_contract() -> dict[str, Any]:
    """Return the SCPN-CONTROL disruption bridge dependency contract.

    The contract is intentionally mirrored locally instead of importing
    SCPN-CONTROL, so this backend can be validated before CONTROL is installed.
    """

    spec = ITERFeatureSpec()
    payload: dict[str, Any] = {
        "schema_version": SCPN_CONTROL_BRIDGE_CONTRACT_SCHEMA_VERSION,
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "control_package": "scpn-control",
        "quantum_package": "scpn-quantum-control",
        "quantum_module": QUANTUM_MODULE,
        "report_schema_versions": {
            "bridge": SCPN_CONTROL_BRIDGE_REPORT_SCHEMA_VERSION,
            "kernel": SCPN_CONTROL_KERNEL_REPORT_SCHEMA_VERSION,
            "certificate": SCPN_CONTROL_CERTIFICATE_SCHEMA_VERSION,
        },
        "required_public_surface": {
            "classifier_class": "QuantumDisruptionClassifier",
            "constructor_kwargs": ["seed"],
            "predict_method": "predict",
            "predict_input": {
                "shape": [11],
                "feature_names": list(spec.names),
                "normalised_range": [0.0, 1.0],
                "dtype": "float64-compatible",
            },
            "predict_output": {
                "type": "scalar-float",
                "range": [0.0, 1.0],
            },
        },
        "feature_contract": {
            "control_feature_names": list(CONTROL_FEATURE_NAMES),
            "iter_feature_names": list(spec.names),
            "extra_iter_features": list(EXTRA_ITER_FEATURE_NAMES),
            "centre_defaults_allowed_only_when_declared": True,
        },
        "dependency_groups": {
            "control_runtime": ["numpy"],
            "quantum_core": list(QUANTUM_CORE_DEPENDENCIES),
            "quantum_optional_providers": list(QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "required_downstream_policy": list(REQUIRED_DOWNSTREAM_POLICY),
        "admitted_for_control": False,
        "publication_safe": False,
    }
    payload["contract_sha256"] = _contract_digest(payload)
    return validate_scpn_control_bridge_dependency_contract(payload)


def validate_scpn_control_bridge_dependency_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the SCPN-CONTROL disruption bridge dependency contract."""

    if not isinstance(payload, dict):
        raise ValueError("SCPN-CONTROL bridge dependency contract must be an object")
    if payload.get("schema_version") != SCPN_CONTROL_BRIDGE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("SCPN-CONTROL bridge dependency contract schema_version is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract control_facade_owner is unsupported"
        )
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract quantum_backend_owner is unsupported"
        )
    if payload.get("quantum_module") != QUANTUM_MODULE:
        raise ValueError("SCPN-CONTROL bridge dependency contract quantum_module is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("SCPN-CONTROL bridge dependency contract claim_boundary is unsupported")
    if payload.get("admitted_for_control") is not False:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract admitted_for_control must be false"
        )
    if payload.get("publication_safe") is not False:
        raise ValueError("SCPN-CONTROL bridge dependency contract publication_safe must be false")
    _validate_bridge_report_schemas(payload.get("report_schema_versions"))
    _validate_bridge_public_surface(payload.get("required_public_surface"))
    _validate_bridge_feature_contract(payload.get("feature_contract"))
    _validate_bridge_dependency_groups(payload.get("dependency_groups"))
    policy = payload.get("required_downstream_policy")
    if not isinstance(policy, list) or any(
        not isinstance(item, str) or not item for item in policy
    ):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract required_downstream_policy must be strings"
        )
    for required_policy in REQUIRED_DOWNSTREAM_POLICY:
        if required_policy not in policy:
            raise ValueError(
                f"SCPN-CONTROL bridge dependency contract missing policy {required_policy}"
            )
    declared_digest = payload.get("contract_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract contract_sha256 must be a SHA-256 hex digest"
        )
    if _contract_digest(payload) != declared_digest.lower():
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract contract_sha256 does not match payload"
        )
    return payload


def generate_synthetic_iter_data(
    n_samples: int,
    disruption_fraction: float = 0.3,
    rng: np.random.Generator | None = None,
    *,
    allow_synthetic: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Synthetic ITER disruption data for classifier benchmarking.

    Safe samples: drawn from normal distributions near ITER operational point.
    Disruption samples: shifted locked_mode up, q95 down, beta_N up.
    Returns (X, y) where X is (n_samples, 11) normalized, y is binary labels.
    """
    if not allow_synthetic:
        raise RuntimeError(
            "Refusing generated ITER disruption data without allow_synthetic=True. "
            "Use from_fusion_core_shot() or measured plasma diagnostics for publication-safe claims."
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0.0 < disruption_fraction < 1.0:
        raise ValueError("disruption_fraction must be strictly between 0 and 1.")
    if rng is None:
        rng = np.random.default_rng()

    spec = ITERFeatureSpec()
    n_disrupt = int(n_samples * disruption_fraction)
    n_safe = n_samples - n_disrupt

    # ITER operational point (approximate centers)
    centers = np.array([15.0, 3.0, 0.85, 0.85, 1.8, 30.0, 0.0001, 0.3, 350.0, 1.7, 0.0])
    spreads = np.array([1.0, 0.3, 0.1, 0.1, 0.3, 5.0, 0.0001, 0.1, 20.0, 0.05, 0.5])

    safe = rng.normal(centers, spreads, (n_safe, 11))

    disrupt_centers = centers.copy()
    disrupt_centers[6] = 0.005  # locked_mode high
    disrupt_centers[1] = 2.0  # q95 low (stability boundary)
    disrupt_centers[4] = 3.5  # beta_N high (beta limit)
    disrupt_spreads = spreads.copy()
    disrupt_spreads[6] = 0.002
    disrupt = rng.normal(disrupt_centers, disrupt_spreads, (n_disrupt, 11))

    X = np.vstack([safe, disrupt])
    y = np.concatenate([np.zeros(n_safe), np.ones(n_disrupt)])

    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]
    X = normalize_iter_features(X, spec)
    return X, y


def from_fusion_core_shot(
    shot_data: dict[str, Any],
    *,
    allow_center_defaults: bool = False,
    allow_density_proxy: bool = False,
) -> tuple[NDArray[np.float64], int, list[str]]:
    """Convert a fusion-core NPZ disruption shot to ITER feature vector.

    shot_data: dict loaded from scpn_fusion.io.tokamak_disruption_archive
               with keys like Ip_MA, q95, beta_N, locked_mode_amp, ne_1e19,
               is_disruption, disruption_time_idx.

    Returns (features_11, label, warnings) where features are time-averaged
    scalars normalized to [0, 1], label is 0 (safe) or 1 (disruption),
    and warnings lists any explicitly allowed centre defaults.

    ``ne_1e19`` maps to the n_GW slot only when ``allow_density_proxy`` is
    true. For production Greenwald fraction input, provide ``n_GW`` directly.
    """
    spec = ITERFeatureSpec()
    centers = np.array([15.0, 3.0, 0.85, 0.85, 1.8, 30.0, 0.0001, 0.3, 350.0, 1.7, 0.0])
    raw = centers.copy()
    warnings: list[str] = []

    mapped_indices: set[int] = set()
    key_map = dict(_FUSION_CORE_KEY_MAP)
    if "n_GW" in shot_data:
        key_map["n_GW"] = 3
    elif "ne_1e19" in shot_data and not allow_density_proxy:
        raise ValueError("ne_1e19 cannot be used as n_GW without allow_density_proxy=True")

    for key, idx in key_map.items():
        if key == "ne_1e19" and not allow_density_proxy:
            continue
        if key in shot_data:
            arr = np.asarray(shot_data[key], dtype=np.float64)
            raw[idx] = float(np.mean(arr)) if arr.ndim > 0 else float(arr)
            mapped_indices.add(idx)

    missing = [name for idx, name in enumerate(spec.names) if idx not in mapped_indices]
    if missing and not allow_center_defaults:
        missing_list = ", ".join(missing)
        raise ValueError(
            "missing required ITER features; pass allow_center_defaults=True "
            f"only for labelled non-publication fallback values: {missing_list}"
        )

    for idx, name in enumerate(spec.names):
        if idx not in mapped_indices:
            warnings.append(f"{name} defaulted to center value {centers[idx]}")

    features = normalize_iter_features(raw, spec)
    label = int(shot_data.get("is_disruption", 0))
    return features, label, warnings


class DisruptionBenchmark:
    """ITER disruption classification benchmark using quantum circuit classifier."""

    def __init__(
        self,
        n_train: int = 100,
        n_test: int = 50,
        seed: int = 42,
        *,
        allow_synthetic: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.source_mode = "synthetic"
        self.publication_safe = False
        self.X_train, self.y_train = generate_synthetic_iter_data(
            n_train,
            rng=self.rng,
            allow_synthetic=allow_synthetic,
        )
        self.X_test, self.y_test = generate_synthetic_iter_data(
            n_test,
            rng=self.rng,
            allow_synthetic=allow_synthetic,
        )
        self.classifier = QuantumDisruptionClassifier(seed=seed)

    def run(self, epochs: int = 10, lr: float = 0.1) -> dict[str, Any]:
        """Train and evaluate. Returns accuracy + predictions."""
        self.classifier.train(self.X_train, self.y_train, epochs=epochs, lr=lr)
        predictions = np.array([self.classifier.predict(x) for x in self.X_test])
        binary_pred = (predictions > 0.5).astype(float)
        accuracy = float(np.mean(binary_pred == self.y_test))
        return {
            "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "source_mode": self.source_mode,
            "publication_safe": self.publication_safe,
        }


def _validate_bridge_report_schemas(value: object) -> None:
    if value != {
        "bridge": SCPN_CONTROL_BRIDGE_REPORT_SCHEMA_VERSION,
        "kernel": SCPN_CONTROL_KERNEL_REPORT_SCHEMA_VERSION,
        "certificate": SCPN_CONTROL_CERTIFICATE_SCHEMA_VERSION,
    }:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract report_schema_versions are unsupported"
        )


def _validate_bridge_public_surface(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract required_public_surface must be an object"
        )
    if value.get("classifier_class") != "QuantumDisruptionClassifier":
        raise ValueError("SCPN-CONTROL bridge dependency contract classifier_class is unsupported")
    if value.get("constructor_kwargs") != ["seed"]:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract constructor_kwargs are unsupported"
        )
    if value.get("predict_method") != "predict":
        raise ValueError("SCPN-CONTROL bridge dependency contract predict_method is unsupported")
    predict_input = value.get("predict_input")
    if not isinstance(predict_input, dict):
        raise ValueError("SCPN-CONTROL bridge dependency contract predict_input must be an object")
    if predict_input.get("shape") != [11]:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_input shape is unsupported"
        )
    if predict_input.get("feature_names") != ITERFeatureSpec().names:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_input feature_names are unsupported"
        )
    if predict_input.get("normalised_range") != [0.0, 1.0]:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_input normalised_range is unsupported"
        )
    if predict_input.get("dtype") != "float64-compatible":
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_input dtype is unsupported"
        )
    predict_output = value.get("predict_output")
    if not isinstance(predict_output, dict):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_output must be an object"
        )
    if predict_output.get("type") != "scalar-float":
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_output type is unsupported"
        )
    if predict_output.get("range") != [0.0, 1.0]:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract predict_output range is unsupported"
        )


def _validate_bridge_feature_contract(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract feature_contract must be an object"
        )
    if value.get("control_feature_names") != list(CONTROL_FEATURE_NAMES):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract control_feature_names are unsupported"
        )
    if value.get("iter_feature_names") != ITERFeatureSpec().names:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract iter_feature_names are unsupported"
        )
    if value.get("extra_iter_features") != list(EXTRA_ITER_FEATURE_NAMES):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract extra_iter_features are unsupported"
        )
    if value.get("centre_defaults_allowed_only_when_declared") is not True:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract centre default policy is unsupported"
        )


def _validate_bridge_dependency_groups(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract dependency_groups must be an object"
        )
    if value.get("control_runtime") != ["numpy"]:
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract control_runtime dependencies are unsupported"
        )
    if value.get("quantum_core") != list(QUANTUM_CORE_DEPENDENCIES):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract quantum_core dependencies are unsupported"
        )
    if value.get("quantum_optional_providers") != list(QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES):
        raise ValueError(
            "SCPN-CONTROL bridge dependency contract optional provider dependencies are unsupported"
        )


def _contract_digest(contract: Mapping[str, Any]) -> str:
    content = {key: value for key, value in contract.items() if key != "contract_sha256"}
    return _payload_digest({"dependency_contract": content})


def _payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)
