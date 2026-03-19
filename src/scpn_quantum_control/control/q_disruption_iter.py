# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""ITER-specific disruption classifier with 11 physics-based features.

Feature ranges from ITER Physics Basis, Nuclear Fusion 39 (12), 1999.

Integration with scpn-fusion-core: use from_fusion_core_shot() to load
real tokamak disruption data from NPZ archives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .q_disruption import QuantumDisruptionClassifier


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
    mins: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, -5.0])
    )
    maxs: np.ndarray = field(
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


def normalize_iter_features(raw: np.ndarray, spec: ITERFeatureSpec | None = None) -> np.ndarray:
    """Min-max normalize using ITER physics ranges, clip to [0, 1]."""
    if spec is None:
        spec = ITERFeatureSpec()
    denom = spec.maxs - spec.mins
    denom = np.where(denom > 0, denom, 1.0)
    normed: np.ndarray = np.clip((raw - spec.mins) / denom, 0.0, 1.0)
    return normed


def generate_synthetic_iter_data(
    n_samples: int,
    disruption_fraction: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ITER disruption data for classifier benchmarking.

    Safe samples: drawn from normal distributions near ITER operational point.
    Disruption samples: shifted locked_mode up, q95 down, beta_N up.
    Returns (X, y) where X is (n_samples, 11) normalized, y is binary labels.
    """
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


def from_fusion_core_shot(shot_data: dict) -> tuple[np.ndarray, int, list[str]]:
    """Convert a fusion-core NPZ disruption shot to ITER feature vector.

    shot_data: dict loaded from scpn_fusion.io.tokamak_disruption_archive
               with keys like Ip_MA, q95, beta_N, locked_mode_amp, ne_1e19,
               is_disruption, disruption_time_idx.

    Returns (features_11, label, warnings) where features are time-averaged
    scalars normalized to [0, 1], label is 0 (safe) or 1 (disruption),
    and warnings lists any features that defaulted to ITER center values.

    Note: ne_1e19 maps to the n_GW slot as a proxy. For accurate Greenwald
    fraction, divide by n_GW = I_p / (pi * a²) externally.
    """
    spec = ITERFeatureSpec()
    centers = np.array([15.0, 3.0, 0.85, 0.85, 1.8, 30.0, 0.0001, 0.3, 350.0, 1.7, 0.0])
    raw = centers.copy()
    warnings: list[str] = []

    mapped_indices: set[int] = set()
    for key, idx in _FUSION_CORE_KEY_MAP.items():
        if key in shot_data:
            arr = np.asarray(shot_data[key], dtype=np.float64)
            raw[idx] = float(np.mean(arr)) if arr.ndim > 0 else float(arr)
            mapped_indices.add(idx)

    for idx, name in enumerate(spec.names):
        if idx not in mapped_indices:
            warnings.append(f"{name} defaulted to center value {centers[idx]}")

    features = normalize_iter_features(raw, spec)
    label = int(shot_data.get("is_disruption", 0))
    return features, label, warnings


class DisruptionBenchmark:
    """ITER disruption classification benchmark using quantum circuit classifier."""

    def __init__(self, n_train: int = 100, n_test: int = 50, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.X_train, self.y_train = generate_synthetic_iter_data(n_train, rng=self.rng)
        self.X_test, self.y_test = generate_synthetic_iter_data(n_test, rng=self.rng)
        self.classifier = QuantumDisruptionClassifier(seed=seed)

    def run(self, epochs: int = 10, lr: float = 0.1) -> dict:
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
        }
