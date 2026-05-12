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
    *,
    allow_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
    shot_data: dict,
    *,
    allow_center_defaults: bool = False,
    allow_density_proxy: bool = False,
) -> tuple[np.ndarray, int, list[str]]:
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
            "source_mode": self.source_mode,
            "publication_safe": self.publication_safe,
        }
