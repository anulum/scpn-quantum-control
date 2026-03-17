# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""ITER disruption classifier demo: synthetic data + quantum circuit classifier."""

from __future__ import annotations

from scpn_quantum_control.control.q_disruption_iter import (
    DisruptionBenchmark,
    ITERFeatureSpec,
    generate_synthetic_iter_data,
)


def main() -> None:
    spec = ITERFeatureSpec()
    print("ITER Disruption Classifier Demo")
    print("=" * 50)
    print(f"Features ({len(spec.names)}):")
    for name, lo, hi in zip(spec.names, spec.mins, spec.maxs):
        print(f"  {name:>15}: [{lo:.1f}, {hi:.1f}]")

    X, y = generate_synthetic_iter_data(200, disruption_fraction=0.3, rng=None)
    print(f"\nSynthetic data: {X.shape[0]} samples, {int(y.sum())} disruptions")

    bench = DisruptionBenchmark(n_train=80, n_test=40, seed=42)
    result = bench.run(epochs=5, lr=0.1)
    print(f"Accuracy: {result['accuracy']:.1%} ({result['n_test']} test samples)")


if __name__ == "__main__":
    main()
