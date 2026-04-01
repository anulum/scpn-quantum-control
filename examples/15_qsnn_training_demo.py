# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""QSNN training demo: parameter-shift gradient descent on QuantumDenseLayer."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer


def main() -> None:
    print("QSNN Training Demo")
    print("=" * 50)

    layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
    trainer = QSNNTrainer(layer, lr=0.05)

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (10, 2))
    y = np.column_stack([X[:, 0] > 0.5, X[:, 1] > 0.5]).astype(float)

    print(f"Training: {len(X)} samples, {layer.n_inputs} inputs, {layer.n_neurons} neurons")
    losses = trainer.train(X, y, epochs=5)

    for i, loss in enumerate(losses):
        print(f"  Epoch {i + 1}: loss = {loss:.4f}")

    if losses[-1] < losses[0]:
        print(f"\nLoss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print("\nLoss did not decrease (may need more epochs or different lr)")


if __name__ == "__main__":
    main()
