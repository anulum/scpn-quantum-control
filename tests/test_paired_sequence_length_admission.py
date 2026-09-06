# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — paired-sequence length admission
"""Public APIs that pair two sequences must refuse a length mismatch.

``zip`` truncates to the shorter argument by default. Where the two sequences
are a scientific pairing — samples against labels, births against deaths — that
truncation is silent data loss that still returns a plausible number, which is
the worst shape a defect can take here.

These tests exercise real public entry points rather than the ``zip`` calls
themselves, so they describe the contract a caller can rely on and not an
implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.q_disruption import QuantumDisruptionClassifier
from scpn_quantum_control.topology_control.complexes import PersistenceDiagram


class TestTrainingDatasetPairing:
    """Samples and labels are one dataset; a mismatch is a caller error."""

    def test_training_refuses_more_samples_than_labels(self) -> None:
        """Truncating to the labels would train on a silently smaller set."""
        classifier = QuantumDisruptionClassifier(n_features=2, n_layers=1, seed=0)
        samples = np.zeros((3, 2), dtype=np.float64)
        labels = np.zeros(2, dtype=np.float64)

        with pytest.raises(ValueError, match="shorter"):
            classifier.train(samples, labels, epochs=1)

    def test_training_refuses_more_labels_than_samples(self) -> None:
        """The mismatch must be refused from either side, not only one."""
        classifier = QuantumDisruptionClassifier(n_features=2, n_layers=1, seed=0)
        samples = np.zeros((2, 2), dtype=np.float64)
        labels = np.zeros(3, dtype=np.float64)

        with pytest.raises(ValueError, match="longer"):
            classifier.train(samples, labels, epochs=1)

    def test_training_accepts_a_matched_dataset(self) -> None:
        """The guard must not reject the ordinary case it exists to protect."""
        classifier = QuantumDisruptionClassifier(n_features=2, n_layers=1, seed=0)
        samples = np.zeros((2, 2), dtype=np.float64)
        labels = np.zeros(2, dtype=np.float64)

        classifier.train(samples, labels, epochs=1)


class TestPersistencePairing:
    """A persistence diagram pairs each birth with exactly one death."""

    def test_lifetimes_refuse_an_unpaired_birth(self) -> None:
        """A dropped feature would shorten the lifetime spectrum in silence."""
        diagram = PersistenceDiagram(dimension=1, births=(0.1, 0.2, 0.3), deaths=(0.5, 0.6))

        with pytest.raises(ValueError, match="shorter"):
            _ = diagram.lifetimes

    def test_lifetimes_refuse_an_unpaired_death(self) -> None:
        """The pairing is symmetric, so the opposite mismatch must refuse too."""
        diagram = PersistenceDiagram(dimension=1, births=(0.1,), deaths=(0.5, 0.6))

        with pytest.raises(ValueError, match="longer"):
            _ = diagram.lifetimes

    def test_lifetimes_are_returned_for_a_paired_diagram(self) -> None:
        """Matched births and deaths still produce their lifetimes."""
        diagram = PersistenceDiagram(dimension=1, births=(0.1, 0.2), deaths=(0.5, 0.7))

        assert diagram.lifetimes == pytest.approx((0.4, 0.5))
