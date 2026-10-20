# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — variational baseline qualification
"""Evaluate a seeded positive-amplitude RBM on a predeclared second Hamiltonian."""

from __future__ import annotations

import json

import numpy as np

from scpn_quantum_control.jax_nqs_baseline_product import (
    JAXNQSBaselineSpec,
    qualify_jax_nqs_baseline,
)


def main() -> None:
    """Print real qualification evidence without writing files or using a provider."""
    coupling = np.zeros((2, 2))
    request = JAXNQSBaselineSpec.from_arrays(
        coupling, np.array((0.7, -0.2)), n_iterations=30, seed=7
    )
    result = qualify_jax_nqs_baseline(
        request, held_out={"opposite_field": (coupling, np.array((-0.5, 0.1)))}
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
