#!/usr/bin/env python3
import json

from scpn_quantum_control.accel import rust_kuramoto_classical


def run_test():
    result = rust_kuramoto_classical.run_large_n(
        N=50000, K=0.0, lambda_fim=8.0, delta=0.1, steps=10000
    )
    with open("results/large_n_fim.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Large-N classical FIM validation test completed.")


if __name__ == "__main__":
    run_test()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
