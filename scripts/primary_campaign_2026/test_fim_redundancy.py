#!/usr/bin/env python3
import json

from scpn_quantum_control.accel import rust_kuramoto_classical


def run_test():
    results = {}
    for lam in [0.0, 8.0]:
        res = rust_kuramoto_classical.run_large_n(
            N=10000, K=0.0, lambda_fim=lam, delta=0.1, steps=5000
        )
        results[lam] = res
    with open("results/fim_redundancy.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Falsification test 'Is FIM redundant?' completed.")


if __name__ == "__main__":
    run_test()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
