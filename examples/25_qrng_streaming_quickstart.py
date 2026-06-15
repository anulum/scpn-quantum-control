# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRNG streaming quickstart
"""Generate quantum random bits and run the NIST + FIPS health checks (QUA-C.1).

Run with::

    python examples/25_qrng_streaming_quickstart.py
"""

from __future__ import annotations

from scpn_quantum_control.entropy import QRNGStream
from scpn_quantum_control.entropy.nist_sp800_22 import frequency_test, runs_test


def main() -> None:
    # Bell-pair entropy source with Von Neumann debiasing.
    qrng = QRNGStream("bell_pair", register_qubits=64, debias=True, seed=2026)

    sample = qrng.sample(20_000)
    print(f"generated {sample.size} bits, mean = {sample.mean():.4f}")

    print(f"frequency test P-value = {frequency_test(sample).p_value:.4f}")
    print(f"runs test P-value      = {runs_test(sample).p_value:.4f}")

    report = qrng.health_check()
    print(
        "health check: "
        f"FIPS={'pass' if report.fips.passed else 'fail'}  "
        f"healthy={report.healthy}  "
        f"Shannon/bit={report.shannon_entropy_per_bit:.5f}  "
        f"min-entropy/bit={report.min_entropy_per_bit:.5f}"
    )

    qrng.close()


if __name__ == "__main__":
    main()
