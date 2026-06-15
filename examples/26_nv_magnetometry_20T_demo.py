# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NV-centre 20 T magnetometry demo
"""Recover a high field from a noisy NV ODMR spectrum (simulation only).

Run with::

    python examples/26_nv_magnetometry_20T_demo.py
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NVCenter,
    calibrate_field_from_odmr,
    cw_odmr_dc_sensitivity_t_per_sqrt_hz,
    odmr_resonances_hz,
    simulate_odmr_measurement,
)


def main() -> None:
    nv = NVCenter()
    print(f"DC sensitivity = {cw_odmr_dc_sensitivity_t_per_sqrt_hz(nv) * 1e12:.1f} pT/sqrt(Hz)")

    for b_true in (0.1, 1.0, 20.0):
        lo, hi = odmr_resonances_hz(nv, b_true)
        freqs = np.linspace(hi - 5.0e7, hi + 5.0e7, 4000)
        measured = simulate_odmr_measurement(
            nv=nv, freqs=freqs, field_tesla=b_true, noise_std=0.004, seed=1
        )
        cal = calibrate_field_from_odmr(freqs, measured, nv, true_field_tesla=b_true)
        print(
            f"B = {b_true:>5} T  ODMR = ({lo / 1e9:.3f}, {hi / 1e9:.3f}) GHz  "
            f"recovered = {cal.field_tesla:.6f} T  error = {cal.abs_error_tesla * 1e6:.2f} uT"
        )


if __name__ == "__main__":
    main()
