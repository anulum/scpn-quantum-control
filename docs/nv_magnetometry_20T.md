# NV-Centre 20 T Magnetometry

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.sensing.nv_magnetometry_20T` is a **simulation-only**
nitrogen-vacancy (NV) magnetometry response model valid into the 20 T regime.
The ground-state spin-1 Hamiltonian is diagonalised exactly, so the
optically-detected magnetic-resonance (ODMR) frequencies, the shot-noise DC
sensitivity, and a noisy-spectrum field-calibration loop remain valid past the
ground-state level anti-crossing (GSLAC, ~102 mT axial) and into the high-field
regime where the electron Zeeman term dominates the zero-field splitting.
Hardware calibration against a NIST-traceable reference is a separate,
hardware-gated workstream (`MIF_NV_HARDWARE_CI=1`).

## Spin Hamiltonian

`H = D (Sz² - 2/3) + E (Sx² - Sy²) + gamma_e B·S`, with `D = 2.870 GHz`,
`gamma_e = 28.024951 GHz/T` (Doherty et al., Physics Reports 528, 1, 2013).

```python
from scpn_quantum_control.sensing.nv_magnetometry_20T import NVCenter, odmr_resonances_hz

nv = NVCenter()
lo, hi = odmr_resonances_hz(nv, field_tesla=20.0)   # (557.6 GHz, 563.4 GHz)
```

The two ODMR transitions behave as:

| regime | lower resonance | upper resonance |
|---|---|---|
| zero field | `D` | `D` |
| axial, below GSLAC | `D - gamma_e B` | `D + gamma_e B` |
| above GSLAC / high field | `gamma_e B - D` | `D + gamma_e B` |

The **upper** resonance is `D + gamma_e B` for every field magnitude, which is
what makes the field recovery unambiguous.

## Sensitivity

`cw_odmr_dc_sensitivity_t_per_sqrt_hz` returns the shot-noise-limited CW-ODMR DC
sensitivity `eta = P_F · delta_nu / (gamma_e · C · sqrt(R))` (Barry et al.,
Reviews of Modern Physics 92, 015004, 2020), with the Lorentzian prefactor
`P_F = 4 / (3 sqrt(3))`.

## Field calibration

`simulate_odmr_measurement` produces a noisy spectrum; `calibrate_field_from_odmr`
recovers the field from the deepest dip in a resolved scan window (the upper
resonance, `B = (f_upper - D) / gamma_e`). The scan must resolve the linewidth
(a few samples per FWHM), as a real field-tracking loop does. Measured recovery
across 0.07–20 T is ~2 µT (benchmark `results/nv_magnetometry_benchmark.json`).

## Acceleration

The ODMR Lorentzian-dip spectrum dispatches to a Rust kernel that is **bit-true**
with the NumPy reference (verified over random inputs).

Measured per-call wall-time (release build, median of 7,
`scripts/bench_nv_magnetometry.py`, `functional_non_isolated`):

| grid size | NumPy | Rust | speed-up |
|---|---|---|---|
| 1 024 | 12.1 µs | 2.70 µs | 4.5× |
| 8 192 | 35.5 µs | 18.1 µs | 2.0× |
| 65 536 | 247.7 µs | 261.7 µs | 0.95× |
| 262 144 | 5 160 µs | 730 µs | 7.1× |

The Rust kernel is faster on small and large grids; NumPy's vectorisation is
competitive at the mid range. Both paths are kept and measured honestly.

## Consumers

SCPN-MIF-CORE may consume the model for B-dot probe cross-validation in the FRC
compression environment; this is not on the critical path before MIF 0.5.0.
