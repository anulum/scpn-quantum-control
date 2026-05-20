# Paper 0 UPDE Field-Coupling Fixture Report

Date: 2026-05-13

## Source Boundary

Spec: `upde.field_coupling`

Protocol: `paper0.upde.field.global_phase_coupling`

Source equations: `EQ0034`, `EQ0041`, `EQ0043`

Source ledgers: `P0R02512`, `P0R02634`, `P0R02644`

Hardware status: `simulator_only_no_provider_submission`

## Executable Fixture

Implementation:

- `src/scpn_quantum_control/paper0/upde_field_validation.py`
- `tests/test_paper0_upde_field_validation.py`

The fixture implements the source equation

`C_Field = zeta_L Psi_Global cos(theta_i^L - Theta_Psi)`

literally. It validates finite local phases, non-negative bounded field gain,
non-negative bounded field amplitude, finite global field phase, and
deterministic random-phase controls.

## Code-Path Wiring

- Field metadata and phase summary are exported through `UPDEPhaseArtifact`.
- The fixture records no provider payload and does not invoke feedback or
hardware schedulers.
- The randomised global phase null is deterministic through an explicit seed.

## Controls

The current executable fixture records:

- zero-field baseline with `zeta_L = 0`;
- randomised-global-phase projection control;
- bounded-amplitude metadata for `zeta_L * Psi_Global`.

## Measured Local Result

Result artefact:

- `paper0_upde_field_fixture_result_2026-05-13.json`

Measured on the local five-oscillator phase fixture:

- `zeta_L`: `0.42`;
- `Psi_Global`: `1.7`;
- `Theta_Psi`: `0.31`;
- bounded amplitude: `0.714`;
- coherent field-alignment projection: `0.48216188793076825`;
- zero-field baseline, L-infinity: `0.0`;
- randomised-phase projection absolute mean: `0.029577256961382775`;
- single fixture runtime: `6.5162550017703325 ms`.

## Verification

- `PYTHONPATH=src .venv-linux/bin/python -m pytest tests/test_paper0_upde_field_validation.py -q`
  - `5 passed in 0.20s`

## Next Step

Extend this pattern to `upde.natural_gradient`. The fixture must keep the
Fisher metric, regularisation/fail-fast boundary, Euclidean-gradient null, and
finite-difference free-energy check explicit.
