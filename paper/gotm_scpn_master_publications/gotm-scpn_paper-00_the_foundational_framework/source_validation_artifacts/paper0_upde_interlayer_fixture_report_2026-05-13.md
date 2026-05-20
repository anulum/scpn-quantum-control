# Paper 0 UPDE Inter-Layer Fixture Report

Date: 2026-05-13

## Source Boundary

Spec: `upde.interlayer_coupling`

Protocol: `paper0.upde.interlayer.directional_coupling`

Source equations: `EQ0033`, `EQ0040`

Source ledgers: `P0R02510`, `P0R02630`

Hardware status: `simulator_only_no_provider_submission`

## Executable Fixture

Implementation:

- `src/scpn_quantum_control/paper0/upde_interlayer_validation.py`
- `tests/test_paper0_upde_interlayer_validation.py`

The fixture implements the inter-layer source form

`C_InterLayer = epsilon_{L-1} F_D(<theta^{L-1}>, theta_i^L) + epsilon_{L+1} G_U(theta_i^L, <theta^{L+1}>)`

with explicit circular phase means and sine phase-response channels. It keeps
the downward and upward channels separate before summing them, so perturbation
tests can falsify channel mixing.

## Code-Path Wiring

- Phase means and alignment are exported through `UPDEPhaseArtifact`.
- Hierarchical consistency is checked through
  `fep.predictive_coding.hierarchical_prediction_error`.
- Hardware status remains simulator-only; no feedback scheduler or provider
  submission is invoked by this fixture.

## Controls

The current executable fixture records:

- lower-layer perturbation response in the downward channel;
- upper-layer perturbation response in the upward channel;
- lower-to-upward and upper-to-downward cross-channel leakage controls;
- disconnected-layer null control with both inter-layer gains set to zero.

## Measured Local Result

Result artefact:

- `paper0_upde_interlayer_fixture_result_2026-05-13.json`

Measured on the local three-layer phase fixture:

- lower mean phase: `0.22000876621914228`;
- upper mean phase: `1.3799877248373542`;
- lower-to-downward response, L2: `0.10542163465303317`;
- upper-to-upward response, L2: `0.05679842904477164`;
- lower-to-upward leakage, L2: `0.0`;
- upper-to-downward leakage, L2: `0.0`;
- disconnected-layer null, L-infinity: `0.0`;
- predictive-error norm: `0.38808097958475096`;
- single fixture runtime: `10.21192199550569 ms`.

## Verification

- `PYTHONPATH=src .venv-linux/bin/python -m pytest tests/test_paper0_upde_interlayer_validation.py -q`
  - `5 passed in 0.21s`

## Next Step

Extend this pattern to `upde.field_coupling`. The fixture must keep the
global-field phase, field amplitude, coupling gain, randomised-global-phase
null, and zero-field baseline explicit.
