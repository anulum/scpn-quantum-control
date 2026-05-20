# Paper 0 UPDE Adaptive-Coupling Fixture Report

Date: 2026-05-13

## Source Boundary

Spec: `upde.adaptive_coupling`

Protocol: `paper0.upde.adaptive_coupling.quasicritical_controller`

Source equation: `EQ0045`

Source ledger: `P0R02910`

Hardware status: `simulator_only_no_provider_submission`

## Executable Fixture

Implementation:

- `src/scpn_quantum_control/paper0/upde_adaptive_coupling_validation.py`
- `tests/test_paper0_upde_adaptive_coupling_validation.py`

The fixture implements the source update laws

`dot K_ij^L = gamma_L(R_L - R_L*) - lambda_L K_ij^L + xi_ij^L(t)`

and

`dot eta^L = -alpha_L(sigma_L - 1)`

on a finite symmetric zero-diagonal coupling matrix. The zero diagonal is kept
as a graph invariant: no self-coupling edge is created by the adaptive update.

## Controls

The current executable fixture records:

- zero-gain null with `gamma_L = lambda_L = alpha_L = 0` and zero noise;
- wrong-sign feedback response for `gamma_L` and `alpha_L`;
- bounded update magnitude;
- rejection of non-finite matrices, non-symmetric noise, negative decay, and
  gains exceeding `max_abs_gain`.

## Measured Local Result

Result artefact:

- `paper0_upde_adaptive_coupling_fixture_result_2026-05-13.json`

Measured on the local three-node coupling fixture:

- `K_dot` L-infinity: `0.1405`;
- `eta_dot`: `-0.08099999999999997`;
- `K_next` L-infinity: `0.302975`;
- `eta_next`: `0.39595`;
- bounded update L-infinity: `0.007025000000000001`;
- zero-gain `K_dot` L-infinity: `0.0`;
- zero-gain `eta_dot` absolute value: `0.0`;
- wrong-sign `K_dot` response, L2: `0.509493866498901`;
- wrong-sign `eta_dot` response: `0.16199999999999995`;
- single fixture runtime: `1.2968219962203875 ms`.

## Verification

- `PYTHONPATH=src .venv-linux/bin/python -m pytest tests/test_paper0_upde_adaptive_coupling_validation.py -q`
  - `4 passed in 0.22s`

## Next Step

The first UPDE validation family is now covered by executable simulator
fixtures. The next programme step is to create an aggregate UPDE validation
index that records all five fixture statuses and then move to the next Paper 0
mechanism family.
