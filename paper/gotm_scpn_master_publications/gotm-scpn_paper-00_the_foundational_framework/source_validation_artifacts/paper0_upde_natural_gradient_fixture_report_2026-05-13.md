# Paper 0 UPDE Natural-Gradient Fixture Report

Date: 2026-05-13

## Source Boundary

Spec: `upde.natural_gradient`

Protocol: `paper0.upde.natural_gradient.fim_free_energy`

Source equation: `EQ0042`

Source ledger: `P0R02642`

Hardware status: `simulator_only_no_provider_submission`

## Executable Fixture

Implementation:

- `src/scpn_quantum_control/paper0/upde_natural_gradient_validation.py`
- `tests/test_paper0_upde_natural_gradient_validation.py`

The fixture implements the source equation component

`dot(theta)^L = -eta_L g_F^{-1} grad_theta F_L`

on a differentiable quadratic free-energy fixture. It keeps the Fisher metric,
gradient, descent rate, finite-difference check, Euclidean-gradient null, and
singular-metric boundary explicit.

## Controls

The current executable fixture records:

- finite-difference gradient agreement;
- Euclidean-gradient versus natural-gradient drift difference;
- identity-FIM control, which must match Euclidean descent;
- regularised singular-metric response magnitude;
- fail-fast rejection for non-positive-definite FIMs.

## Measured Local Result

Result artefact:

- `paper0_upde_natural_gradient_fixture_result_2026-05-13.json`

Measured on the local three-coordinate fixture:

- free energy: `0.175945`;
- finite-difference gradient error, L-infinity:
  `4.081540661005079e-12`;
- FIM condition number: `2.07465066550813`;
- Euclidean-versus-natural drift difference, L2:
  `0.10222084991284003`;
- identity-FIM versus Euclidean drift, L-infinity: `0.0`;
- regularised singular-metric response, L-infinity: `119880.0`;
- single fixture runtime: `1.7767710087355226 ms`.

## Verification

- `PYTHONPATH=src .venv-linux/bin/python -m pytest tests/test_paper0_upde_natural_gradient_validation.py -q`
  - `5 passed in 0.16s`

## Next Step

Extend this pattern to `upde.adaptive_coupling`. The fixture must keep bounded
adaptive gains, target order parameter, criticality target, wrong-sign feedback
null, zero-gain null, and unbounded-gain rejection explicit.
