# Paper 0 UPDE Base-Phase Fixture Report

Date: 2026-05-13

## Source Boundary

Spec: `upde.base_phase`

Protocol: `paper0.upde.base_phase.xy_gradient_and_locking`

Source equations: `EQ0003`, `EQ0032`, `EQ0037`, `EQ0039`, `EQ0129`

Source ledgers: `P0R00520`, `P0R02507`, `P0R02530`, `P0R02622`, `P0R06120`

Hardware status: `simulator_only_no_provider_submission`

## Executable Fixture

Implementation:

- `src/scpn_quantum_control/paper0/upde_validation.py`
- `tests/test_paper0_upde_base_phase_validation.py`

The fixture validates the base UPDE/Kuramoto phase law

`d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i)`

against the negative gradient of the symmetric negative-cosine potential. This
is valid for finite, symmetric, real-valued `K_nm` with zero diagonal and finite
`omega` and `theta`.

## Controls

The current executable fixture records:

- zero-coupling drift control;
- sign-flipped coupling response;
- deterministic topology-shuffle response;
- weak-versus-strong off-onset order-parameter delta.

## Measured Local Result

Result artefact:

- `paper0_upde_base_phase_fixture_result_2026-05-13.json`

Measured on the local 4-oscillator fixture:

- gradient error, L-infinity: `2.0708079695452852e-10`;
- zero-coupling drift error, L-infinity: `0.0`;
- sign-flip response, L2: `0.9612504667356742`;
- shuffled-topology response, L2: `1.3307731499410094`;
- off-onset order-parameter delta: `0.010172365312265952`;
- single fixture runtime: `1.7314529977738857 ms`.

## Verification

- `PYTHONPATH=src .venv-linux/bin/python -m pytest tests/test_paper0_upde_base_phase_validation.py -q`
  - `4 passed in 0.17s`

## Next Step

Extend this pattern to `upde.interlayer_coupling` only after direct inspection
of the candidate FEP, bridge, and feedback code paths. Keep that family marked
`validation_spec_pending_direct_implementation_audit` until its source equation,
tensor shapes, perturbation controls, and simulator fixture are verified.
