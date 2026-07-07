# S1d IBM Policy-Direction Sweep Result

Date: 2026-05-20
Backend: `ibm_kingston`
Experiment ID: `s1d_policy_direction_sweep_2026-05-20`
Parent experiment: `s1_dynamic_feedback_preregistration_2026-05-06`

## Artefacts

- Readiness: `data/s1_feedback_loop/s1d_xy_observable_readiness_ibm_kingston_20260520T134614Z.json`
- Readiness SHA-256: `ecbfc54979c86159ca40a05da2249bf7fb807963893f2a812b3a07fcf36258d7`
- Raw counts: `data/s1_feedback_loop/s1d_xy_observable_raw_counts_ibm_kingston_20260520T134614Z.json`
- Raw counts SHA-256: `f26f3c442a2e91a215e47ec6bf0a515170a70cede3ce0ee4639ba1e7b121129d`
- Analysis: `data/s1_feedback_loop/s1d_xy_observable_analysis_ibm_kingston_20260520T134614Z.json`
- Analysis SHA-256: `6e6bc2f513f23633c72756016392b58cbf140d89c6f9cc7fdd07880c66b61c76`

## Purpose

S1d is a same-paper continuation of S1/S1b/S1c. It tests whether the observed
S1c weakness is best explained by an incorrect feedback correction direction,
excess gain, or a more fundamental limitation of the monitored-feedback policy
on the selected backend/calibration window.

The sweep keeps the S1c one-round shallow body and direct XY observable family,
but compares three preregistered policy variants:

| Variant | Rounds | Correction angle | Base gain | Purpose |
|---|---:|---:|---:|---|
| `current_shallow_positive` | 1 | `0.06` | `0.4` | Repeat the completed S1c shallow positive-correction policy. |
| `polarity_flipped` | 1 | `-0.06` | `0.4` | Test whether S1c's negative shift was a correction-polarity effect. |
| `weak_positive` | 1 | `0.03` | `0.2` | Test whether positive correction needs a lower-gain stability boundary. |

All policy variants passed readiness with maximum transpiled depth `237` and a
total estimated QPU budget of `72.0` seconds under the `120.0` second ceiling.

## Execution Notes

The first approved submission produced 14 completed IBM jobs before the local
approval scheduler stopped because provider wall-clock wait time was reported
as `qpu_seconds`. The accounting bug was fixed so QPU budget accounting uses the
preregistered arm estimate while preserving provider wait time as
`wall_time_s`. The run was resumed without duplicating completed arms: 14 jobs
were recovered from IBM job lookup and the 10 missing arms were submitted.

Recovered jobs:

- `d86rm9h789is7390302g`
- `d86rmbas46sc73f7cufg`
- `d86rmd5g7okc73elhcsg`
- `d86rmeh789is739030a0`
- `d86rmg9789is739030bg`
- `d86rmhop0eas73dlcipg`
- `d86rmjlg7okc73elhd40`
- `d86rml2s46sc73f7curg`
- `d86rmmis46sc73f7cut0`
- `d86rmo0p0eas73dlcj10`
- `d86rmpop0eas73dlcj2g`
- `d86rmr8p0eas73dlcj50`
- `d86rmsp789is739030u0`
- `d86rmulg7okc73elhdk0`

Resume-submitted jobs:

- `d86roris46sc73f7d1b0`
- `d86rq18p0eas73dlcmj0`
- `d86rqkp789is7390358g`
- `d86rqmh789is739035c0`
- `d86rqo5g7okc73elhhs0`
- `d86rqpp789is739035fg`
- `d86rqras46sc73f7d3ig`
- `d86rqt1789is739035kg`
- `d86rquop0eas73dlcnig`
- `d86rr0gp0eas73dlcnl0`

## Results

Feedback minus matched open-loop control by sorted direct-XY observable row:

| Variant | IXX | IYY | XXI | YYI | Mean signed | Mean absolute |
|---|---:|---:|---:|---:|---:|---:|
| `current_shallow_positive` | `-0.0078125` | `-0.0136718750` | `-0.0605468750` | `0.1848958333` | `0.0257161458` | `0.0667317708` |
| `polarity_flipped` | `-0.0130208333` | `-0.0065104167` | `0.0156250000` | `-0.0123697917` | `-0.0040690104` | `0.0118815104` |
| `weak_positive` | `-0.0026041667` | `-0.0039062500` | `0.0065104167` | `-0.0201822917` | `-0.0050455729` | `0.0083007813` |

By mean signed feedback-control delta, the best variant is
`current_shallow_positive`. This ranking is driven by the `YYI` channel; it does
not by itself establish a stable feedback improvement.

## Interpretation

S1d does not close the paper with a simple positive-feedback claim. It shows
that the direct-XY response is policy- and calibration-window-sensitive. The
completed S1c run moved all four direct-XY channels negative, while the S1d
repeat of the same shallow positive policy produced one large positive `YYI`
shift and three negative shifts. The polarity-flipped and weak-positive variants
are closer to zero and do not rescue the policy.

The conservative same-paper conclusion is therefore diagnostic: the tested
monitored-feedback law is not robustly beneficial at N = 3 system qubits on this
backend class, but the direct-XY sector exposes structured policy sensitivity
that is hidden by the original binary proxy. Any stronger claim requires a
predeclared statistical repeat or a redesigned policy law, not post hoc selection
of the favourable `YYI` channel.
