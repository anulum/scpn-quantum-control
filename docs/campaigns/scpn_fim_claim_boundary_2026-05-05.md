# SCPN/FIM Claim Boundary

Date: 2026-05-05

This document converts the generated SCPN/FIM artefacts into manuscript-safe
claim language. It is deliberately conservative: a claim is usable only if it is
backed by a committed artefact and does not exceed the evidence boundary.

## Artefact base

| Artefact group | Files | Status |
| --- | --- | --- |
| Exact spectra | `fim_spectrum_summary_2026-05-05.json`, `.csv` | generated |
| Magnetisation-sector spectra | `fim_sector_spectrum_summary_2026-05-05.csv` | generated |
| Adjacent-gap ratios | `fim_level_spacing_summary_2026-05-05.json`, `.csv` | generated |
| Bipartition entropy | `fim_entanglement_summary_2026-05-05.json`, rows/aggregate CSV | generated |
| Sector-conservation checks | `fim_sector_survival_prediction_2026-05-05.json`, summary/rows CSV | generated |
| VQE ground-state scoring | `fim_vqe_ground_state_summary_2026-05-05.json`, rows/aggregate CSV | generated |
| IBM pilot candidate | `fim_ibm_candidate_protocol_2026-05-05.json`, `.csv` | generated, not submitted |

## Manuscript-safe claims

| Claim | Status | Supporting artefact | Safe wording | Limitation |
| --- | --- | --- | --- | --- |
| The FIM term changes the exact small-n energy landscape. | validated offline | `fim_spectrum_summary_2026-05-05.json` | For n=4,6,8, increasing `lambda` lowers the ground energy and increases the spectral gap/width in the generated dense exact spectra. | Exact dense small-n only. |
| The FIM term creates explicit magnetisation-sector energy shifts. | validated offline | `fim_sector_spectrum_summary_2026-05-05.csv` | The `-lambda M^2/n` term separates magnetisation sectors by a known diagonal energy contribution. | This is a Hamiltonian-structure statement, not a hardware robustness result. |
| The ideal Hamiltonian conserves total magnetisation. | validated offline | `fim_sector_survival_prediction_2026-05-05.json` | The generated commutator and off-sector block checks are zero for the tested grid, so ideal `H_XY + H_FIM` has no unitary sector leakage. | Hardware leakage must be treated as noise/circuit/readout behaviour. |
| Low-energy entanglement changes with larger lambda in the tested grid. | validated offline | `fim_entanglement_summary_2026-05-05.json` | For n=4, the mean low-energy bipartition entropy decreases from about 0.676 bits at `lambda=0` to about 0.490 bits at `lambda=8`. | Low-energy exact eigenstate diagnostic only; not a universal localisation proof. |
| Adjacent-gap ratios shift on n=6 and n=8 as lambda increases. | validated offline | `fim_level_spacing_summary_2026-05-05.json` | The full-spectrum adjacent-gap mean is lower at large lambda for n=6 and n=8 in the generated artefact. | Small-n exact diagnostic; use "localisation-like" only with caveats. |
| A topology-informed ansatz performs better than generic baselines in the n=4 FIM VQE grid. | validated offline | `fim_vqe_ground_state_summary_2026-05-05.json` | In the n=4, reps=2, three-seed benchmark, the K_nm-informed ansatz has lower median relative energy error than `TwoLocal` and `EfficientSU2` for `lambda in {0,1,4}`. | Small optimiser budget and n=4 only; not an ansatz theorem. |
| The IBM pilot is ready as a candidate protocol. | protocol-ready, not executed | `fim_ibm_candidate_protocol_2026-05-05.json` | A non-submitting n=4 pilot protocol exists with lambda/depth/state/readout controls and a falsification rule. | Requires backend selection, live transpilation, QPU-time estimate, and explicit approval. |
| The IBM pilot raw counts exist. | data collected, analysis pending | `fim_ibm_pilot_raw_counts_2026-05-05_ibm-run-4c0bd60c3fc2c532.json` | The n=4 SCPN/FIM pilot completed on `ibm_kingston` with 61 circuits and 249856 shots. | No result claim is promoted until the analysis and readout-control scripts are run. |

## Claims that remain blocked

| Blocked claim | Reason |
| --- | --- |
| The FIM term improves real IBM hardware coherence. | FIM IBM raw counts now exist, but the analysis/readout-control result has not yet been computed or promoted. |
| The FIM term causes ideal magnetisation-sector leakage suppression. | The ideal Hamiltonian has zero sector leakage; suppression is not the correct framing. |
| The result demonstrates quantum advantage. | All current SCPN/FIM artefacts are exact small-n offline computations or protocol design. |
| The model proves strict many-body localisation. | Current evidence is level-spacing and entanglement diagnostics on n=4,6,8 only. |
| The FIM mechanism is universally protective. | No multi-size hardware or open-system validation exists. |

## Quantitative anchors for the current draft

Exact spectrum:

- n=4: spectral gap increases from 1.1317 at `lambda=0` to 12.6060 at
  `lambda=4`.
- n=6: spectral gap increases from 0.4468 at `lambda=0` to 13.7801 at
  `lambda=4`.
- n=8: spectral gap increases from 0.2827 at `lambda=0` to 13.7173 at
  `lambda=4`.

Adjacent-gap ratios:

- n=6 full-spectrum mean adjacent-gap ratio changes from 0.4334 at `lambda=0`
  to 0.4071 at `lambda=4`.
- n=8 full-spectrum mean adjacent-gap ratio changes from 0.4368 at `lambda=0`
  to 0.4048 at `lambda=4` and 0.3852 at `lambda=8`.

Entanglement:

- n=4 mean low-energy bipartition entropy is about 0.676 bits for
  `lambda <= 1`.
- n=4 mean low-energy bipartition entropy decreases to about 0.584 bits at
  `lambda=2`, 0.502 bits at `lambda=4`, and 0.490 bits at `lambda=8`.

VQE:

- K_nm-informed ansatz median relative error is 0.372%, 0.413%, and 0.848% at
  `lambda=0`, `lambda=1`, and `lambda=4`, respectively.
- Generic baselines in the same grid remain at multi-percent median error.

IBM candidate:

- n=4.
- `lambda in {0, 1, 4}`.
- depths `{2, 4, 6}`.
- five representative magnetisation sectors.
- full 16-state readout baseline.
- 4096 shots per candidate circuit.
- 61 candidate circuits and 249856 candidate shots.
- submission status: `not_submitted`.
- local circuit-preparation artefact exists with max local transpiled depth 262
  and max local two-qubit gate count 144; live backend transpilation is still
  required before any QPU submission.
- live non-submitting backend transpilation on `ibm_kingston` exists with max
  depth 540 and max two-qubit gate count 157; QPU submission still requires
  review and explicit approval.

## Recommended paper framing

Use:

> We introduce and characterise a collective Fisher-information-inspired
> magnetisation-feedback term in a Kuramoto-XY Hamiltonian. Exact small-n
> artefacts show sector-energy separation, gap widening, shifts in spectral and
> low-energy entanglement diagnostics, and a hardware-facing pilot protocol.

Avoid:

> The FIM term has been shown to protect quantum coherence on hardware.

Avoid:

> The model demonstrates quantum advantage or strict many-body localisation.

## Next decision

The project can now either:

1. Draft the paper as an offline theory/computation note with a proposed IBM
   pilot.
2. Refine the pilot through live backend/transpilation checks before writing the
   hardware-facing section.
3. Add a noise-model sector-survival harness before spending QPU time.
## 2026-05-05 IBM pilot boundary update

The n=4 SCPN/FIM pilot on `ibm_kingston` completed as job
`ibm-run-4c0bd60c3fc2c532`. It is a valid hardware pilot, but it is not a
positive hardware-protection result.

Allowed claims:

- The SCPN/FIM circuit family was executed on IBM Heron r2 hardware.
- The raw count dictionaries, row metrics, lambda-trend comparisons, and
  readout-baseline summaries are archived in `data/scpn_fim_hamiltonian/`.
- The run is useful for protocol debugging and for designing a repeated,
  randomized follow-up campaign.

Blocked claims:

- Do not claim that `H_FIM = -lambda M^2 / n` improved hardware coherence in
  this pilot.
- Do not claim hardware many-body localisation.
- Do not report hardware p-values from this pilot; there is one sample per
  lambda/depth/state condition.
- Do not describe the readout correction as full confusion-matrix mitigation.

## 2026-05-05 repeated follow-up boundary update

The repeated/randomized SCPN/FIM follow-up completed as IBM job
`ibm-run-cf4835290f607387`.

This is now stronger than a pilot boundary: the simple hardware-protection
interpretation is falsified for the tested `ibm_kingston` circuit family.

Allowed claims:

- The repeated run gives evidence that, in the tested implementation,
  `lambda = 4` increases leakage relative to `lambda = 0`.
- The result is a valid negative hardware result for this backend/circuit family.

Blocked claims:

- Do not claim that FIM improves hardware coherence in this implementation.
- Do not claim backend-general behaviour.
- Do not claim hardware many-body localisation.
- Do not generalize the negative result to all possible FIM Hamiltonian
  implementations; it applies to this digital Trotter construction, depth set,
  backend, calibration window, and observable set.
