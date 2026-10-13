# DLA-parity statistical reproducer — input domain and scalar inventory

First delivery of the programme-supplier task for the result-pack tool. It describes the programme **as it is**:
`scpn_quantum_control.dla_parity.reproduce.reproduce_statistics` with its loader and record types. No behaviour is
changed by this delivery and nothing here qualifies the programme for any role. The machine-readable companion is
`scalar_inventory_v1.json` in this directory; it was generated from the dataclasses and from one run of the
programme on the committed data, and it pins the three source files, the four raw run files and the published
summary by byte length and SHA-256.

## What the programme does

1. **Loads** four run files in publication order. Each must hold eight top-level keys and, per circuit, eight
   metadata keys and a text-to-integer count table. A digest check against the four digests in the loader exists
   and is **off by default**.
2. **Selects** circuits with four qubits, an experiment name beginning `A_dla_parity_n4`, and sector `even` or
   `odd`. Everything else is skipped without a record.
3. **Per circuit**, recomputes parity leakage from the raw counts: the fraction of shots whose bitstring has a
   different parity of ones than the initial state. The precomputed statistics inside the records are not read.
4. **Per depth**, from the leakages of the even and the odd sector: mean, standard error (sample standard
   deviation over the square root of the count), relative asymmetry `(mean_even − mean_odd) / mean_odd`, and Welch's
   unequal-variance t statistic with its two-sided p-value.
5. **Across depths**, Fisher's combined statistic `−2 Σ ln p` over the usable p-values, its degrees of freedom and
   combined p-value, the count of p-values below 0.05, the peak and the unweighted mean of the relative asymmetry.
6. **Compares** recomputed scalars with a published summary file under a tolerance bundle and raises
   `AssertionError` at the first claim that fails.

There is no random number in the programme and no rounding: all arithmetic is IEEE-754 binary64.

## Comparison rule, exactly

- *absolute* claims: `abs(published − recomputed) <= tolerance`.
- *scaled* claims (`welch_t`, `welch_p`, `fisher.chi2`):
  `abs(published − recomputed) / max(abs(published), abs(recomputed), 1.0) <= tolerance`.
  **This is not a relative comparison for values below one**: the divisor is then 1.0 and the rule is absolute.
- Both values NaN pass; exactly one NaN fails.

A consumer that compares with `abs(observed − recorded) <= absolute + relative × abs(recorded)` applies a
different rule and must not be treated as equivalent.

## On the committed data (one run, digest check on)

342 circuits in four runs on one backend; 282 selected; eight depths (2, 4, 6, 8, 10, 14, 20, 30) with 12 or 21
replicates per sector; 57 claims checked (seven per depth and the Fisher statistic), largest difference 0.0. Every
recomputed scalar is finite. The published summary has a row for every computed depth and no other.

## Findings — where the programme drops, hides or weakens something

Each is a property of the current code. None shows on the committed data unless said so; each matters once the
programme is run on other inputs, which is the role it is nominated for.

| # | Finding | Consequence |
| --- | --- | --- |
| 1 | A computed depth with no row in the published summary is skipped without a claim. | Scalars of that depth are returned but never compared; the result does not say so. |
| 2 | A published depth that the data does not produce is never noticed: the loop runs over computed depths only. | A missing depth passes as success. |
| 3 | A published row that is not an object is skipped silently. | A malformed summary can pass. |
| 4 | Of the published per-depth fields, `n_reps_even`, `n_reps_odd`, `degrees_of_freedom`, `ci95_even`, `ci95_odd` and `asymmetry_absolute` are not compared; the last four are not computed at all. | The replicate counts behind every statistic are unchecked. |
| 5 | Of the Fisher result only `chi2` is compared; `degrees_of_freedom`, `combined_p`, the significant count and the tested count are not. | A change in which p-values are usable alters four scalars with no failing claim. |
| 6 | `mean_asymmetry_relative` and `peak_asymmetry_relative` are compared only when the summary has those keys. **The committed summary has neither**, so both are computed and never compared today; `peak_asymmetry_depth` is never compared. | Three returned scalars carry no check. |
| 7 | The combined p-value is computed as `1 − cdf`. **On the committed data it is 0.0**; the survival function gives 1.2e−18. The published value is also 0.0. | A reported p of exactly zero is a loss of precision, not a result. |
| 8 | Fisher's statistic uses only p-values in `(0, 1]`; a NaN or a zero is left out without a record, and `n_depths_tested` still counts every depth. | The degrees of freedom shrink silently; "tested" does not mean "used". |
| 9 | For the scaled rule the tolerance of 1e−6 on `welch_p` is absolute. The smallest p on the committed data is 1.4e−6. | That claim would pass with the p-value wrong by more than half of itself. |
| 10 | With fewer than two replicates in a sector the standard error is returned as 0.0, and the Welch values as NaN. | A zero error bar that means "undefined". |
| 11 | A circuit with zero shots gives NaN leakage, which enters the sector mean. | One empty record turns every scalar of its depth to NaN; the comparison then fails or, against a NaN, passes. |
| 12 | The declared shot count is not compared with the sum of counts; bitstring length and alphabet are not checked; the sector label is not compared with the parity of the initial state. | The per-circuit scalar trusts three things it could verify. (On the committed data: declared and counted shots agree for all 282 selected circuits and all bitstrings have length four.) |
| 13 | Selection is by a prefix of the experiment name. Sixteen four-qubit circuits named `I_n4_extended_depth` are excluded with the scaling and baseline circuits. | Whether they belong to the domain is a scientific decision that the code makes by spelling. |
| 14 | An empty selection returns NaN scalars and depth −1 and then fails on a published key or a NaN mismatch, not with a refusal that names the cause. | No distinct "nothing to analyse" outcome. |
| 15 | The peak is the first maximum in ascending depth; a NaN asymmetry makes the peak and the mean depend on position or become NaN. | Tie and NaN behaviour are implicit. |
| 16 | The default data directory and the default published summary are found relative to the source checkout (three directories above the module). | Inside an installed package those paths do not exist; the defaults work only from a clone. |
| 17 | The mismatch signal is `AssertionError`, the first failure ends the run, and a missing key in the summary surfaces as `KeyError`. | A caller cannot collect all mismatches or tell categories apart by type. |
| 18 | The digest check is optional and off by default. | The same call can run on unverified bytes. |

## What the next delivery fixes, and what it leaves

The worker of the next delivery takes every input as an explicit, digest-pinned path, verifies the digests before
reading, emits every scalar of the inventory in one deterministic document, and turns findings 1–3, 8, 10, 11, 12,
14, 16, 17 and 18 into either an emitted scalar or a refusal with a fixed category. It does **not** change the
numbers the existing function returns on the committed data. Findings 4–7, 9 and 13 touch what is compared and
which circuits belong to the claim; they are stated to the consumer and to the architecture owner as open
decisions, and the worker reports the affected scalars without a comparison claim rather than inventing one.
