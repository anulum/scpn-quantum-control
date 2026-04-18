<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Classical irreproducibility of the DLA-parity asymmetry

The DLA-parity hardware campaign on `ibm_kingston` measured a
non-zero asymmetry between the "even" and "odd" initial-state
sectors: peak +17.48 % at depth 6, mean +9.25 % across the eight
depths surveyed, with Fisher combined $\chi^{2} = 123.4$ across
sixteen degrees of freedom. The observation is real in the
statistical sense. This page states precisely which reading of
"classical simulators cannot reproduce this" is supported by the
evidence and which reading is not, closing audit item **D1** with
a narrow claim and an explicit, honest limitation.

## Two readings of "quantum result beyond classical"

| Reading | Claim | This campaign |
| --- | --- | --- |
| **Narrow — signature irreproducibility** | No classical simulator of the *idealised* Kuramoto-XY Hamiltonian $H = \sum_{i} \omega_{i} Z_{i} + \sum_{\langle i,j \rangle} K_{ij}(X_{i}X_{j} + Y_{i}Y_{j})$ can reproduce the observed asymmetry, because the Hamiltonian conserves total parity exactly. | **Proved below.** |
| **Broad — quantum computational advantage** | No classical algorithm, given polynomial resources, can simulate the hardware evolution at the system size used in the campaign. | **Not proved.** Phase 1 used $n = 4$ qubits; classical simulation of this system is trivial (we perform it routinely in this repository). |

These are different statements. Running them together — "the
observed asymmetry is beyond classical" without specifying which
reading is meant — is overclaim. The narrow reading is the honest
interpretation of the Phase 1 data; the broad reading requires
scaling to $n \gtrsim 30$ and closing different open problems
tracked as Gap 1 and Gap 2 in
[`GAP_CLOSURE_STATUS.md`](https://github.com/anulum/scpn-quantum-control/blob/main/GAP_CLOSURE_STATUS.md).

## Narrow claim: proof by total-parity conservation

**Statement.** Let $H$ be the campaign Hamiltonian and let
$P = \prod_{i} Z_{i}$ be the total-parity observable. Then
$[H, P] = 0$, and consequently any unitary $U(t) = e^{-i H t}$
(including every Lie-Trotter approximation of it built from $H_{Z}$
and $H_{XY}$ separately) commutes with $P$. If the initial state
$|\psi_{0}\rangle$ is an eigenstate of $P$ with eigenvalue $\pm 1$,
then every subsequent state remains an eigenstate with the same
eigenvalue, and the probability of measuring any opposite-parity
computational basis string is identically zero.

**Proof.** Every summand of $H$ commutes with $P$ individually:

* $[Z_{i}, P] = 0$ because $P$ is a product of $Z$ operators and
  $[Z_{i}, Z_{j}] = 0$.
* $[X_{i}X_{j}, P]$: $X_{i}$ anticommutes with $Z_{i}$ and commutes
  with every other $Z_{k}$; similarly for $X_{j}$. Conjugating
  $X_{i}X_{j}$ by $P$ therefore picks up two sign flips, i.e.
  $(-1)^{2} = +1$. So $P X_{i} X_{j} P^{-1} = X_{i} X_{j}$, hence
  $[X_{i}X_{j}, P] = 0$.
* $[Y_{i}Y_{j}, P] = 0$ by the same argument — $Y$ also
  anticommutes with $Z$ on the same site.

Since each summand commutes with $P$, the sum does, so $[H, P] = 0$
and consequently $[e^{-iHt}, P] = 0$ for all real $t$. Because
$H_{Z}$ and $H_{XY}$ each commute with $P$ individually, every
Lie-Trotter factor $e^{-i H_{Z} \tau}$ and $e^{-i H_{XY} \tau}$ also
commutes with $P$, and so does every composed step and every power
of the step.

**Initial parity eigenvalues.** The even sector uses
$|\psi_{0}\rangle = |0011\rangle$ with popcount 2
($P|\psi_{0}\rangle = +|\psi_{0}\rangle$); the odd sector uses
$|0001\rangle$ with popcount 1 ($P|\psi_{0}\rangle = -|\psi_{0}\rangle$).
Applying the commuting unitary preserves each eigenvalue exactly,
so

$$\langle \psi_{0}| U(t)^{\dagger} \Pi_{\text{opp}} U(t) |\psi_{0}\rangle = 0$$

where $\Pi_{\text{opp}}$ is the projector onto computational basis
states of opposite parity. Measuring in the computational basis
on the evolved state therefore returns zero leakage for every
depth and every $t_{\text{step}}$.

## Mechanical verification in the repository

The proof is not left as prose. It is enforced by the test suite
and the numerical reference.

* [`tests/test_classical_irreproducibility.py`](https://github.com/anulum/scpn-quantum-control/blob/main/tests/test_classical_irreproducibility.py)
  (28 tests) performs three independent algebraic checks at every
  relevant system size:

    1. `[H, P] = 0` as a `SparsePauliOp` identity — each term pair
       is reduced symbolically, and every residue coefficient is
       numerically below $10^{-12}$ (in practice 0.0 exactly). Run
       at $n = 3, 4, 6$.
    2. `[H_{Z}, P] = [H_{XY}, P] = 0` individually, i.e. each
       Lie-Trotter generator commutes with $P$ by itself. This is
       the stronger statement that makes the Trotter decomposition
       exact on $P$, not just first-order accurate.
    3. For a parity-eigenstate initial condition, the opposite-parity
       mask applied to $U(t) |\psi_{0}\rangle$ has total
       probability below $10^{-18}$ at depth 1, 4, 10, and 30 and
       at $t_{\text{step}}$ = 0.1, 0.3, and 1.7.

* [`src/scpn_quantum_control/dla_parity/baselines.py`](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/dla_parity/baselines.py)
  computes the noiseless leakage reference across the published
  depth sweep using two independent backends (numpy dense and
  optional QuTiP). Both report `max_abs_leakage < 1e-10` and
  agree with each other to $10^{-12}$.

* [`tests/test_cross_validation_qutip_dynamiqs.py`](https://github.com/anulum/scpn-quantum-control/blob/main/tests/test_cross_validation_qutip_dynamiqs.py)
  separately confirms that the Hamiltonian matrix itself agrees
  between our internal Qiskit-based builder, QuTiP, and
  Dynamiqs to $10^{-10}$ relative. The invariant $[H, P] = 0$
  is therefore a property of the physics we specified, not an
  artefact of the specific Hamiltonian-assembly code.

## What this does *not* say

* **Nothing about quantum advantage at scale.** The classical
  evolution at $n = 4$ runs in milliseconds on a laptop. The
  statement here is strictly about reproducibility of the
  *specific observed signature*, not about complexity-class
  separations. The broad claim ("no efficient classical
  algorithm can simulate this for any $n$") remains open and is
  tracked under Gap 2 in `GAP_CLOSURE_STATUS.md`.
* **Nothing about "hidden" classical models.** The statement
  covers classical simulators that faithfully implement the
  idealised Hamiltonian. It does not rule out *noise-matched*
  classical models that deliberately inject a biased depolarising
  channel to recreate the observed asymmetry empirically. Such a
  model would have to reproduce the *exact* depth and sector
  dependence, and would be measurement-driven fitting rather
  than first-principles simulation.

## Why the hardware asymmetry is non-zero anyway

Because the parity-conservation proof depends on $U(t)$ being
*exactly* the idealised unitary, and hardware circuits are not
exactly unitary. Any physical channel that allows a single bit to
flip — relaxation during an idle, measurement crosstalk, a
miscalibrated single-qubit rotation — breaks parity conservation
in the measured distribution. The observed asymmetry between
even and odd sectors is therefore a **signature of the hardware's
noise channel interacting with the chosen initial state**, not a
property of the Hamiltonian. The short paper
[`paper/phase1_dla_parity.tex`](https://github.com/anulum/scpn-quantum-control/blob/main/paper/phase1_dla_parity.tex)
presents this as a hardware-calibration observation, not as a
quantum-advantage claim, and the distinction made in the table
above is the reason.

## Relationship to falsification criterion C8

`docs/falsification.md` registers a Phase 1 reproducibility
criterion: the 342-circuit dataset, rerun through the analysis
pipeline, must recover $\chi^{2} = 123.4 \pm 1$ and the
${+}17.5 \%$ peak asymmetry at depth 6. That criterion guards
against regression in our analysis code. The present document
guards against a different failure mode: **overclaim of scope**.
Both are load-bearing: the reproduction proves our analysis is
stable; the parity-conservation argument proves the hardware
signature is the only source of the result; the open Gap 2
proves we are not yet claiming computational advantage.

## Audit closure

This document and `tests/test_classical_irreproducibility.py`
together close the narrow reading of audit item **D1** from
[`docs/internal/audit_2026-04-17T0800_claude_gap_audit.md`](internal/audit_2026-04-17T0800_claude_gap_audit.md).
The broad reading remains open and is tracked in
[`GAP_CLOSURE_STATUS.md`](https://github.com/anulum/scpn-quantum-control/blob/main/GAP_CLOSURE_STATUS.md) as Gap 2,
with the concrete next step being scaling the campaign via
circuit cutting and/or MPS-falsifying regimes.
