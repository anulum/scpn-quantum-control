# L16 director functional evidence

- Schema: `l16_director_product.v2`
- Functional passed: `true`
- Promotion ready: `false`
- Action diversity: `false`
- Content digest: `a35948cb2ce54e0915fefb04fb74dcac2e8ed62f39cabcc92ed70aac7e4c8c6a`
- Execution: bounded local exact simulation; no provider, QPU, or hardware actuation.

## Indicator certificates

| Scenario | Echo | Variance | Susceptibility | R | Score | Heuristic | Safety action | Informative |
|---|---:|---:|---:|---:|---:|---|---|---|
| paper27_baseline | 1 | 0 | 0 | 1 | 1 | continue | allow | none |
| susceptibility_probe | 1 | 5.55111512e-17 | 14.1898827 | 1 | 0.766458323 | continue | allow | fidelity_susceptibility |
| weak_coupling_probe | 1 | 0 | 0 | 1 | 1 | continue | allow | none |

## Governed routes

| Route | Status | Boundary |
|---|---|---|
| adapter:l16.local_indicator | supported | bounded local only |
| adapter:l16.autonomous_hardware_control | permanent_boundary | weighted indicator composite is not a Lyapunov, PCS, or stability certificate; owner-ticketed hardware and partner control validation cannot be inferred |

## Promotion blockers

- weighted composite is a heuristic policy, not a Lyapunov or PCS certificate
- no provider, QPU, plant, or realtime-hardware execution
- no closed-loop stability theorem or causal diagnosis
- frozen real scenarios did not establish action diversity
- at least one scenario has fewer than two nontrivial raw indicators

## Claim boundary

bounded exact-simulator L16 indicator and heuristic safety-routing evidence; no classical or quantum Lyapunov-exponent proof, PCS certificate, causal diagnosis, stability guarantee, autonomous actuation, provider, QPU, or production-control claim
