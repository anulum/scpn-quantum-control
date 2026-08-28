# RL research-governance evidence

- Schema: `scpn.rl-research-governance.v1`
- Policy: `rl-policy-5308e77754f9f46a`
- Preregistration: `RL-GOVERNANCE-LOCAL-WITNESS-REPLAY-v1`
- Content digest: `bfee938832bd91d28add062acefaa78205ac5fa57dc5674eeec97c33fa1319e1`
- Result: **PASS**
- Environment API: `not_applicable_static_candidate_search` (no Gym environment is introduced)
- Reward: `witness_score_dense_composite_v1` (dense composite; component gaming remains a research risk)
- Boundary: local seeded research replay only; no product policy, autonomous control, provider submission, QPU or pulse execution, hardware validity, advantage, optimal-policy, realtime, clinical, or publication claim.

| Seed | Evaluations | Best composite score | Replay identical | Best candidate |
|---:|---:|---:|---|---|
| 104729 | 5 | 2.44503888757 | true | coupling_scale=1.97894575458, omega_scale=0.99058231326, phase_bias=-0.5 |
| 130363 | 5 | 2.44382012437 | true | coupling_scale=1.95039701323, omega_scale=0.979721836472, phase_bias=-0.00472922976752 |
| 155921 | 5 | 2.44662184753 | true | coupling_scale=1.86538263485, omega_scale=0.678327680165, phase_bias=-0.395179756156 |

The score is fixture-local and carries no policy-performance claim.
