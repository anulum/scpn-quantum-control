<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- realtime control latency consolidation -->

# Realtime Control Latency Consolidation

Timestamp: `2026-05-21T23:26:06.364826+00:00`

## Claim Boundary

- Local lane: host-only no-QPU control loop latency (Python vs Rust full loop).
- IBM lane: externally visible runtime windows only (submit-to-result); no direct intra-shot hardware feedforward latency claim.

## Local Host Loop (ms/tick)

- Python mean: `4.271815` ms, std: `0.668494`, n=3
- Rust mean: `0.000264` ms, std: `0.000053`, n=3

## IBM Runtime (s submit-to-result)

- Feedback dynamic mean: `8.592` s, std: `0.522`, n=4
- Control open-loop mean: `7.794` s, std: `0.926`, n=4
- Capacity sweep mean: `6.936` s, std: `0.506`, n=4

## Sources

- Local summary: `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/realtime_control_e2e_summary_2026-05-22.json`
- IBM campaigns:
  - `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_latency_campaign_ibm_kingston_20260521T231917Z.json`
  - `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_latency_campaign_ibm_kingston_20260521T232106Z.json`

## Job IDs
- `d87p5hgp0eas73dmgdig`
- `d87p5jp789is73917l50`
- `d87p5m9789is73917l7g`
- `d87p5p0p0eas73dmgdqg`
- `d87p5ris46sc73f8h040`
- `d87p679789is73917lqg`
- `d87p69p789is73917lt0`
- `d87p6c5g7okc73emko00`
- `d87p6eqs46sc73f8h0qg`
- `d87p6hh789is73917m6g`
- `d87p6k1789is73917mag`
- `d87p6mdg7okc73emkod0`
