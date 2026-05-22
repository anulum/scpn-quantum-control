<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- realtime control latency consolidation -->

# Realtime Control Latency Consolidation

Timestamp: `2026-05-22T00:13:57.520930+00:00`

## Claim Boundary

- Local lane: host-only no-QPU control loop latency (Python vs Rust full loop).
- IBM lane: externally visible runtime windows only (submit-to-result); no direct intra-shot hardware feedforward latency claim.

## Local Host Loop (ms/tick)

- Python mean: `3.664295` ms, std: `0.959325`, n=3
- Rust mean: `0.000136` ms, std: `0.000034`, n=3

## IBM Runtime (s submit-to-result)

- Feedback dynamic mean: `14.345` s, std: `15.358`, n=8
- Control open-loop mean: `12.163` s, std: `11.893`, n=8
- Capacity sweep mean: `7.723` s, std: `1.757`, n=12

## IBM Runtime Rust Orchestrator (s submit-to-done)

- Rust submit-to-done mean: `7.338` s, std: `0.343`, n=4

## Sources

- Local summary: `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/realtime_control_e2e_summary_2026-05-22.json`
- IBM campaigns:
  - `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_latency_campaign_ibm_kingston_20260521T231917Z.json`
  - `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_latency_campaign_ibm_kingston_20260521T232106Z.json`
  - `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_latency_campaign_ibm_kingston_20260521T233721Z.json`
- IBM Rust run: `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/s1_feedback_loop/ibm_runtime_rust_latency_run_2026-05-22.json`

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
- `d87pcggp0eas73dmgn8g`
- `d87pcjqs46sc73f8h9h0`
- `d87pcm1789is73917umg`
- `d87pcoas46sc73f8h9n0`
- `d87pd3as46sc73f8ha40`
- `d87pdgqs46sc73f8hajg`
- `d87pdj2s46sc73f8ham0`
- `d87pdm1789is73917vrg`
- `d87pdotg7okc73eml1dg`
- `d87pdqop0eas73dmgomg`
- `d87pdt2s46sc73f8hb1g`
- `d87pdvqs46sc73f8hb40`
- `d87pe20p0eas73dmgoug`
- `d87pe59789is739180dg`
- `d87pe7dg7okc73eml1u0`
- `d87peais46sc73f8hbig`

## Rust Job IDs
- `d87pv0gp0eas73dmhdtg`
- `d87pv2lg7okc73emlmf0`
- `d87pv49789is73918lng`
- `d87pv60p0eas73dmhe50`
