# BL-60 Chimera and Multiscale Control Evidence

- Schema: `chimera_multiscale_control_evidence.v1`
- Generated on: `2026-07-29`
- Population size: `64` per population
- Content digest: `a50a32ac6171502e3caebaef63d5b3b34b83f3dddb952100c578700f07e682fd`
- Claim boundary: deterministic finite-N synthetic Kuramoto-Sakaguchi analysis and unapplied control proposals; no thermodynamic-limit, stability, controllability, biological, EEG, clinical, hardware, or deployment claim

## Frozen finite synthetic measurements

| Metric | Chimera transient | Synchronised control |
|---|---:|---:|
| Chimera index | 0.0725616776344 | 0.000260388587214 |
| Global order mean | 0.716193516233 | 0.984868082521 |
| Population 1 mean | 0.999999026245 | 0.999955863897 |
| Population 2 mean | 0.504199062198 | 0.974441609074 |
| Population 2 minimum | 0.062349774716 | 0.899115860998 |
| Population 2 standard deviation | 0.210782084664 | 0.0197839711299 |
| Objective before proposal | 0.0374267934088 | 1.55956300093e-05 |
| Objective after proposal | 0.0374080418468 | 1.55948396973e-05 |

The two rows share the exact finite-N equation, integration step, phase lag, seed, and initial-condition construction; only their frozen coupling regime and run length differ. The measurements are regression evidence for this configuration, not an attractor, generalisation, or physical-domain claim.

## Differentiation and topology custody

- Maximum analytic-versus-central-difference gradient error: `2.49120728928e-11`.
- Topology ledger violation before projection: `182.2484375`.
- Topology ledger violation after projection: `2.84217094304e-14`.
- Topology content digest: `3b599a11eb6a1869ca334963a6f69d1ea53a8657ac88f2ba41c4a815d6d8d5d1`.

## Scope matrix

| Capability | Status | Evidence | Non-claim |
|---|---|---|---|
| S60.1 synthetic chimera generators | supported | exact production Sakaguchi force with deterministic RK4 and two frozen regimes | finite trajectories do not prove a thermodynamic-limit attractor |
| S60.2 differentiable chimera and cluster losses | supported | composed existing analytic cluster-order gradients with finite-difference replay | a local phase objective is not a closed-loop stability certificate |
| S60.3 multiscale order-parameter suite | supported | nested population and ensemble partitions measured through oscillatools | synthetic hierarchy does not validate a biological or EEG hierarchy |
| S60.4 BL-32 F5-F6 catalogue rows | descoped | BL-60 has a direct tested facade; the optional unrelated registry extension has no consumer | absence from BL-32 does not imply missing BL-60 production APIs |
| S60.5 topology-constraint interaction | bounded | existing TopologyConstraintLedger projection with before/after violation custody | projection is not differentiable learning, PH, DLA, hardware, or controllability proof |
| S60.6 notebook and evidence artefact | supported | executable notebook 50 and deterministic JSON/Markdown byte-check runner | tutorial output is configuration-specific research evidence |

## Reproduction

```bash
PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py
PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py --check
```
