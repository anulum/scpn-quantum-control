# SCPN Quantum-Control Publication Stack Overview

This overview is the cross-paper map for the current submission stack. It
exists to keep repeated caveats short inside individual papers and to make the
positive contribution of the falsification programme explicit.

## Evidence Boundary

The current live hardware evidence is IBM-limited because authenticated QPU
access for these campaigns was available through IBM Quantum at execution
time. The software stack has provider-neutral execution and artefact-packaging
surfaces for other gate-model, trapped-ion, neutral-atom, photonic, annealing,
analogue, and simulator targets, but those routes are readiness infrastructure
until live provider data exist.

The publication claim is therefore not cross-provider universality. The claim
is that a reproducible Kuramoto--XY hardware programme can define, test, and
demote mechanisms with raw-count artefacts, preregistered reducers, SHA256
digests, and explicit claim boundaries.

## Programme Map

| Paper | Role | Positive knowledge gained |
|---|---|---|
| Software preview | Entry point for reproducible workflow use. | The framework packages installable Kuramoto--XY workflows with raw-count lineage. |
| Rust/VQE methods | Software and hardware-methods validation. | Rust kernels and artefact regeneration are useful workflow components; ansatz performance is hardware-sensitive at larger width. |
| Main overview | Umbrella map of exact diagnostics, legacy rows, and resource boundaries. | Exact-simulation and finite-size diagnostics define what the hardware papers should not overclaim. |
| Phase 1 DLA parity | Primary positive hardware observation. | The Kingston leakage asymmetry is real but demoted to a parity/excitation/state/layout/backend correlated response. |
| FIM Hamiltonian | Theory plus hardware falsification. | Exact sector shifts do not translate into direct digital coherence protection on IBM Heron hardware. |
| Phase 3 reduced Pauli | Mechanism-separation paper. | Dominant deviations localise to transverse DLA channels and survive readout/ZNE stress tests, with larger-width readout boundaries. |
| S1 feedback control | Dynamic-circuit control boundary. | Paired feedback/control arms reveal backend-dependent response and reject backend-general promotion of the tested controller. |

## Reviewer-Facing Framing

The programme should be described as a successful boundary-setting campaign,
not as a sequence of failed advantage claims. The central result is the
method: each proposed mechanism is forced through a reproducible raw-count
pipeline and either promoted only inside its measured boundary or explicitly
falsified.

Individual papers should avoid carrying the full IBM-only caveat at excessive
length. They should state the limitation once, then point to this overview or
the repository artefact catalogue for the broader provider-access boundary.
