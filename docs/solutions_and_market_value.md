# Solutions, applications, and market value

`scpn-quantum-control` turns coupled-oscillator models into reproducible
simulation, optimisation, hardware-readiness, and evidence workflows. Its
commercial value is governed uncertainty reduction: teams can decide what is
ready for a local pilot, what needs hardware evidence, and what must remain a
research claim before they spend engineering or QPU budget.

This page describes credible value routes. It does not publish a fabricated
total-addressable-market number, promise quantum advantage, or promote a
research proxy into a validated industrial or clinical product.

## The problem it solves

Synchronisation appears in power networks, plasma dynamics, Josephson systems,
biological rhythms, control loops, quantum devices, and distributed sensing.
The underlying mathematics is reusable, but most projects rebuild the same
pipeline:

1. encode a network and its natural frequencies;
2. choose a classical, quantum, or hybrid solver;
3. compare outputs and gradients;
4. decide whether a hardware run is justified;
5. preserve evidence that another team can audit;
6. translate a research workflow into a supported integration.

The software provides those layers under shared `K_nm` and `omega` contracts,
with explicit boundaries between simulation, method evidence, hardware
evidence, and deployable interfaces.

## Value by stakeholder

| Stakeholder | Immediate outcome | Economic or operational value |
|---|---|---|
| Quantum R&D lead | Reproducible Kuramoto-XY experiments with classical references | Reject weak hardware candidates before paying provider and staff costs |
| Control or systems engineer | Typed network, objective, constraint, and replay contracts | Compare candidate methods without silently changing the physical problem |
| Quantum-hardware operator | Capability probes, no-submit readiness gates, result packs, and raw-count custody | Reduce failed submissions and prevent simulator results from becoming hardware claims |
| ML or optimisation engineer | Parameter-shift, framework, finite-difference, and compiler-AD routes with explicit support states | See exact, approximate, stochastic, and blocked gradients before integration |
| Research group | Tutorials, notebooks, benchmark protocols, and publication evidence boundaries | Reproduce and review results without reconstructing undocumented environments |
| Product or compliance team | Stable facades, security controls, licence routes, SBOM/release gates, and claim ledgers | Evaluate integration and distribution risk before committing to a pilot |

## Application portfolio

| Application lane | What can be evaluated now | Current readiness | Required promotion evidence |
|---|---|---|---|
| Quantum synchronisation research | XY compilation, Trotter dynamics, VQE, witnesses, topology, open systems, and exact/classical comparisons | Mature local software and governed research modules | Named benchmark or hardware artefacts for performance and physical claims |
| Quantum-control method development | Bounded topology, DLA, QFI, differentiable objectives, pulse feasibility, and feedback contracts | Simulator-first engineering with explicit blocked branches | Calibrated plant/backend model, preregistration, safety review, and raw execution evidence |
| Power-grid oscillator studies | Synthetic and standard-case graph coupling, stability features, and comparison adapters | Research/pilot candidate; not an operational grid controller | Utility-owned data, unit calibration, uncertainty, null models, and operator validation |
| Plasma and fusion workflows | ITER-inspired synthetic benchmarks, disruption interfaces, and cross-repository bridge contracts | Research/pilot candidate; no facility-control authority | Facility-approved data, safety case, independent validation, and control-system qualification |
| Josephson and superconducting networks | Coupled-junction simulation, magnitude studies, and Kuramoto-XY comparisons | Local scientific workflow | Device calibration, fabrication metadata, uncertainty, and independent hardware replication |
| EEG/MEG and biological rhythms | Source-bounded synchronisation features, synthetic fixtures, and privacy-aware adapters | Research only; non-clinical | Governed human data, ethics approval, external validation, clinical study, and regulatory route |
| Quantum sensing | Readiness contracts and synchronisation-order candidate observables | Preregistered research boundary | Fisher-information baseline, calibrated perturbations, shot budget, and raw-count uncertainty archive |
| Quantum ML and differentiable computing | QNN/QGNN/QSNN examples, framework parity, gradient contracts, and compiler-AD evidence | Software-method evaluation | Held-out tasks, matched baselines, isolated benchmarks, and domain-specific validation |
| Provider and hardware integration | Offline SDK smoke, capability normalization, circuit/result contracts, and no-submit gates | Integration-ready software boundary | Credentials, account authority, approved spend, provider-specific preflight, and execution custody |

## Commercial routes

The repository supports several honest commercial engagements without relying
on speculative advantage claims:

- **Commercial licence:** closed-source products, proprietary network services,
  embedded deployments, and non-AGPL redistribution require a negotiated grant.
- **Integration and hardening:** adapt stable facades, provider boundaries,
  observability, packaging, and evidence gates to an organisation's stack.
- **Governed pilot:** define one falsifiable oscillator-network use case, build
  the classical baseline, run local evidence first, and open hardware spend only
  after the readiness gate passes.
- **Research engineering:** build reproducible benchmark, gradient, simulation,
  or hardware-evidence packages for a named scientific question.
- **Training and enablement:** use the curated tutorials, notebook catalogue,
  API contracts, and release process for team onboarding.

Pricing, support levels, service guarantees, publication rights, background IP,
and hardware spend are contract-specific. The public repository does not imply
a commercial licence or a service-level agreement.

## Why this is differentiated

The strongest differentiation is the composition of capabilities rather than
one isolated algorithm:

- a common physical input contract across classical, quantum, and hybrid paths;
- stable first-path facades plus a deep research and compiler surface;
- Rust acceleration for selected kernels without removing Python fallbacks;
- deterministic evidence generators and negative-space registries;
- explicit provider, hardware, clinical, advantage, and publication boundaries;
- public release, security, SBOM, licence, documentation, and provenance gates.

This combination makes the platform useful before fault-tolerant quantum
advantage exists: it improves method selection, reproducibility, integration,
and evidence quality now.

## Pilot decision framework

Use this sequence for a commercial or research pilot:

1. **Fit:** express the system as `K_nm`, `omega`, observables, constraints, and
   an explicit outcome metric.
2. **Baseline:** reproduce a classical or exact reference locally.
3. **Method:** select a supported simulator, gradient, compiler, or control
   route through the API and support matrices.
4. **Falsification:** state what result would stop the pilot.
5. **Evidence:** fix seeds, dependencies, datasets, and result-pack schema.
6. **Hardware gate:** submit only if the no-submit readiness checks pass and
   authority, credentials, budget, and custody are explicit.
7. **Adoption gate:** integrate through stable facades; treat advanced modules
   as independently versioned research surfaces.

Start with [Onboarding](onboarding.md), run the [Quickstart](quickstart.md),
select a route from [Tutorials](tutorials.md), and inspect the
[complete API catalog](api/module_catalog.md). For a release or procurement
decision, also read [Release Readiness](release_readiness.md),
[Threat Model](THREAT_MODEL.md), and the [Licensing FAQ](licensing_faq.md).

## Evidence boundary

Market value here means potential cost reduction, integration leverage, and
research productivity supported by current software contracts. It does not
mean demonstrated market share, revenue, clinical efficacy, operational plant
performance, universal quantum advantage, or guaranteed return on investment.

For a scoped commercial evaluation, contact `protoscience@anulum.li`.
