# Benchmark Harness Registry

This registry distinguishes implemented public benchmark harnesses from planned entries. Planned rows are visible so the roadmap is transparent, but they are not treated as available benchmark results.

## Counts
- Implemented: `1`
- Planned: `4`
- Blocked: `0`

## Families

### phase1_dla_parity
- Title: Phase 1 DLA-parity leakage reproduction
- Status: `implemented`
- Public API: `scpn_quantum_control.benchmark_harness.run_phase1_benchmark`
- Command: `scpn-bench s5-benchmark-suite`
- Dataset: `data/phase1_dla_parity/*.json`
- Generated artefact: `data/s5_benchmark_harness/phase1_benchmark_harness_2026-05-06.json`
- Baseline: noiseless numpy/qutip parity-conservation reference
- Claim boundary: Reproduces committed raw-count statistics and a noiseless classical reference; does not submit QPU jobs or claim quantum advantage.

### chsh_hardware
- Title: CHSH hardware sanity benchmark
- Status: `planned`
- Public API: `None`
- Command: `None`
- Dataset: `None`
- Generated artefact: `None`
- Baseline: classical CHSH bound and Bell-state simulator reference
- Claim boundary: Not exposed until raw counts, loader, tolerance bundle, and baseline are committed.
- Blocker: promote existing CHSH artefacts into a typed raw-data loader and reproducer

### bkt_phase_transition
- Title: BKT phase-transition diagnostic benchmark
- Status: `planned`
- Public API: `None`
- Command: `None`
- Dataset: `None`
- Generated artefact: `None`
- Baseline: classical XY/Kuramoto finite-size diagnostic reference
- Claim boundary: Not exposed until the dataset schema and finite-size tolerance policy are committed.
- Blocker: define stable BKT raw-data schema, summary statistics, and reproducibility tolerances

### otoc_scrambling
- Title: OTOC scrambling benchmark
- Status: `planned`
- Public API: `None`
- Command: `None`
- Dataset: `None`
- Generated artefact: `None`
- Baseline: exact diagonalisation or tensor-network OTOC reference
- Claim boundary: Not exposed until OTOC raw data and classical reference are committed.
- Blocker: identify canonical OTOC dataset and baseline implementation

### dla_dimension
- Title: Dynamical Lie-algebra dimension benchmark
- Status: `planned`
- Public API: `None`
- Command: `None`
- Dataset: `None`
- Generated artefact: `None`
- Baseline: symbolic or exact commutator-closure reference
- Claim boundary: Not exposed until exact algebra rows and independent reference checks are committed.
- Blocker: package DLA-dimension rows with independent closure validation
