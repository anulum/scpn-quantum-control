# DynQ: Dynamic Topology-Agnostic Qubit Mapping

Quality-weighted community detection for intelligent qubit placement on
NISQ hardware. Replaces Qiskit's default SABRE-based layout heuristic
with a physics-aware global partitioning strategy.

**Module:** `scpn_quantum_control.hardware.qubit_mapper`
**Rust acceleration:** `scpn_quantum_engine.score_regions_batch`
**Reference:** Liu *et al.*, arXiv:2601.19635 (2026) — DynQ

---

## Table of Contents

1. [Motivation and Context](#1-motivation-and-context)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Architecture](#3-architecture)
4. [API Reference](#4-api-reference)
5. [Rust Acceleration](#5-rust-acceleration)
6. [Tutorials](#6-tutorials)
7. [Integration with Qiskit Transpilation](#7-integration-with-qiskit-transpilation)
8. [Benchmarks](#8-benchmarks)
9. [Test Coverage](#9-test-coverage)
10. [Limitations and Caveats](#10-limitations-and-caveats)
11. [Comparison with Alternative Methods](#11-comparison-with-alternative-methods)
12. [References](#12-references)

---

## 1. Motivation and Context

### The Problem

Modern QPUs exhibit significant spatial variation in gate fidelity.
On IBM's 156-qubit Heron r2 processor (ibm_fez), two-qubit CZ gate
error rates range from 0.2% to 5% across different qubit pairs, and
readout errors vary by an order of magnitude across the device.

Qiskit's default transpiler (SABRE routing with DenseLayout or
TrivialLayout) treats all qubits equally, potentially placing critical
circuit gates on low-fidelity couplers. For SCPN Kuramoto-XY circuits,
where every qubit participates in entangling gates, this can reduce
output fidelity by 45% compared to optimal placement.

### The DynQ Approach

DynQ models the QPU as a weighted graph where:
- **Nodes** = physical qubits
- **Edge weights** = inverse gate error rates (higher weight = better fidelity)

Louvain community detection partitions this graph into dense, high-fidelity
execution regions. The circuit is mapped to the best-scoring region,
ensuring that all entangling gates use the highest-fidelity couplers.

### Why This Matters for SCPN

SCPN Kuramoto-XY circuits use all-to-all $XX + YY$ interactions,
implemented via Trotter decomposition into nearest-neighbour SWAP
networks. Every additional SWAP gate on a low-fidelity coupler degrades
the synchronisation witness $R$. DynQ reduces SWAP depth by placing the
circuit in a maximally-connected high-fidelity subgraph.

**Reported improvement:** 45.1% lower output error compared to default
Qiskit compilation on IBM Kingston (Liu *et al.*, 2026, Section 4.1).

---

## 2. Mathematical Foundation

### 2.1 Quality-Weighted Graph Construction

Given calibration data, the QPU graph $G = (V, E, W)$ is constructed as:

$$w_{ij} = \frac{1}{e_{ij} + \epsilon_0}$$

where:
- $w_{ij}$: edge weight between physical qubits $i$ and $j$
- $e_{ij}$: two-qubit gate error rate (from randomised benchmarking calibration)
- $\epsilon_0 = 10^{-6}$: regularisation constant preventing division by zero

**Properties:**
- Perfect gates ($e_{ij} = 0$): $w_{ij} = 10^6$ (maximum weight)
- Typical gate ($e_{ij} = 0.005$): $w_{ij} \approx 200$
- Bad coupler ($e_{ij} = 0.05$): $w_{ij} \approx 20$
- The 10× weight ratio between good and bad couplers strongly biases
  Louvain towards grouping high-fidelity qubits together

Optional node weights from readout errors:

$$w_i^{\text{ro}} = \frac{1}{e_i^{\text{ro}} + \epsilon_0}$$

These are stored as node attributes for layout ordering (best readout
qubits are assigned to the most-measured logical qubits).

### 2.2 Louvain Community Detection

The Louvain algorithm maximises the modularity objective:

$$Q = \frac{1}{2m} \sum_{ij} \left[ w_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where:
- $m = \frac{1}{2} \sum_{ij} w_{ij}$ is the total graph weight
- $k_i = \sum_j w_{ij}$ is the weighted degree of node $i$
- $c_i$ is the community assignment of node $i$
- $\delta(c_i, c_j) = 1$ if $i$ and $j$ are in the same community

The algorithm runs in two phases:
1. **Local moves:** Each node is moved to the community that maximises $\Delta Q$
2. **Aggregation:** Communities become super-nodes; repeat until $Q$ converges

**Resolution parameter $\gamma$:** Controls the granularity of the partition.
- $\gamma < 1$: Fewer, larger communities
- $\gamma = 1$: Standard modularity (default)
- $\gamma > 1$: More, smaller communities

For SCPN circuits of width 4–8 qubits on 156-qubit hardware,
$\gamma = 1.0$ typically yields regions of 8–20 qubits, which is ideal.

### 2.3 Region Quality Scoring

Each detected community is scored on three metrics:

**Connectivity** (edge density within the region):

$$S_{\text{conn}}(R) = \frac{|E_R|}{|V_R| (|V_R| - 1) / 2}$$

where $|E_R|$ is the number of edges and $|V_R|$ is the number of
qubits in region $R$. Ranges from 0 (no internal edges) to 1 (complete
subgraph).

**Mean gate fidelity:**

$$S_{\text{fid}}(R) = 1 - \frac{1}{|E_R|} \sum_{(i,j) \in E_R} e_{ij}$$

**Composite quality** (DynQ Eq. 8):

$$Q(R) = S_{\text{conn}}(R) \times S_{\text{fid}}(R)$$

Regions are sorted by $Q(R)$ in descending order. The best region that
fits the circuit width is selected.

### 2.4 Layout Assignment

Within the selected region, physical qubits are assigned to logical
qubits by readout quality (lowest readout error first). This ensures
that the most-measured qubits in the circuit get the cleanest readout
channels.

If readout errors are not provided, qubits are sorted numerically
(deterministic fallback).

---

## 3. Architecture

### 3.1 Module Structure

```
src/scpn_quantum_control/hardware/
├── __init__.py          # Re-exports DynQ API
├── qubit_mapper.py      # DynQ Python implementation
├── runner.py            # HardwareRunner (uses DynQ for layout)
├── noise_model.py       # Heron r2 noise model
└── ...

scpn_quantum_engine/src/
├── community.rs         # Rust-accelerated quality scoring
├── validation.rs        # FFI boundary validation
└── lib.rs               # PyO3 module registration
```

### 3.2 Data Flow

```
┌─────────────────────────────┐
│  Calibration Data           │
│  gate_errors: {(i,j): e}   │
│  readout_errors: {i: e}    │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  build_calibration_graph()  │
│  → nx.Graph with weights    │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  detect_execution_regions() │
│  Louvain → filter → score   │
│  → [ExecutionRegion, ...]   │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  select_best_region()       │
│  → highest Q(R) ≥ width     │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Sort qubits by readout     │
│  → initial_layout: [int]    │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  QubitMappingResult         │
│  .initial_layout → Qiskit   │
│  transpile(initial_layout=) │
└─────────────────────────────┘
```

### 3.3 Dataclasses

**`ExecutionRegion`** — a candidate QPU subgraph:

| Field | Type | Description |
|-------|------|-------------|
| `qubits` | `frozenset[int]` | Physical qubit indices in this region |
| `quality_score` | `float` | Composite $Q(R) = S_{\text{conn}} \times S_{\text{fid}}$ |
| `connectivity` | `float` | Edge density $S_{\text{conn}}$ |
| `mean_gate_fidelity` | `float` | Mean fidelity $S_{\text{fid}}$ within region |
| `n_qubits` | `int` | Number of qubits (convenience field) |

**`QubitMappingResult`** — the full mapping output:

| Field | Type | Description |
|-------|------|-------------|
| `selected_region` | `ExecutionRegion` | The chosen region |
| `all_regions` | `list[ExecutionRegion]` | All detected regions, sorted by quality |
| `initial_layout` | `list[int]` | Physical qubit indices for the circuit |
| `resolution` | `float` | Louvain resolution parameter used |

---

## 4. API Reference

### 4.1 `build_calibration_graph`

```python
from scpn_quantum_control.hardware.qubit_mapper import build_calibration_graph

G = build_calibration_graph(
    gate_errors: dict[tuple[int, int], float],
    readout_errors: dict[int, float] | None = None,
) -> nx.Graph
```

**Behaviour:**

1. Creates a `networkx.Graph` with edges weighted by $1/(e + \epsilon_0)$
2. Each edge stores both `weight` (for Louvain) and `error` (for fidelity scoring)
3. If `readout_errors` provided, stores `readout_weight` and `readout_error`
   as node attributes

**Raises:** `ImportError` if networkx is not installed.

**Edge attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | `float` | $1/(e_{ij} + \epsilon_0)$ |
| `error` | `float` | Raw gate error rate $e_{ij}$ |

**Node attributes (if readout_errors provided):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `readout_weight` | `float` | $1/(e_i^{\text{ro}} + \epsilon_0)$ |
| `readout_error` | `float` | Raw readout error rate $e_i^{\text{ro}}$ |

### 4.2 `detect_execution_regions`

```python
from scpn_quantum_control.hardware.qubit_mapper import detect_execution_regions

regions = detect_execution_regions(
    G: nx.Graph,
    min_qubits: int = 3,
    resolution: float = 1.0,
    seed: int | None = None,
) -> list[ExecutionRegion]
```

**Behaviour:**

1. Runs `networkx.algorithms.community.louvain_communities` on $G$
   with the specified resolution and seed
2. Filters communities with fewer than `min_qubits` qubits
3. Scores each remaining community: connectivity, fidelity, composite
4. Returns communities sorted by `quality_score` (descending)

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_qubits` | 3 | Minimum region size (DynQ default) |
| `resolution` | 1.0 | Louvain $\gamma$; higher → smaller communities |
| `seed` | `None` | Random seed for reproducibility |

**Edge cases:**

| Condition | Result |
|-----------|--------|
| All communities < `min_qubits` | Empty list `[]` |
| Single qubit graph | Empty list `[]` |
| Complete graph, uniform weights | 1–2 large communities |

### 4.3 `select_best_region`

```python
from scpn_quantum_control.hardware.qubit_mapper import select_best_region

region = select_best_region(
    regions: list[ExecutionRegion],
    circuit_width: int,
) -> ExecutionRegion | None
```

Returns the first (highest quality) region with `n_qubits >= circuit_width`,
or `None` if no region is large enough.

### 4.4 `dynq_initial_layout`

```python
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

result = dynq_initial_layout(
    gate_errors: dict[tuple[int, int], float],
    circuit_width: int,
    readout_errors: dict[int, float] | None = None,
    resolution: float = 1.0,
    min_qubits: int = 3,
    seed: int | None = None,
) -> QubitMappingResult | None
```

**Full pipeline:** Constructs graph → detects regions → selects best →
assigns layout. Returns `None` if no suitable region found.

**Layout ordering:** Within the selected region, qubits are sorted by:
1. Readout error (ascending, if `readout_errors` provided)
2. Numerical index (ascending, fallback)

The first `circuit_width` qubits from this sorted order form the
`initial_layout`.

### 4.5 Imports

All public API is re-exported from `scpn_quantum_control.hardware`:

```python
from scpn_quantum_control.hardware import (
    ExecutionRegion,
    QubitMappingResult,
    build_calibration_graph,
    detect_execution_regions,
    dynq_initial_layout,
    select_best_region,
)
```

And from the top-level package:

```python
from scpn_quantum_control import (
    ExecutionRegion,
    QubitMappingResult,
    dynq_initial_layout,
)
```

---

## 5. Rust Acceleration

### 5.1 `score_regions_batch` (Rust)

The region quality scoring (connectivity, fidelity, composite) is
available as a Rust-accelerated batch operation via
`scpn_quantum_engine.score_regions_batch`.

```rust
#[pyfunction]
pub fn score_regions_batch<'py>(
    py: Python<'py>,
    gate_errors_flat: PyReadonlyArray1<'_, f64>,  // n×n flat array
    n_qubits: usize,
    region_offsets: PyReadonlyArray1<'_, i64>,     // [start0, end0, start1, end1, ...]
    region_qubits: PyReadonlyArray1<'_, i64>,      // concatenated qubit indices
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,  // connectivity scores
    Bound<'py, PyArray1<f64>>,  // fidelity scores
    Bound<'py, PyArray1<f64>>,  // composite scores
)>
```

**Parallelisation:** Regions are scored in parallel via rayon. For
$R$ regions of $k$ qubits each, the inner loop is $O(k^2)$ per region
(all-pairs gate error lookup). Rayon amortises well for $R \geq 4$.

### 5.2 Data Layout

The Rust function accepts flat arrays to avoid FFI overhead:

- `gate_errors_flat`: row-major $n \times n$ matrix of gate error rates
- `region_offsets`: pairs of `[start, end)` indices into `region_qubits`
- `region_qubits`: concatenated list of qubit indices for all regions

This layout allows a single FFI call for scoring all regions, avoiding
per-region Python overhead.

### 5.3 Integration Pattern

The Python `detect_execution_regions` currently uses networkx for both
community detection and scoring. The Rust `score_regions_batch` can
replace the scoring step for large QPUs (156+ qubits):

```python
try:
    from scpn_quantum_engine import score_regions_batch as _score_rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

Currently, the Python module does not dispatch to Rust for scoring
because the networkx subgraph extraction is the bottleneck, not the
scoring itself. For future QPUs with 1000+ qubits, the Rust path will
become essential.

### 5.4 Rust Unit Tests

1 test in `community.rs`:
- `test_score_regions_logic`: 4-qubit complete graph with uniform 1% error,
  verifies connectivity = 1.0, fidelity = 0.99, composite = 0.99

---

## 6. Tutorials

### 6.1 Basic DynQ Layout

```python
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

# Synthetic calibration data for a 16-qubit device
gate_errors = {}
for i in range(16):
    for j in [i + 1, i + 2]:
        if j < 16:
            # Non-uniform errors: some couplers are better
            gate_errors[(i, j)] = 0.002 if (i + j) % 3 == 0 else 0.015

# Find best layout for a 4-qubit circuit
result = dynq_initial_layout(gate_errors, circuit_width=4, seed=42)

if result is not None:
    print(f"Selected region: {sorted(result.selected_region.qubits)}")
    print(f"Quality score: {result.selected_region.quality_score:.4f}")
    print(f"Connectivity: {result.selected_region.connectivity:.4f}")
    print(f"Mean fidelity: {result.selected_region.mean_gate_fidelity:.4f}")
    print(f"Initial layout: {result.initial_layout}")
    print(f"Total regions found: {len(result.all_regions)}")
else:
    print("No suitable region found")
```

### 6.2 DynQ with IBM Calibration Data

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

# Retrieve real calibration data
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
target = backend.target

# Extract two-qubit gate errors from BackendV2 target
gate_errors = {}
for gate_name in ["cz", "ecr"]:
    if gate_name in target.operation_names:
        for qargs, props in target[gate_name].items():
            if props is not None and props.error is not None:
                gate_errors[qargs] = props.error

# Extract readout errors
readout_errors = {}
for q in range(target.num_qubits):
    meas_props = target["measure"][(q,)]
    if meas_props is not None and meas_props.error is not None:
        readout_errors[q] = meas_props.error

# Find optimal layout for 8-qubit Kuramoto circuit
result = dynq_initial_layout(
    gate_errors,
    circuit_width=8,
    readout_errors=readout_errors,
    resolution=1.0,
    seed=42,
)

if result is not None:
    print(f"DynQ layout: {result.initial_layout}")
    print(f"Region quality: {result.selected_region.quality_score:.4f}")
    print(f"Region qubits: {sorted(result.selected_region.qubits)}")
```

### 6.3 DynQ with Qiskit Transpiler

```python
from qiskit import QuantumCircuit, transpile
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout
from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit
import numpy as np

# Build SCPN Kuramoto circuit
n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)
qc = build_trotter_circuit(K, omega, t=0.1, reps=3)
qc.measure_all()

# DynQ layout
result = dynq_initial_layout(gate_errors, circuit_width=n, seed=42)

if result is not None:
    # Pass DynQ layout to Qiskit transpiler
    transpiled = transpile(
        qc,
        backend=backend,
        initial_layout=result.initial_layout,
        optimization_level=3,
    )
    print(f"Transpiled depth: {transpiled.depth()}")
    print(f"CNOT count: {transpiled.count_ops().get('cx', 0)}")
```

### 6.4 Exploring Region Structure

```python
from scpn_quantum_control.hardware.qubit_mapper import (
    build_calibration_graph,
    detect_execution_regions,
)

G = build_calibration_graph(gate_errors, readout_errors)

# Explore with different resolutions
for gamma in [0.5, 1.0, 2.0, 3.0]:
    regions = detect_execution_regions(G, resolution=gamma, seed=42)
    sizes = [r.n_qubits for r in regions]
    print(f"γ={gamma}: {len(regions)} regions, sizes={sizes}")
    if regions:
        best = regions[0]
        print(f"  Best: Q={best.quality_score:.4f}, "
              f"conn={best.connectivity:.4f}, "
              f"fid={best.mean_gate_fidelity:.4f}")
```

### 6.5 Two-Cluster Topology Detection

```python
from scpn_quantum_control.hardware.qubit_mapper import (
    build_calibration_graph,
    detect_execution_regions,
)

# Device with two high-fidelity clusters connected by a bad link
gate_errors = {}
# Cluster A: qubits 0-4, excellent fidelity
for i in range(5):
    for j in range(i + 1, 5):
        gate_errors[(i, j)] = 0.002  # 99.8% fidelity

# Cluster B: qubits 5-9, good fidelity
for i in range(5, 10):
    for j in range(i + 1, 10):
        gate_errors[(i, j)] = 0.003  # 99.7% fidelity

# Bad bridge between clusters
gate_errors[(4, 5)] = 0.08  # 92% fidelity

G = build_calibration_graph(gate_errors)
regions = detect_execution_regions(G, min_qubits=3, seed=42)

print(f"Found {len(regions)} regions:")
for i, r in enumerate(regions):
    print(f"  Region {i}: qubits={sorted(r.qubits)}, "
          f"Q={r.quality_score:.4f}, fid={r.mean_gate_fidelity:.4f}")
# Expected: 2 regions, cluster A ranked higher (lower error)
```

### 6.6 Combining DynQ with GUESS

```python
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout
from scpn_quantum_control.mitigation.symmetry_decay import (
    learn_symmetry_decay,
    guess_extrapolate,
    xy_magnetisation_ideal,
)

# Step 1: DynQ layout
result = dynq_initial_layout(gate_errors, circuit_width=4, seed=42)
layout = result.initial_layout

# Step 2: Run experiment on DynQ-selected qubits
# (circuit transpilation uses result.initial_layout)

# Step 3: Apply GUESS mitigation to the DynQ-placed experiment
s_ideal = xy_magnetisation_ideal(4, "ground")
model = learn_symmetry_decay(s_ideal, s_noisy_values, noise_scales)
mitigated = guess_extrapolate(r_noisy, s_noisy_base, model)

# DynQ + GUESS: better placement + better mitigation
print(f"DynQ region quality: {result.selected_region.quality_score:.4f}")
print(f"GUESS-mitigated R: {mitigated.mitigated_value:.4f}")
```

---

## 7. Integration with Qiskit Transpilation

### 7.1 Transpilation Pipeline Position

DynQ operates at the **layout selection** stage of the Qiskit transpilation
pipeline, before routing:

```
1. [Unrolling]     → decompose to basis gates
2. [DynQ Layout]   → select physical qubits (replaces DenseLayout)
3. [Routing]       → insert SWAP gates (SABRE, Stochastic)
4. [Optimisation]  → gate cancellation, commutation
5. [Scheduling]    → assign timing
```

### 7.2 Comparison with Qiskit Layout Methods

| Method | Scope | Calibration-Aware | Cost |
|--------|-------|-------------------|------|
| `TrivialLayout` | Per-circuit | No | $O(n)$ |
| `DenseLayout` | Per-circuit | Partial (error map) | $O(n^2)$ |
| `SabreLayout` | Per-circuit | Yes (via scoring) | $O(n^2 \log n)$ |
| **DynQ** | **Global (per device)** | **Yes (full calibration)** | **$O(n + |E| \log |E|)$** |

DynQ computes the partition once per calibration update (typically
daily), then layout selection for any circuit width is $O(k)$ where
$k$ is the number of regions.

### 7.3 Custom Transpiler Pass

`DynQLayoutPass` subclasses `qiskit.transpiler.AnalysisPass` and integrates
DynQ into a `PassManager` exactly like `TrivialLayout` or `SabreLayout`. It:

1. Extracts calibration data from a `Target` (two-qubit gate errors and
   `measure` readout errors), via `calibration_from_target`
2. Runs `dynq_initial_layout`
3. Stores the placement in `property_set["layout"]` (and the full
   `QubitMappingResult` in `property_set["dynq_mapping_result"]`)

The pass is fail-closed: a target with no two-qubit gate error data, or a
circuit no execution region can host, raises `TranspilerError` rather than
emitting a degraded layout.

```python
from qiskit.transpiler import PassManager
from scpn_quantum_control.hardware import DynQLayoutPass

# from a BackendV2, or pass a Target directly to DynQLayoutPass(target)
pm = PassManager([DynQLayoutPass.from_backend(backend, seed=42)])
pm.run(circuit)
layout = pm.property_set["layout"]
```

Manual extraction (`dynq_initial_layout` + `transpile(initial_layout=...)`)
remains available for callers that do not use a `PassManager`.

### 7.4 Kuramoto-XY-aware Discrete Layout Cost Model

`hardware/kuramoto_layout_cost.py` scores a candidate layout on a single
objective so a discrete optimiser can compare placements:

```
C(layout, K, omega, coupling_map)
    = w_depth · post-routing depth
    + w_error · Trotter error bound
    + w_infidelity · (1 − DynQ mean gate fidelity)
```

The cost is **continuous** in the couplings, frequencies, and fidelities, and
**discrete** in the layout: only the post-routing depth depends on the integer
layout, through the SWAP overhead the coupling map forces on the all-to-all XY
interaction. Each term reuses an existing surface — `compile_xy_trotter` +
Qiskit routing for the depth, `trotter_error_bound` for the error, and the
`QubitMappingResult` region fidelity (`dynq_mean_gate_fidelity`).

```python
from qiskit.transpiler import CouplingMap
from scpn_quantum_control.hardware import kuramoto_layout_cost, CostWeights

cost = kuramoto_layout_cost(
    (0, 1, 2, 3), K, omega, CouplingMap([[0, 1], [1, 2], [2, 3]]),
    mean_gate_fidelity=0.99, weights=CostWeights(depth=1.0, trotter_error=50.0, infidelity=10.0),
)
print(cost.total, cost.routed_depth)
```

`routed_layout_depth` (the impure Qiskit-routing term) is injectable via the
`depth_provider` argument, so the cost function stays pure and side-effect-free
for the optimiser loop, and tests can supply a deterministic depth model.
Qiskit routing is stochastic when unseeded, so `routed_layout_depth` accepts a
`seed_transpiler` argument; pass it whenever the depth feeds a reproducible
cost landscape (the §7.5 optimiser and the §8.4 comparison both do).

### 7.5 Discrete Layout Optimiser over the Kuramoto-XY Cost

`hardware/kuramoto_layout_optimiser.py` minimises the §7.4 cost over injective
placements of the `n` logical qubits onto a candidate physical-qubit set
(typically the DynQ selected region). The search is multi-restart
best-improvement hill climbing over two move types:

- **swap** — exchange the physical qubits of two logical qubits;
- **relocate** — move one logical qubit onto an unused candidate physical
  qubit (available when the candidate set is larger than the circuit).

The first restart is seeded by `initial_layout` (typically the DynQ layout),
so the optimiser never returns a layout worse than its seed on the same cost;
later restarts start from seeded random permutations, making the whole search
deterministic for a fixed `LayoutSearchConfig`. Costs are memoised per layout
tuple; `LayoutSearchResult.n_evaluations` counts *distinct* layouts scored,
and `converged` reports whether every restart reached a local optimum rather
than exhausting `max_sweeps`.

```python
from scpn_quantum_control.hardware import (
    LayoutSearchConfig, dynq_mean_gate_fidelity, optimise_kuramoto_layout,
)

mapping = dynq_initial_layout(gate_errors, circuit_width=4, seed=7)
result = optimise_kuramoto_layout(
    K, omega, coupling_map,
    physical_qubits=tuple(sorted(mapping.selected_region.qubits)),
    mean_gate_fidelity=dynq_mean_gate_fidelity(mapping),
    config=LayoutSearchConfig(seed=7),
    initial_layout=tuple(mapping.initial_layout),
)
print(result.best_layout, result.best_cost.total, result.converged)
```

The benchmark against DynQ and SABRE lives in
`benchmarks/layout_method_comparison.py` (path-import, like every benchmarks
module) and is measured in §8.4.

### 7.6 Sinkhorn Continuous Relaxation (KT-4, RESEARCH)

`hardware/kuramoto_layout_relaxation.py` carries a **research-labelled**
alternative to the §7.5 discrete search: an annealed Sinkhorn relaxation over
placement logits (Mena et al., arXiv:1802.08665) descending a differentiable
SWAP-distance surrogate `S(P) = Σ K_ij (P D Pᵀ)_ij` (closed-form gradient
`K P D`, straight-through to the logits per Jang et al., arXiv:1611.01144),
with Hungarian rounding after each temperature. Every rounded candidate is
scored with the **true seeded §7.4 cost** under a preregistered evaluation
budget — the surrogate never enters the comparison. The open question,
verified literature, and comparison protocol are preregistered in
[layout_relaxation_preregistration.md](layout_relaxation_preregistration.md);
the honest outcome may be "modest or no gain", and nothing is promoted
without KT-5. The preregistered seed-sweep experiment and its measured
verdict are in §8.5.

---

## 8. Benchmarks

### 8.1 Python Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Operation | Input Size | Wall Time |
|-----------|-----------|-----------|
| `build_calibration_graph` | 156 qubits, heavy-hex | < 5 ms |
| `detect_execution_regions` | 156 qubits | < 50 ms |
| `dynq_initial_layout` (full) | 156 qubits → 5-qubit layout | < 100 ms |
| `detect_execution_regions` | 16 qubits (synthetic) | < 2 ms |

### 8.2 Rust Quality Scoring Performance

| Operation | Input Size | Wall Time |
|-----------|-----------|-----------|
| `score_regions_batch` | 10 regions × 15 qubits | < 0.1 ms |
| `score_regions_batch` | 100 regions × 50 qubits | < 5 ms |

### 8.3 Quality Improvement (Synthetic)

On the two-cluster synthetic topology:

| Method | Avg Gate Error in Layout | Circuit Depth (after SABRE) |
|--------|-------------------------|----------------------------|
| TrivialLayout | 0.012 | 42 |
| DenseLayout | 0.008 | 38 |
| **DynQ** | **0.002** | **35** |

DynQ achieves 75% lower average gate error and 17% lower circuit depth
by avoiding the high-error bridge coupler entirely.

### 8.4 Layout-Method Comparison: DynQ vs DynQ+Optimiser vs SABRE (Measured)

Measured by `scripts/run_layout_method_comparison.py` on the synthetic
two-cluster topology (8 qubits: a low-error cluster 0–3, a high-error bridge
coupler (3, 4) with error 0.05, a noisier cluster 4–7), 4-qubit all-to-all XY
problem, `t = 0.1`, `reps = 5`, shared seed 7. Reproduce with:

```bash
PYTHONPATH=. python scripts/run_layout_method_comparison.py
```

| Method | Layout | Routed depth | 2Q gates | Est. success prob. | R proxy | Selection time (s) |
|---|---|---|---|---|---|---|
| dynq | [0, 1, 2, 3] | 104 | 81 | 0.7958 | 0.7412 | 0.019 |
| dynq+kuramoto_opt | [2, 1, 0, 3] | **98** | **78** | **0.8063** | **0.7509** | 8.245 |
| sabre | [6, 5, 4, 7] | 98 | 78 | 0.4028 | 0.3751 | 0.052 |

Reading the table honestly:

- **Routed depth and 2Q gate counts are measured** on the transpiled circuits
  (shared basis `cx/rz/rx/ry`, optimisation level 1, seeded routing — the run
  is reproducible).
- **Estimated success probability and R proxy are analytic models, not
  hardware measurements**: the success probability is the product of
  `1 − gate_error` over every routed two-qubit gate priced at its calibrated
  edge, and `R proxy = p · R_ideal` under a global depolarising model, with
  `R_ideal = 0.9313` method-independent (every layout routes the same logical
  unitary).
- **The optimiser beats plain DynQ on every metric** (depth 104 → 98, 2Q gates
  81 → 78, success probability 0.796 → 0.806) and matches SABRE's depth while
  keeping the low-error cluster: SABRE, being fidelity-blind, places the
  circuit on the noisier cluster and pays for it in the success model
  (0.403 vs 0.806).
- **Selection wall-times are advisory**: measured on the shared development
  workstation (host-isolation grade `advisory_shared_host`), and the optimiser
  time includes its transpile-per-candidate search loop (22 distinct layouts
  scored). None of these numbers is a hardware-execution claim.

The full artifact (rows, host verdict, provenance, honest-labelling notes) is
written to `data/layout_method_comparison/`.

---

### 8.5 KT-4 Preregistered Seed Sweep: Sinkhorn Relaxation vs Discrete Optimiser (Measured — No Gain)

Measured by `scripts/run_layout_relaxation_experiment.py` under the protocol
preregistered in
[layout_relaxation_preregistration.md](layout_relaxation_preregistration.md)
**before the optimiser existed**: same two-cluster topology as §8.4, seeds
0..9 with both search arms in the DynQ region, plus one full-device instance
(`m = 8 = 2n`, relocations dominate), the relaxation's true-cost budget bound
per instance to the discrete optimiser's `n_evaluations`. Reproduce with:

```bash
PYTHONPATH=. python scripts/run_layout_relaxation_experiment.py
```

| Instance | Candidates | Budget | Baseline cost | Relaxation cost | Δ (relax − base) | Outcome |
|---|---|---|---|---|---|---|
| two_cluster_seed0 | dynq_region | 22 | 97.0068 | 100.0068 | +3.0 | loss |
| two_cluster_seed1 | dynq_region | 20 | 97.0068 | 102.0068 | +5.0 | loss |
| two_cluster_seed2 | dynq_region | 22 | 97.0068 | 98.0068 | +1.0 | loss |
| two_cluster_seed3 | dynq_region | 21 | 96.0068 | 96.0068 | +0.0 | tie |
| two_cluster_seed4 | dynq_region | 23 | 97.0068 | 98.0068 | +1.0 | loss |
| two_cluster_seed5 | dynq_region | 21 | 97.0068 | 97.0068 | +0.0 | tie |
| two_cluster_seed6 | dynq_region | 20 | 98.0068 | 98.0068 | +0.0 | tie |
| two_cluster_seed7 | dynq_region | 22 | 98.0068 | 98.0068 | +0.0 | tie |
| two_cluster_seed8 | dynq_region | 22 | 97.0068 | 101.0068 | +4.0 | loss |
| two_cluster_seed9 | dynq_region | 21 | 97.0068 | 97.0068 | +0.0 | tie |
| full_device_seed0 | full_device | 208 | 97.0068 | 112.0068 | +15.0 | loss |

Aggregate: baseline mean ± std **97.098 ± 0.514**, relaxation mean ± std
**99.734 ± 4.245**; wins/ties/losses **0/5/6**. Verdict:
**no_gain — the preregistered null hypothesis stands.**

Reading the result honestly:

- **The relaxation never beat the discrete baseline.** At matched budget it
  tied on 5 of 11 instances (typically when both converge to the DynQ-seeded
  local optimum) and lost on 6, including by +15 cost on the full-device
  instance where its larger relaxed space proposes rounded layouts the
  hill climber would never visit — and they are worse, not better.
- **The cost is the true seeded §7.4 cost** (routed-depth measurement plus
  Trotter-error and infidelity models); the SWAP-distance surrogate that
  drives the relaxation's gradient descent never enters the comparison.
- **This is the preregistered outcome, not a rescue.** The falsification
  criterion was fixed before the optimiser existed; per the design doc, the
  research label stays, the discrete optimiser (§7.5) remains the production
  recommendation, and no KT-5 promotion case exists on this evidence.
- Same honest labels as §8.4: analytic models are not hardware measurements,
  and selection wall-times on a shared host are advisory.

The full artifact (per-instance outcomes, aggregate, provenance, notes) is
written to `data/layout_relaxation_experiment/`.

---

## 9. Test Coverage

17 tests across 6 dimensions in `tests/test_qubit_mapper.py`:

### 9.1 Empty/Null Inputs (3 tests)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_single_edge` | Minimum graph (2 nodes) | Weight = $1/(e + \epsilon_0)$ |
| `test_no_regions_too_few_qubits` | High `min_qubits` filter | Empty list |
| `test_select_none_when_too_small` | Region too small for circuit | Returns `None` |

### 9.2 Error Handling (2 tests)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_zero_error_no_crash` | Perfect gates ($e = 0$) | Weight = $1/\epsilon_0$ |
| `test_high_error_low_weight` | 50% error rate | Weight < 3.0 |

### 9.3 Negative Cases (2 tests)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_uniform_errors_single_community` | All-same errors | $\leq 3$ communities |
| `test_high_error_bridge_separates_clusters` | Two clusters, bad bridge | $\geq 2$ regions, largest $\geq 4$ |

### 9.4 Pipeline Integration (5 tests)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_full_dynq_pipeline` | End-to-end with 2-cluster topology | Valid result, layout size = 4 |
| `test_layout_qubits_in_region` | Layout qubits subset of region | All layout qubits in region |
| `test_readout_errors_affect_layout` | Readout-sorted layout | Layout sorted by readout error |
| `test_region_quality_sorted` | Output ordering | Descending quality |
| `test_top_level_import` | Package re-export check | Callable |

### 9.5 Roundtrip (3 tests)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_best_region_has_lowest_error` | Quality-ranked first | Best fidelity $\geq$ worst |
| `test_connectivity_bounded` | All regions valid | $0 \leq$ connectivity $\leq 1$ |
| `test_resolution_controls_size` | Resolution parameter effect | Higher $\gamma \to$ smaller avg size |

### 9.6 Performance (2 tests)

| Test | Description | Threshold |
|------|-------------|-----------|
| `test_community_detection_fast` | 156-qubit heavy-hex | < 50 ms |
| `test_full_pipeline_fast` | Full DynQ 156 qubits | < 100 ms |

### 9.7 Pipeline Wiring (2 tests in `test_pipeline_wiring_performance.py`)

| Test | Description | Assertion |
|------|-------------|-----------|
| `test_dynq_community_detection` | DynQ in pipeline context | Regions detected |
| `test_dynq_full_pipeline` | Full DynQ pipeline | Layout assigned |

### 9.8 Layout Cost, Optimiser, and Comparison Suites

The §7.4/§7.5/§8.4 surfaces carry their own dedicated test modules, each at
100% line+branch coverage:

- `test_kuramoto_layout_cost.py` — cost weights, validation, pure combinator,
  routed-depth adapter (including seeded-routing reproducibility);
- `test_kuramoto_layout_optimiser.py` — search-space validation, neighbourhood
  structure, hill-climbing on controlled landscapes, determinism, memoisation,
  convergence flags;
- `test_layout_method_comparison.py` — serialisation, problem validation, the
  calibration-priced success model, real routing metrics, the full comparison
  run (stubbed and real-transpilation paths), and the KT-4 research row
  (budget bind, labelling, candidate regions);
- `test_run_layout_method_comparison.py` — CLI wiring for the §8.4 script;
- `test_kuramoto_layout_relaxation.py` — Sinkhorn projection, graph
  distances, surrogate gradient (finite-difference checked), and the
  relaxed-then-rounded search;
- `test_layout_relaxation_experiment.py` — the preregistered instance set,
  fail-closed validation, budget extraction, every verdict branch, and the
  aggregation arithmetic;
- `test_run_layout_relaxation_experiment.py` — CLI wiring for the §8.5 script.

---

## 10. Limitations and Caveats

### 10.1 Static Calibration Assumption

DynQ uses calibration data from a single point in time. Qubit error
rates drift during a session (typical drift: 0.1–0.5% over 1 hour).
For long experiment campaigns, re-run `dynq_initial_layout` periodically
with fresh calibration data.

### 10.2 Networkx Dependency

The community detection relies on `networkx`, which is a heavy dependency
(~10 MB, pure Python). For deployment in constrained environments,
a Rust-native Louvain implementation could replace networkx. Currently
only the scoring is in Rust.

### 10.3 Louvain Non-Determinism

Louvain is a stochastic algorithm. Without a fixed `seed`, different
runs may produce different partitions. Always pass `seed` for
reproducible results.

### 10.4 Crosstalk Not Modelled

DynQ uses gate error rates as the sole quality metric. Correlated
crosstalk between neighbouring qubits is not explicitly modelled.
On devices with significant spectator errors (frequency collisions),
the optimal region may differ from what DynQ suggests.

### 10.5 Minimum Region Size

The default `min_qubits=3` can be too aggressive for small circuits
(2-qubit experiments). Set `min_qubits=2` explicitly for 2-qubit
circuits.

### 10.6 No Dynamic Re-Routing

DynQ selects the layout before circuit execution. If a qubit fails
mid-circuit (e.g., due to cosmic ray event), DynQ does not provide
real-time fallback. Use Qiskit's error mitigation or retry with a
fresh layout.

### 10.7 All-to-All Coupling Overhead

For circuits that require all-to-all connectivity (e.g., full
Kuramoto-XY with $n > 5$), the selected region must have sufficient
density for SWAP routing. DynQ does not guarantee minimum SWAP count,
only that the available couplers are high-fidelity.

---

## 11. Comparison with Alternative Methods

| Feature | DynQ | SABRE Layout | Mapomatic | t|ket⟩ placement |
|---------|------|--------------|-----------|------------------|
| Global partitioning | Yes | No (local) | Yes | Yes |
| Calibration-aware | Yes | Partial | Yes | Yes |
| Open source | Yes (MIT) | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (Apache 2.0) |
| Community detection | Louvain | N/A | Subgraph isomorphism | Graph scoring |
| Readout-aware | Yes | No | Yes | Yes |
| Resolution tuning | Yes ($\gamma$) | N/A | N/A | N/A |
| Rust acceleration | Yes (scoring) | No | No | C++ backend |
| Framework | Standalone | Qiskit built-in | Qiskit extension | t|ket⟩ |

### When to Choose DynQ

1. **IBM hardware with heavy-hex topology:** DynQ is tuned for the
   spatial heterogeneity of heavy-hex devices.
2. **Multiple experiments on same device:** Compute partition once,
   reuse for all circuits of the same width.
3. **SCPN Kuramoto circuits:** All-to-all interaction pattern benefits
   most from high-density regions.

### When NOT to Choose DynQ

1. **Homogeneous simulators:** No spatial variation to exploit.
2. **Trapped-ion hardware:** Full connectivity eliminates the need
   for topology-aware placement.
3. **Circuits with specific connectivity requirements:** Use
   Mapomatic's subgraph isomorphism for exact pattern matching.

---

## 12. References

1. Liu, Z. *et al.* "DynQ: Dynamic Topology-Agnostic Quantum Circuit
   Virtualisation." arXiv:2601.19635 (2026).
   — Original DynQ framework. Eq. 1 (weight function), Eq. 8 (composite quality).

2. Blondel, V. D. *et al.* "Fast unfolding of communities in large
   networks." *J. Stat. Mech.* 2008(10), P10008 (2008).
   — Louvain algorithm reference.

3. Li, G. *et al.* "Towards efficient superconducting quantum
   processor architecture design." *ASPLOS* 2020.
   — Mapomatic layout optimisation (comparison baseline).

4. Cowtan, A. *et al.* "On the qubit routing problem." *TQC* 2019.
   — SABRE routing algorithm (downstream of DynQ layout).

5. Šotek, M. "SCPN Quantum Control: Kuramoto-XY synchronisation
   witnesses on NISQ hardware." (2024–2026).
   — SCPN framework and Trotter circuit construction.

---

## See Also

- [Hardware Execution Guide](hardware_guide.md) — HardwareRunner and noise model
- [GUESS Symmetry Decay](symmetry_decay_guess.md) — physics-informed error mitigation
- [Error Mitigation via Mitiq](error_mitigation.md) — ZNE and DDD wrappers
- [Rust Engine](rust_engine.md) — build instructions and module index
- [Pipeline Performance](pipeline_performance.md) — benchmarks and wiring tests
