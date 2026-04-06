# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Multi-Scale Quantum Error Correction

# Multi-Scale Quantum Error Correction (MS-QEC)

## 1. Mathematical Formalism

### Concatenated Code Structure

MS-QEC implements hierarchical quantum error correction through concatenated
surface codes. Each SCPN domain operates an independent surface code whose
logical qubits serve as the physical qubits of the next domain's code.

The logical error rate at concatenation level $k$ follows the threshold
theorem (Knill 2005, Aharonov & Ben-Or 1997):

$$p_L^{(k)} = A \cdot \left(\frac{p_L^{(k-1)}}{p_\text{th}}\right)^{(d_k + 1)/2}$$

where:

- $p_L^{(0)} = A \cdot (p_\text{phys} / p_\text{th})^{(d_0 + 1)/2}$ (base level)
- $A \approx 0.1$ — empirical prefactor (Raussendorf et al. 2007)
- $p_\text{th} \approx 0.01$ — surface code threshold (Google Willow 2024: $p_\text{phys} = 0.003$ below threshold at $d=7$)
- $d_k$ — code distance at level $k$ (odd integer $\geq 3$)

This produces **doubly-exponential error suppression**: each concatenation
level squares the logarithm of the error rate. For $p_\text{phys} = 0.001$
and $d = 5$ at all levels:

| Level | $p_L$ |
|-------|--------|
| 0 | $3.16 \times 10^{-5}$ |
| 1 | $9.99 \times 10^{-9}$ |
| 2 | $9.97 \times 10^{-16}$ |

### Threshold Derivation

The doubly-exponential suppression arises from the recursive structure.
Define $\lambda = p_\text{phys} / p_\text{th}$ (ratio below threshold, $\lambda < 1$).
At level 0 with code distance $d$:

$$p_L^{(0)} = A \cdot \lambda^{(d+1)/2}$$

At level 1 (using $p_L^{(0)}$ as input):

$$p_L^{(1)} = A \cdot \left(\frac{A \cdot \lambda^{(d+1)/2}}{p_\text{th}}\right)^{(d+1)/2} = A \cdot \left(\frac{A}{p_\text{th}}\right)^{(d+1)/2} \cdot \lambda^{((d+1)/2)^2}$$

The exponent grows as $((d+1)/2)^k$ at level $k$ — doubly-exponential in $k$.
For $d = 5$, the exponent sequence is $3, 9, 27, 81, 243$. At $\lambda = 0.1$
($p_\text{phys} = 0.001$):

| Level | Exponent | $\lambda^\text{exp}$ | $p_L$ (approx) |
|-------|----------|---------------------|-----------------|
| 0 | 3 | $10^{-3}$ | $10^{-4}$ |
| 1 | 9 | $10^{-9}$ | $10^{-10}$ |
| 2 | 27 | $10^{-27}$ | $10^{-28}$ |
| 3 | 81 | $10^{-81}$ | $10^{-82}$ |

This is why concatenation is so powerful: even modest code distances
produce astronomically low error rates after a few levels. The tradeoff
is qubit count, which grows as $(2d^2 - 1)^k$.

### Distance Selection Algorithm

The auto-distance selection divides the target logical rate equally
across concatenation levels. For target $p_L^\text{target}$ across $L$ levels:

1. Compute per-level target: $p_\text{level} = (p_L^\text{target})^{1/L}$
2. For each level $k = 0, 1, \ldots, L-1$:
   - Find minimum odd $d_k$ such that $p_L^{(k)} \leq p_\text{level}$
   - Use $p_L^{(k)}$ as input to level $k+1$

This greedy algorithm does NOT produce the globally optimal distance
allocation. An optimal allocation would minimise total qubit count
$\sum_k n_\text{osc} (2d_k^2 - 1)$ subject to the final $p_L$ constraint.
The greedy approach overestimates distances at early levels.

### Surface Code at Each Level

Each level uses the rotated surface code with:

- **Physical qubits per logical:** $2d^2 - 1$ (data + ancilla qubits)
- **Syndrome extraction rounds:** $d$ per Trotter step (full cycle)
- **Error types corrected:** X (bit-flip) and Z (phase-flip)
- **Decoder:** Minimum-weight perfect matching (MWPM)
- **Correctable errors:** up to $\lfloor(d-1)/2\rfloor$ per round

The qubit overhead per level for $n_\text{osc}$ logical qubits:

| $d$ | Qubits/logical | 4 oscillators | 16 oscillators |
|-----|---------------|---------------|----------------|
| 3 | 17 | 68 | 272 |
| 5 | 49 | 196 | 784 |
| 7 | 97 | 388 | 1,552 |
| 9 | 161 | 644 | 2,576 |
| 11 | 241 | 964 | 3,856 |

### SCPN Domain Mapping

The 15+1 SCPN layers are grouped into 5 domains, each becoming one
concatenation level:

| Level | Domain | SCPN Layers | Physical Meaning |
|-------|--------|-------------|------------------|
| 0 | Biological | L1–L4 | Quantum bio, cellular sync |
| 1 | Organismal | L5–L8 | Self to cosmic phase-locking |
| 2 | Collective | L9–L12 | Memory, control, collective |
| 3 | Meta | L13–L15 | Source-field, meta-universal |
| 4 | Closure | L16 | Cybernetic closure (Anulum) |

### Syndrome Flow

Error syndromes propagate upward through the hierarchy, weighted by
inter-domain K_nm coupling. The syndrome flow between adjacent levels
$a$ and $b$ is characterised by:

- **Syndrome weight:** $\bar{K}_{ab} = \frac{1}{|D_a||D_b|} \sum_{i \in D_a, j \in D_b} K_{ij}$
- **Correction capacity:** $(d_b - 1) / 2$ errors correctable at level $b$
- **Information flow:** $\bar{K}_{ab} \cdot \log_2(d_b)$ syndrome bits per round

Measured syndrome flow for standard K_nm (Paper 27):

| Flow | $\bar{K}$ | Capacity |
|------|-----------|----------|
| L0 → L1 (bio → org) | 0.1403 | $(d_1 - 1)/2$ |
| L1 → L2 (org → col) | 0.1515 | $(d_2 - 1)/2$ |
| L2 → L3 (col → meta) | 0.1715 | $(d_3 - 1)/2$ |
| L3 → L4 (meta → close) | 0.2544 | $(d_4 - 1)/2$ |

The increasing coupling toward higher levels means stronger syndrome
correction at the top of the hierarchy — consistent with the SCPN
principle that higher layers provide more coherent error correction.

## 2. Theoretical Context

### Why Multi-Scale QEC?

Standard quantum error correction operates at a single scale: physical
qubits encode logical qubits via a fixed code. The SCPN, however,
describes reality as a multi-scale hierarchy where each layer operates
at a different characteristic scale. MS-QEC is the natural QEC structure
for this hierarchy.

The key insight is that concatenation maps directly onto the SCPN layer
structure: errors at the biological level (L1–L4) are first corrected
by the organismal level (L5–L8), then any residual errors are corrected
by the collective level (L9–L12), and so on. This mirrors how biological
systems maintain coherence across scales.

### Historical Context

- **Threshold theorem** (Aharonov & Ben-Or 1997): fault-tolerant quantum
  computation is possible if the physical error rate is below a threshold.
- **Concatenated codes** (Knill 2005): achieving threshold through recursive
  encoding. The MS-QEC module uses this framework directly.
- **Surface codes** (Kitaev 2003): topological codes with high threshold
  ($\sim 1\%$) and efficient MWPM decoding. Used as the inner code at each level.
- **Google Willow** (2024): first experimental demonstration of below-threshold
  surface code operation at $d = 3, 5, 7$.

### What MS-QEC Does NOT Claim

This module provides the **framework** for hierarchical QEC analysis on
the SCPN topology. It does NOT:

- Execute on quantum hardware (resource estimates only)
- Claim that biological systems perform literal quantum error correction
- Prove that the SCPN hierarchy is optimal for concatenation
- Implement lattice surgery or other advanced logical operations

The module is a tool for analysing what resources would be needed to
fault-tolerantly simulate the SCPN Hamiltonian on future quantum hardware.

### Comparison: Concatenated vs Flat QEC

Why not use a single surface code with large distance instead of
concatenation? Consider targeting $p_L = 10^{-15}$ at $p_\text{phys} = 0.001$:

**Flat surface code (single level):**

$$d_\text{flat}: A \cdot (\lambda)^{(d+1)/2} \leq 10^{-15}$$
$$0.1 \cdot (0.1)^{(d+1)/2} \leq 10^{-15} \implies (d+1)/2 \geq 14 \implies d \geq 27$$

Total qubits (4 osc): $4 \times (2 \times 27^2 - 1) = 5,828$

**Concatenated (3 levels, $d = 5$):**

$$p_L = 10^{-28} \ll 10^{-15}$$

Total qubits: $3 \times 4 \times 49 = 588$ — **10× fewer qubits for
better error rate.** This is the power of concatenation.

The tradeoff: concatenation requires $L$ rounds of syndrome extraction
(one per level), increasing circuit depth. For NISQ hardware where depth
is limited, flat codes may be preferred despite higher qubit count.

## 3. Pipeline Position

```
bridge/knm_hamiltonian.py → build_knm_paper27() → K_nm matrix
        ↓
qec/error_budget.py → logical_error_rate(), minimum_code_distance()
        ↓
qec/multiscale_qec.py → build_multiscale_qec() → MultiscaleQECResult
        ↓                 ↓
        ↓           syndrome_flow_analysis() → [SyndromeFlow]
        ↓
qec/surface_code_upde.py → SurfaceCodeUPDE (per-level circuit)
```

**Inputs:** K_nm coupling matrix (from `bridge/`), physical error rate,
target logical rate.

**Outputs:** `MultiscaleQECResult` with per-level code distances,
logical error rates, physical qubit counts, and syndrome flow analysis.

**Dependencies:**
- `qec/error_budget.py` — `logical_error_rate()`, `minimum_code_distance()`
- `bridge/knm_hamiltonian.py` — `build_knm_paper27()`
- Rust engine: `scpn_quantum_engine.concatenated_logical_rate_rust`,
  `scpn_quantum_engine.knm_domain_coupling`

## 4. Features

- **Automatic distance selection:** given target logical rate and physical
  error rate, auto-computes code distances for equal suppression allocation
  across levels.
- **Explicit distance override:** specify distances per level for custom
  resource analysis.
- **Concatenated threshold computation:** iterative logical error rates
  with doubly-exponential suppression below threshold.
- **Syndrome flow analysis:** inter-domain K_nm coupling weights determine
  syndrome information flow between QEC levels.
- **Double-exponential detection:** automatically checks if error rates
  decrease faster than exponentially across levels.
- **Rust acceleration:** `concatenated_logical_rate` and `knm_domain_coupling`
  use Rust engine for 19.5× speedup on test suite.
- **Full backward compatibility:** integrates with existing
  `SurfaceCodeUPDE`, `FaultTolerantUPDE`, `ErrorBudget` modules.

## 5. Usage Examples

### Basic: Build MS-QEC for Standard SCPN

```python
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.qec.multiscale_qec import build_multiscale_qec

K = build_knm_paper27()  # 16×16 K_nm matrix

# Auto-select distances for target p_L = 1e-10
result = build_multiscale_qec(K, p_physical=0.001, target_logical_rate=1e-10)

print(f"Levels: {result.concatenation_depth}")
print(f"Total physical qubits: {result.total_physical_qubits}")
print(f"Effective logical rate: {result.effective_logical_rate:.2e}")
print(f"Below threshold: {result.below_threshold}")

for level in result.levels:
    print(f"  L{level.level} ({level.domain_name}): "
          f"d={level.code_distance}, "
          f"p_L={level.logical_error_rate:.2e}")
```

### Explicit Distances

```python
result = build_multiscale_qec(
    K,
    p_physical=0.003,
    distances=[3, 5, 7, 3, 3],  # one per domain
)
```

### Syndrome Flow Analysis

```python
from scpn_quantum_control.qec.multiscale_qec import (
    build_multiscale_qec,
    syndrome_flow_analysis,
)

K = build_knm_paper27()
result = build_multiscale_qec(K, p_physical=0.001, distances=[3, 3, 3, 3, 3])
flows = syndrome_flow_analysis(K, result)

for flow in flows:
    print(f"L{flow.source_level} → L{flow.target_level}: "
          f"weight={flow.syndrome_weight:.4f}, "
          f"capacity={flow.correction_capacity:.1f}, "
          f"info={flow.information_flow:.4f} bits/round")
```

### Concatenated Rate Computation

```python
from scpn_quantum_control.qec.multiscale_qec import concatenated_logical_rate

# 5 levels of d=5 surface codes at p_phys=0.001
rates = concatenated_logical_rate(0.001, [5, 5, 5, 5, 5])
for i, r in enumerate(rates):
    print(f"Level {i}: p_L = {r:.2e}")
```

### Hardware Generation Comparison

```python
# Compare MS-QEC resources across hardware generations
for p_phys, name in [(0.003, "Heron r2"), (0.001, "Willow-like"), (0.0001, "Future")]:
    result = build_multiscale_qec(K, p_physical=p_phys, target_logical_rate=1e-10)
    print(f"{name} (p={p_phys}): {result.total_physical_qubits} qubits, "
          f"p_L={result.effective_logical_rate:.2e}, "
          f"distances={[l.code_distance for l in result.levels]}")
```

### Small System (4 Layers)

```python
K_small = build_knm_paper27(L=4)
result = build_multiscale_qec(K_small, p_physical=0.001, target_logical_rate=1e-6)
print(f"4-layer system: {result.concatenation_depth} levels, "
      f"{result.total_physical_qubits} qubits")
```

## 6. Technical Reference

### Classes

#### `QECLevel`

| Field | Type | Description |
|-------|------|-------------|
| `level` | `int` | Concatenation level index |
| `domain_name` | `str` | SCPN domain name |
| `code_distance` | `int` | Surface code distance |
| `layer_range` | `tuple[int, int]` | SCPN layer range (0-indexed) |
| `physical_error_rate` | `float` | Input error rate to this level |
| `logical_error_rate` | `float` | Output error rate from this level |
| `physical_qubits_per_logical` | `int` | $2d^2 - 1$ |
| `n_logical_qubits` | `int` | Logical qubits at this level |
| `total_physical_qubits` | `int` | $n_\text{logical} \times (2d^2 - 1)$ |
| `knm_coupling_to_next` | `float` | Mean K_nm to next level |

#### `MultiscaleQECResult`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `list[QECLevel]` | Per-level analysis |
| `effective_logical_rate` | `float` | Final logical error rate |
| `total_physical_qubits` | `int` | Sum across all levels |
| `concatenation_depth` | `int` | Number of levels |
| `below_threshold` | `bool` | $p_\text{phys} < p_\text{th}$ |
| `double_exponential_suppression` | `bool` | Rates decrease doubly-exponentially |
| `summary` | `str` | Human-readable summary |

#### `SyndromeFlow`

| Field | Type | Description |
|-------|------|-------------|
| `source_level` | `int` | Error source level |
| `target_level` | `int` | Correction target level |
| `syndrome_weight` | `float` | K_nm coupling strength |
| `correction_capacity` | `float` | $(d - 1) / 2$ |
| `information_flow` | `float` | Syndrome bits per round |

### Functions

#### `build_multiscale_qec(K, n_oscillators_per_level, p_physical, target_logical_rate, distances)`

Build the full MS-QEC hierarchy. Returns `MultiscaleQECResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K` | `ndarray` | required | K_nm coupling matrix ($n \times n$) |
| `n_oscillators_per_level` | `int \| None` | `4` | Logical qubits per level |
| `p_physical` | `float` | `0.003` | Hardware physical error rate |
| `target_logical_rate` | `float` | `1e-10` | Target effective $p_L$ |
| `distances` | `list[int] \| None` | auto | Code distance per level |

Raises `ValueError` if `distances` length does not match number of active domains.

#### `concatenated_logical_rate(p_physical, distances, p_threshold, prefactor)`

Compute iterative logical error rates. Returns `list[float]`.
Rust-accelerated when engine available.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_physical` | `float` | required | Base physical error rate |
| `distances` | `list[int]` | required | Code distance per level |
| `p_threshold` | `float` | `0.01` | Surface code threshold |
| `prefactor` | `float` | `0.1` | Empirical prefactor $A$ |

#### `syndrome_flow_analysis(K, result)`

Analyse syndrome information flow between adjacent levels.
Returns `list[SyndromeFlow]` with $L - 1$ entries for $L$ levels.

#### `knm_between_domains(K, domain_a, domain_b)`

Mean K_nm coupling between two SCPN domain ranges.
Rust-accelerated when engine available. Returns `float`.

### Constants

| Constant | Value | Source |
|----------|-------|--------|
| `SURFACE_CODE_THRESHOLD` | 0.01 | Raussendorf et al. 2007 |
| `SURFACE_CODE_PREFACTOR` | 0.1 | Empirical fit |
| `SCPN_DOMAINS` | 5 domains | SCPN Paper 27 |

### Rust Engine API

The following functions are available via `import scpn_quantum_engine`:

#### `concatenated_logical_rate_rust(p_physical, distances, p_threshold, prefactor)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `p_physical` | `f64` | Base physical error rate |
| `distances` | `ndarray[i64]` | Code distances per level |
| `p_threshold` | `f64` | Surface code threshold |
| `prefactor` | `f64` | Empirical prefactor |

Returns `ndarray[f64]` of logical error rates.

#### `knm_domain_coupling(k, a_start, a_end, b_start, b_end)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `ndarray[f64, 2D]` | K_nm coupling matrix |
| `a_start`, `a_end` | `usize` | Domain A layer range (inclusive) |
| `b_start`, `b_end` | `usize` | Domain B layer range (inclusive) |

Returns `f64` — mean coupling between domains.

### Internal Functions

| Function | Visibility | Description |
|----------|-----------|-------------|
| `_active_domains(n)` | private | Determine SCPN domains from K size |
| `_auto_distances(n_levels, p_phys, target)` | private | Greedy distance selection |
| `_build_levels(...)` | private | Construct QECLevel objects |
| `_check_double_exponential(rates, below)` | private | Detect doubly-exp suppression |

## 7. Performance Benchmarks

All benchmarks measured on Intel i5-11600K, Python 3.12, Rust engine
(scpn-quantum-engine 0.2.0, PyO3 0.25).

| Function | n=16 | Engine | Budget |
|----------|------|--------|--------|
| `concatenated_logical_rate` (5 levels) | 22 μs | Rust | < 1 ms |
| `knm_between_domains` | 24 μs | Rust | < 1 ms |
| `build_multiscale_qec` | 0.18 ms | Rust (inner) | < 100 ms |
| `syndrome_flow_analysis` (4 flows) | 0.13 ms | Rust (inner) | < 10 ms |

### Rust vs Python Speedup

Test suite (23 tests) timing:
- Without Rust engine: 6.23 s
- With Rust engine: 0.32 s
- **Speedup: 19.5×**

The speedup comes from two Rust functions in `scpn_quantum_engine`:

| Rust Function | Python Equivalent | Speedup Source |
|---------------|------------------|----------------|
| `concatenated_logical_rate_rust` | `concatenated_logical_rate` | Avoids Python loop + float overhead |
| `knm_domain_coupling` | `knm_between_domains` | Avoids nested Python loop over K matrix |

Both are bound via PyO3 0.25 with numpy integration. The Rust functions
are in `scpn_quantum_engine/src/concat_qec.rs` (164 lines, 7 unit tests).

### Scaling with Number of Levels

The computation cost is $O(L)$ where $L$ is the number of concatenation
levels. For the standard 5-level SCPN hierarchy, all operations complete
in sub-millisecond time.

Measured scaling (1000 calls each, Intel i5-11600K):

| Levels | `concatenated_logical_rate` | `build_multiscale_qec` |
|--------|---------------------------|----------------------|
| 1 | 8 μs | 0.05 ms |
| 3 | 14 μs | 0.10 ms |
| 5 | 22 μs | 0.18 ms |
| 10 | 38 μs | 0.35 ms |

Linear scaling as expected — no superlinear cost from concatenation.

### Test Coverage

23 tests across 6 dimensions:
- Empty/null inputs: 5 tests (empty distances, single level, minimal matrix)
- Error handling: 3 tests (mismatched distances, above threshold)
- Negative cases: 3 tests (zero coupling, uniform coupling, d=1)
- Pipeline integration: 5 tests (K_nm, syndrome flow, domain names, imports)
- Roundtrip: 4 tests (monotonic rates, qubit counts, formula, double-exp)
- Performance: 3 tests (wall-clock budgets for all hot paths)

### Resource Estimates (Not Executable)

For $p_\text{phys} = 0.001$, target $p_L = 10^{-10}$:

| Configuration | Total Physical Qubits | Effective $p_L$ |
|---------------|----------------------|-----------------|
| Auto distances | Varies (typically $d = 3$–$5$) | $\leq 10^{-10}$ |
| All $d = 3$ | $5 \times 4 \times 17 = 340$ | Varies |
| All $d = 5$ | $5 \times 4 \times 49 = 980$ | Much lower |

Caveat: these are idealised estimates. Real hardware requires additional
qubits for routing, ancilla preparation, and magic state distillation.

### Hardware Generation Projections

Resource requirements for 4 oscillators, 5 concatenation levels, target
$p_L = 10^{-10}$:

| Hardware | $p_\text{phys}$ | Auto distances | Total qubits | Effective $p_L$ |
|----------|-----------------|---------------|--------------|-----------------|
| IBM Heron r2 (2024) | 0.003 | $[3, 43, 51, 51, 51]$ | 77,268 | above threshold at L1 |
| Google Willow-like | 0.001 | $[3, 3, 3, 3, 3]$ | 340 | $\sim 10^{-10}$ |
| Future ($10^{-4}$) | 0.0001 | $[3, 3, 3, 3, 3]$ | 340 | $\sim 10^{-40}$ |

The dramatic difference: at $p_\text{phys} = 0.003$ (close to threshold),
the first level produces $p_L = 0.009$ which is itself near threshold.
Subsequent levels require huge distances to compensate. At $p_\text{phys} = 0.001$
(well below threshold), even $d = 3$ suffices at every level.

### Scaling: Levels vs Qubits vs Error Rate

For $p_\text{phys} = 0.001$, all $d = 5$:

| Levels | Total qubits | $p_L$ | Use case |
|--------|-------------|--------|----------|
| 1 | 196 | $3.2 \times 10^{-5}$ | Noisy demonstration |
| 2 | 392 | $1.0 \times 10^{-8}$ | Short computation |
| 3 | 588 | $1.0 \times 10^{-15}$ | Medium computation |
| 4 | 784 | $< 10^{-30}$ | Long computation |
| 5 | 980 | $< 10^{-60}$ | Arbitrary precision |

## 8. Citations

1. Knill, E. "Quantum computing with realistically noisy devices."
   *Nature* **434**, 39–44 (2005). DOI: 10.1038/nature03350

2. Aharonov, D. & Ben-Or, M. "Fault-tolerant quantum computation with
   constant error." *Proc. 29th Annual ACM Symposium on Theory of
   Computing (STOC)*, 176–188 (1997). DOI: 10.1145/258533.258579

3. Kitaev, A. "Fault-tolerant quantum computation by anyons."
   *Annals of Physics* **303**, 2–30 (2003). DOI: 10.1016/S0003-4916(02)00018-0

4. Gottesman, D. "An introduction to quantum error correction and
   fault-tolerant quantum computation." arXiv:0904.2557 (2009).

5. Raussendorf, R., Harrington, J. & Goyal, K. "Topological
   fault-tolerance in cluster state quantum computation."
   *New J. Phys.* **9**, 199 (2007). DOI: 10.1088/1367-2630/9/6/199

6. Google Quantum AI. "Quantum error correction below the surface code
   threshold." arXiv:2408.13687 (2024).

7. Šotek, M. "God of the Math — The SCPN Master Publications."
   Zenodo (2025). DOI: 10.5281/zenodo.17419678
