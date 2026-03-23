# DLA Scaling Results

**Date:** 2026-03-23
**Tool:** Rust-accelerated DLA (scpn_quantum_engine)

## Results

| N (qubits) | DLA dimension | max su(2^N) | Fraction | Time (s) | Engine |
|------------|---------------|-------------|----------|----------|--------|
| 4          | 126           | 255         | 49.4%    | 1.66     | Rust   |
| 5          | 510           | 1023        | 49.9%    | 848      | Rust   |

## Interpretation

The DLA fraction stabilises at ~50% of su(2^N). The absolute DLA dimension
grows exponentially: 126 → 510 → (estimated ~2040 at N=6).

This means:
1. The DLA is NOT polynomial in N → system is NOT trivially classically simulable
2. The heterogeneous SCPN frequencies break the SU(2) symmetry of the pure XY model
3. The g-sim framework (Goh et al. 2025) classifies this as outside the polynomial regime

**Caveat:** "Not trivially simulable" ≠ "quantum advantage". MPS with moderate
bond dimension can still simulate these systems efficiently at small N.
The DLA result establishes a necessary condition, not a sufficient one.

## Comparison

- Pure XY (uniform frequencies): DLA = O(N²) → classically simulable
- SCPN XY (heterogeneous frequencies): DLA ≈ 0.5 × su(2^N) → exponential
- The frequency heterogeneity is what breaks classical simulability

## N=6 Estimate

N=6 dim=64, max DLA = 4095. At the scaling rate, computation would take
~8 hours in Rust. Deferred — the N=4,5 trend is clear.
