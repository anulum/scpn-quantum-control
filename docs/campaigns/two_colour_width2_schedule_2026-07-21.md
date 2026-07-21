# Two-edge-colour width-2 schedule for the XY-Trotter chain — 2026-07-21

**Status:** design + classical evidence complete. Addresses audit AUD-7 (B-5):
hand-scheduled, genuinely width-2 chain dynamics instead of a deep serial
synthesis. The hardware submission remains an owner-gated QPU run (AUD-5/7).
**Code:** `src/scpn_quantum_control/analysis/two_colour_schedule.py`, tests
`tests/test_two_colour_schedule.py` (24).

## Problem

The max-width Kuramoto-XY campaign emits the 1-D chain edge-by-edge (a single
`PauliEvolutionGate` that the transpiler synthesises serially), so each Trotter
step is laid out as ``n−1`` sequential two-qubit interactions and the two-qubit
depth grows like ``O(reps · n)``. A referee can object that the reported
"width" is a property of a deep serial circuit, not of parallel dynamics.

## The 2-edge-colouring

The path graph is 2-edge-colourable: even-indexed edges ``{(0,1),(2,3),…}`` form
colour A and odd-indexed edges ``{(1,2),(3,4),…}`` form colour B. Every edge
within a colour acts on disjoint qubits, so a colour is a single parallel layer.
Each Trotter step becomes: one single-qubit ``rz`` layer, then colour A, then
colour B — a **constant** number of two-qubit layers independent of ``n``.

This is the same first-order Trotter approximation (a reordering of the same
`rz` / `rxx·ryy` terms), and every gate conserves total excitation number, so
the scheduled circuit realises genuine parity-preserving width-2 dynamics.

## Classical evidence (measured, this module)

- **Excitation-number conservation preserved:** the exact opposite-parity
  probability of the 2-colour circuit is `< 1e-9` for every tested width, initial
  state, and depth — identical to the serial baseline. The reschedule does not
  change the physics.
- **Constant two-qubit depth per step (genuine width-2):** for a fixed
  Trotter-step count the 2-colour two-qubit depth is invariant across
  ``n = 4, 8, 16, 32`` (measured: 20 layers at 5 steps = 4 layers/step), and
  scales linearly with the step count. The serial baseline grows with ``n``
  (30 → 78 two-qubit layers for ``n = 8 → 32`` at 5 steps), so the reduction
  factor increases with chain length (≈1.5× at n=8, ≈3.9× at n=32, and larger
  still at the campaign widths). Measured conservatively against a clean
  edge-by-edge `rxx·ryy` baseline; against the campaign's transpiled
  `PauliEvolutionGate` synthesis the reduction is larger.

## What this unblocks

A hardware run of the 2-colour-scheduled circuits would test the same XY
dynamics at a fraction of the depth, separating genuine width from
scheduling-induced noise. That submission is owner-gated (QPU); this module and
its classical evidence are the design deliverable.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
