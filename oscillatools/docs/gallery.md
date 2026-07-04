<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Example Gallery

Every example is a runnable, deterministic Python script that uses only the
public `oscillatools` facade. Each prints a machine-readable JSON summary, so it
doubles as an executable contract and as copyable starting code.

## Handbook worked workflow

`examples/kuramoto_handbook_workflow.py` — a six-oscillator workflow that covers
the full handbook path on one sparse-chain problem: accelerated RK4 integration,
frequency-locking diagnostics, a linear-stability spectrum, phase-coherence
clustering, Gaussian mean-field critical coupling, and a projected
synchronising-coupling design pass. It reports the initial and final order
parameter, the order parameter after the coupling redesign, the locked fraction,
the spectral gap, the cluster partition, and the design cost history.

```bash
python examples/kuramoto_handbook_workflow.py
```

The design pass lifts the final order parameter well above the open-loop run on
the same problem, which is the point the example demonstrates: the differentiable
coupling design is wired to the same facade the diagnostics read. Compare the
printed diagnostics against the [handbook](handbook.md) capability and benchmark
tables.
