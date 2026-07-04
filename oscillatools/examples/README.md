<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# oscillatools examples

Runnable, deterministic examples that use only the public `oscillatools` facade.
Each script prints a machine-readable JSON summary and needs no credentials or
optional tiers — the NumPy floor runs everything, and any installed accelerator
tier is used automatically.

| Example | What it shows |
|---|---|
| `kuramoto_handbook_workflow.py` | The full handbook path on a six-oscillator sparse chain: RK4 integration, frequency-locking diagnostics, linear stability, coherence clustering, Gaussian critical coupling, and synchronising-coupling design. |

Run one with:

```bash
python examples/kuramoto_handbook_workflow.py
```

The rendered [example gallery](../docs/gallery.md) narrates each workflow.
