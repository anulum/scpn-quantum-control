# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — p_h1 Open-Claim Guard Data

# p_h1 Open-Claim Guard Data

This directory stores the QWC-5.3 public-claim guard report for the open
`p_h1 = 0.72` threshold.

Artifact:

| Artifact | Contents |
|---|---|
| `p_h1_open_guard_2026-07-08.json` | Public Markdown scan result, checked p_h1 surfaces, open-boundary marker hits, and wording violations. |

Regenerate with:

```bash
scpn-bench p-h1-open-guard
```

The guard passes only when public wording keeps `p_h1 = 0.72` as an open
empirical/theoretical parameter and does not present it as a closed derivation,
universal constant, or measured TCBO reproduction.
