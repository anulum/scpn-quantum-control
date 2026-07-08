# p_h1 Open-Claim Guard

The QWC-5.3 guard keeps the `p_h1 = 0.72` threshold publicly framed as an open
empirical/theoretical parameter.

The guard scans outward-facing Markdown and rejects wording that promotes the
open empirical/theoretical threshold as a closed derivation, universal constant,
or measured TCBO reproduction. It also requires open-question markers on public
paragraphs that mention both `p_h1` and `0.72`.

Regenerate the report with:

```bash
scpn-bench p-h1-open-guard
```

The current report is:

- `data/p_h1_open_guard/p_h1_open_guard_2026-07-08.json`

The guard is claim-boundary evidence only. The threshold remains open: it does
not derive `p_h1 = 0.72` and does not replace the TCBO reproduction or
first-principles derivation gates.
