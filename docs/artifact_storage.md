# Artifact Storage Audit

This audit covers committed files under `results/`, `data/`, and
`figures/` as of 2026-04-29. It exists to decide whether the current
artifact footprint belongs in Git, Git LFS, or a Zenodo/download-only
route.

## Provenance

Commands:

```bash
git ls-files results data figures | wc -l
git ls-files -z results data figures | xargs -0 du -cb | tail -1
git ls-files -z results | xargs -0 du -cb | tail -1
git ls-files -z data | xargs -0 du -cb | tail -1
git ls-files -z figures | xargs -0 du -cb | tail -1
git ls-files -z results data figures | xargs -0 du -b | sort -nr | head -30
```

## Current Footprint

| Path set | Committed files | Bytes |
|----------|----------------:|------:|
| `results/` | counted in aggregate | 1,433,967 |
| `data/` | counted in aggregate | 393,401 |
| `figures/` | counted in aggregate | 5,215,506 |
| Total | 169 | 7,042,874 |

Largest committed files:

| File | Bytes | Storage route |
|------|------:|---------------|
| `results/upde_16_t01.json` | 447,140 | Keep in Git |
| `results/upde_16_snapshot.json` | 393,405 | Keep in Git |
| `figures/publication/fig3_otoc_time_traces.png` | 391,408 | Keep in Git |
| `figures/publication/fig7_combined_transition.png` | 365,454 | Keep in Git |
| `figures/publication/fig12_full_hardware_analysis.png` | 346,294 | Keep in Git |

## Decision

No current committed artifact requires Git LFS or Zenodo-only storage.
The audited footprint is about 7.0 MB and the largest single file is
under 0.45 MB. Publication figures and reduced reproducibility JSON
stay in Git so tests and documentation remain self-contained.

Use Zenodo or Git LFS for future additions when a single artifact
exceeds 25 MB, an artifact set exceeds 100 MB, or the data are raw
campaign outputs rather than reduced reproducibility inputs.
