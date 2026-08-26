# Domain Application Honesty Evidence

- Schema: `scpn.application-honesty.v1`
- Result: `PASS`
- Content digest: `8f49455d3578b219b5220767491946fcdc2da7b833cd03a5837fb041805d8a37`
- Claim boundary: software-contract and synthetic-or-curated benchmark evidence only; not domain validation, operational control, clinical use, facility prediction, hardware performance, or quantum advantage.

## Honesty kits

| Kit | Support | Data origin | Synthetic only | Forecasting tags |
|---|---|---|:---:|---|
| `power_grid_public_benchmark` | `bounded_research` | `curated_public` | `false` | `grid_like_sim` |
| `josephson_illustrative_simulation` | `simulation_only` | `synthetic` | `true` | `none` |
| `eeg_like_synthetic` | `simulation_only` | `synthetic` | `true` | `eeg_like_sim` |
| `iter_disruption_inspired_simulation` | `simulation_only` | `synthetic` | `true` | `plasma_like_sim` |

## Packaged dataset privacy audit

| Dataset | Source mode | Privacy class | Personal data | Result |
|---|---|---|:---:|:---:|
| `eeg_alpha_plv_8ch` | `curated` | `public_curated_no_personal_data` | `false` | `PASS` |
| `iter_mhd_8mode` | `curated` | `public_curated_no_facility_trace` | `false` | `PASS` |
| `ieee5bus_power_grid` | `curated` | `public_benchmark_constants` | `false` | `PASS` |
| `friston_fep_6node` | `curated` | `public_curated_no_human_subject_data` | `false` | `PASS` |

This evidence validates software metadata and packaged-artifact privacy boundaries only. It is not domain, clinical, facility, hardware, or advantage evidence.
