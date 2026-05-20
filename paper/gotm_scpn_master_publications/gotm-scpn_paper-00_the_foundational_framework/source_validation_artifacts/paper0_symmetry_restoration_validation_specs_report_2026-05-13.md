# Paper 0 Symmetry Restoration Specs

- Source records: `15`
- Consumed source records: `15`
- Coverage status: `match`
- Source span: `P0R06324, P0R06338`
- Spec count: `5`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### symmetry_restoration.mmc_conformal_geometry_boundary

- Protocol: `paper0.symmetry_restoration.mmc_conformal_geometry_boundary`
- Source ledgers: `P0R06324, P0R06325, P0R06326, P0R06327, P0R06328, P0R06329, P0R06330, P0R06331, P0R06332, P0R06333, P0R06334, P0R06335, P0R06336, P0R06337, P0R06338`
- Source formulae: `g_hat_mu_nu = Omega^2 g_mu_nu`
- Null controls: `3`
- Claim boundary: `source-bounded symmetry-restoration simulator contract; not empirical evidence`

### symmetry_restoration.conformal_boundary_masslessness_constraint

- Protocol: `paper0.symmetry_restoration.conformal_boundary_masslessness_constraint`
- Source ledgers: `P0R06324, P0R06325, P0R06326, P0R06327, P0R06328, P0R06329, P0R06330, P0R06331, P0R06332, P0R06333, P0R06334, P0R06335, P0R06336, P0R06337, P0R06338`
- Source formulae: `none`
- Null controls: `3`
- Claim boundary: `source-bounded symmetry-restoration simulator contract; not empirical evidence`

### symmetry_restoration.effective_potential_flip_boundary

- Protocol: `paper0.symmetry_restoration.effective_potential_flip_boundary`
- Source ledgers: `P0R06324, P0R06325, P0R06326, P0R06327, P0R06328, P0R06329, P0R06330, P0R06331, P0R06332, P0R06333, P0R06334, P0R06335, P0R06336, P0R06337, P0R06338`
- Source formulae: `V_eff(|Psi|) = (-mu^2 + c1 T_dS^2 + c2 f(R)) |Psi|^2 + lambda |Psi|^4`
- Null controls: `3`
- Claim boundary: `source-bounded symmetry-restoration simulator contract; not empirical evidence`

### symmetry_restoration.vev_melting_massless_limit

- Protocol: `paper0.symmetry_restoration.vev_melting_massless_limit`
- Source ledgers: `P0R06324, P0R06325, P0R06326, P0R06327, P0R06328, P0R06329, P0R06330, P0R06331, P0R06332, P0R06333, P0R06334, P0R06335, P0R06336, P0R06337, P0R06338`
- Source formulae: `lim_{t -> infinity} v(t) = 0, m_A = g v; m_h = sqrt(2 lambda) v`
- Null controls: `3`
- Claim boundary: `source-bounded symmetry-restoration simulator contract; not empirical evidence`

### symmetry_restoration.legal_conformal_rescaling_boundary

- Protocol: `paper0.symmetry_restoration.legal_conformal_rescaling_boundary`
- Source ledgers: `P0R06324, P0R06325, P0R06326, P0R06327, P0R06328, P0R06329, P0R06330, P0R06331, P0R06332, P0R06333, P0R06334, P0R06335, P0R06336, P0R06337, P0R06338`
- Source formulae: `none`
- Null controls: `3`
- Claim boundary: `source-bounded symmetry-restoration simulator contract; not empirical evidence`

## Policy

These records are source-anchored symmetry-restoration specifications only. Passing any fixture is not empirical evidence and does not validate MMC, CCC, de Sitter asymptotics, or far-future conformal-boundary claims.
