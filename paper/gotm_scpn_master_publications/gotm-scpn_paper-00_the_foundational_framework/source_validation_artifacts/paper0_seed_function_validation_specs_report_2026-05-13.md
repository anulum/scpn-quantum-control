# Paper 0 Seed Function Specs

- Source records: `15`
- Consumed source records: `15`
- Coverage status: `match`
- Source span: `P0R06363, P0R06377`
- Spec count: `4`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### seed_function.python_format_source_boundary

- Protocol: `paper0.seed_function.python_format_source_boundary`
- Source ledgers: `P0R06363, P0R06364, P0R06365, P0R06366, P0R06367, P0R06368, P0R06369, P0R06370, P0R06371, P0R06372, P0R06373, P0R06374, P0R06375, P0R06376, P0R06377`
- Source formulae: `def compute_teleological_seed(prev_cycle_sec, coupling_constant_g):`
- Null controls: `3`
- Claim boundary: `source-bounded seed-function simulator contract; not empirical evidence`

### seed_function.mu_squared_seed_formula

- Protocol: `paper0.seed_function.mu_squared_seed_formula`
- Source ledgers: `P0R06363, P0R06364, P0R06365, P0R06366, P0R06367, P0R06368, P0R06369, P0R06370, P0R06371, P0R06372, P0R06373, P0R06374, P0R06375, P0R06376, P0R06377`
- Source formulae: `mu_squared_seed = sqrt(prev_cycle_sec / coupling_constant_g)`
- Null controls: `3`
- Claim boundary: `source-bounded seed-function simulator contract; not empirical evidence`

### seed_function.return_payload_contract

- Protocol: `paper0.seed_function.return_payload_contract`
- Source ledgers: `P0R06363, P0R06364, P0R06365, P0R06366, P0R06367, P0R06368, P0R06369, P0R06370, P0R06371, P0R06372, P0R06373, P0R06374, P0R06375, P0R06376, P0R06377`
- Source formulae: `ssb_bias_magnitude = mu_squared_seed, is_random_reset = False`
- Null controls: `3`
- Claim boundary: `source-bounded seed-function simulator contract; not empirical evidence`

### seed_function.conformal_continuity_boundary

- Protocol: `paper0.seed_function.conformal_continuity_boundary`
- Source ledgers: `P0R06363, P0R06364, P0R06365, P0R06366, P0R06367, P0R06368, P0R06369, P0R06370, P0R06371, P0R06372, P0R06373, P0R06374, P0R06375, P0R06376, P0R06377`
- Source formulae: `conformal_continuity = prev_cycle_sec > 0`
- Null controls: `3`
- Claim boundary: `source-bounded seed-function simulator contract; not empirical evidence`

## Policy

These records are source-anchored seed-function specifications only. Passing any fixture is not empirical evidence and does not validate teleological seeding or MMC continuity.
