# Paper 0 Two-Timescale Quasicritical Controller Specs

- Source span: P0R06646 - P0R06676
- Source records consumed: 31
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded two-timescale quasicritical controller simulator contract; not empirical evidence

## Specs

### two_timescale_quasicritical.block_framing

Paper 0 frames quasicritical maintenance as a separated-timescale controller for operating near sigma equals one without fine-tuning while maintaining coherence.

Formulae:

Mechanisms:
- two-timescale quasicritical controller
- dual-channel architecture maintains quasicriticality
- separated timescale control uses affective gain scheduling

Null controls:
- missing-timescale-separation control must be rejected
- missing-affective-gain-scheduling control must be rejected
- unsupported-empirical-BIBO-claim control must be rejected

### two_timescale_quasicritical.dual_channel_architecture

The source separates a fast stabilizer channel from a slow explorer channel, with tau_s much larger than tau_f.

Formulae:
- tau_s >> tau_f

Mechanisms:
- fast channel tau_f provides MS-QEC error correction and local homeostatic feedback
- fast gain G_f(sigma,A) maintains coherence and suppresses error
- slow channel tau_s >> tau_f supports controlled drift in the quasicritical band
- slow gain G_s(sigma,A) preserves sensitivity and state-space sampling

Null controls:
- invalid-timescale control must be rejected
- missing-fast-channel control must be rejected
- missing-slow-channel control must be rejected

### two_timescale_quasicritical.affective_gain_scheduling

Affective landscape steepness schedules stabilizing and exploratory gains as functions of sigma and the affective gradient.

Formulae:
- A = -grad F (affective landscape steepness)
- G_f(sigma) = G_f,min + k_f |partial A / partial sigma| + k_f_prime |sigma - 1|
- G_s(sigma) = G_s,max * Window(|sigma - 1| <= delta) * [1 - tanh(c |partial A / partial sigma|)]
- flat landscape + near sigma=1 allows exploration

Mechanisms:
- steep affective gradient prioritizes stability
- near-critical flat landscape permits exploration
- G_f increases with affective-gradient magnitude and sigma deviation
- G_s is enabled only inside the quasicritical window and decreases with steepness

Null controls:
- invalid-delta control must be rejected
- negative-gain-parameter control must be rejected
- non-finite-affective-gradient control must be rejected

### two_timescale_quasicritical.bibo_stability_certificate

The source gives a composite Lyapunov certificate for bounded trajectories under timescale separation and bounded noise.

Formulae:
- V_total = V_fast + V_slow
- V_total = (sigma - 1)^2 + beta (R - R_star)^2
- under tau_f / tau_s << 1: dV_total/dt <= -alpha_f V_fast - alpha_s V_slow + bounded noise
- all trajectories remain bounded (BIBO stable)

Mechanisms:
- V_fast tracks quasicritical sigma deviation
- V_slow tracks coherence deviation from R_star
- bounded-noise drift inequality is the simulator-level certificate

Null controls:
- negative-beta control must be rejected
- negative-alpha control must be rejected
- unsupported-BIBO-empirical-claim control must be rejected

### two_timescale_quasicritical.operational_consequence

The source maps high surprise to exploitation and low surprise near criticality to exploration at the level of criticality maintenance.

Formulae:
- high surprise steep |partial A / partial sigma| drives G_f up and G_s down for exploit
- low surprise flat |partial A / partial sigma| near sigma=1 maintains G_f and raises G_s for explore
- exploration-exploitation dilemma is addressed at the level of criticality maintenance

Mechanisms:
- high surprise prioritizes stabilizing exploitation
- low surprise inside the quasicritical band enables exploratory drift
- criticality maintenance is the control surface for exploration-exploitation

Null controls:
- outside-band-exploration control must be zero
- sign-inverted-surprise control must be rejected
- unsupported-exploration-solution-claim control must be rejected
