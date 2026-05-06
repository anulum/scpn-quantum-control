# S3 Pulse Feasibility Probe

Submission state: no provider session and no hardware submission.

## Decisions
- `ibm_pulse_metadata_template` (ibm_pulse): ready (provider metadata satisfies S3 pulse schedule requirements)
- `neutral_atom_xy_review_template` (neutral_atom_analog): ready (provider metadata satisfies S3 pulse schedule requirements)
- `unknown_pulse_target` (metadata_light): unknown (provider declares neither pulse control nor native XY execution; provider did not declare supported_features)

## Claim Boundary
This is a metadata-only feasibility probe. It does not calibrate pulses, open provider sessions, submit jobs, or establish hardware performance.
