# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting

Report vulnerabilities via email to protoscience@anulum.li.
Do not open public issues for security bugs.

Expected response: 48 hours acknowledgment, 7-day fix timeline.

## Hardening Measures

- **No pickle deserialization** of untrusted data. Hardware results stored as JSON.
- **No subprocess calls** with user-controlled strings.
- **IBM credentials** never stored in code or committed. Use `QISKIT_IBM_TOKEN` env var or `~/.qiskit/qiskit-ibm.json`.
- **Circuit depth bounds**: all hardware experiments enforce `max_depth` to prevent resource exhaustion on QPU.
- **Shot count limits**: capped at 100k to prevent budget drain.
- **RNG isolation**: each module uses local `numpy.random.Generator` instances, no global state mutation.

## Dependencies

- Qiskit releases are pinned to `>=1.0.0` (stable API). We track Qiskit security advisories.
- `qiskit-ibm-runtime` uses IBM's OAuth2 token flow; tokens are scoped per instance.
- Dependabot monitors GitHub Actions dependencies weekly.
