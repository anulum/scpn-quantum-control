# Security Policy

Machine-readable contact information is served at
[`docs/.well-known/security.txt`](docs/.well-known/security.txt)
(RFC 9116) and published at
<https://anulum.github.io/scpn-quantum-control/.well-known/security.txt>.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.9.x   | Yes       |
| < 0.9   | No        |

## Reporting

Two supported channels:

1. **Email** — `protoscience@anulum.li`. Encrypt with PGP if you have
   a published key; plain-text is acceptable.
2. **GitHub Security Advisory** —
   <https://github.com/anulum/scpn-quantum-control/security/advisories/new>.
   Private to project maintainers until published.

Do not open a public issue or pull request for a suspected security
bug — either of the channels above keeps the disclosure coordinated.

Expected response: 48 h acknowledgment, 7-day initial triage, fix or
mitigation timeline communicated in the acknowledgement. A CVE number
is assigned for any issue that results in a published advisory.

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
