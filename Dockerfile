# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

FROM python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4

LABEL org.opencontainers.image.title="scpn-quantum-control"
LABEL org.opencontainers.image.description="NISQ quantum simulation of coupled Kuramoto oscillator networks"
LABEL org.opencontainers.image.source="https://github.com/anulum/scpn-quantum-control"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

RUN useradd --create-home sqc
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[dev,config,logging]"

COPY tests/ tests/
COPY examples/ examples/
COPY results/ results/
# `data/` holds curated hardware-result JSONs that
# `tests/test_phase1_dla_parity_reproduces.py` asserts against; the
# reproducer ERRORs out without the fixture. `scripts/` holds the
# analysis module the reproducer imports.
COPY data/ data/
COPY scripts/ scripts/

USER sqc

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import scpn_quantum_control; print('OK')"

# Skip DLA (27 min/test) and hardware runner (needs IBM creds) by default
CMD ["pytest", "tests/", "-v", "--tb=short", "--ignore=tests/test_dynamical_lie_algebra.py", "--ignore=tests/test_hardware_runner.py"]
