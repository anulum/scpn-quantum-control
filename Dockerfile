# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

FROM python:3.12-slim

LABEL org.opencontainers.image.title="scpn-quantum-control"
LABEL org.opencontainers.image.description="NISQ quantum simulation of coupled Kuramoto oscillator networks"
LABEL org.opencontainers.image.source="https://github.com/anulum/scpn-quantum-control"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

RUN useradd --create-home sqc
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/
COPY examples/ examples/

USER sqc

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import scpn_quantum_control; print('OK')"

CMD ["pytest", "tests/", "-v", "--tb=short"]
