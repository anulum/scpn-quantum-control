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

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY .pre-commit-config.yaml pyproject.toml requirements.txt requirements-dev.txt README.md LICENSE ./
COPY requirements-ci-cross-platform-smoke.txt requirements-ci-py311-linux.txt requirements-ci-py312-linux.txt requirements-ci-py313-linux.txt requirements-ci-studio-platform.txt ./
COPY src/ src/

ENV PYTHONPATH=/app/src:/app
ENV XDG_CACHE_HOME=/home/sqc/.cache
ENV XDG_CONFIG_HOME=/home/sqc/.config
ENV MPLCONFIGDIR=/home/sqc/.config/matplotlib
# Amazon Braket imports its default simulator during adapter collection; Numba
# cache locators can fail in copied container layers, so Docker CI disables JIT.
ENV NUMBA_DISABLE_JIT=1

RUN pip install --no-cache-dir --require-hashes -r requirements-ci-py312-linux.txt \
    && pip install --no-cache-dir --no-deps --require-hashes -r requirements-ci-studio-platform.txt

COPY tests/ tests/
COPY tools/ tools/
COPY .github/workflows/ .github/workflows/
COPY scpn_quantum_engine/src/ scpn_quantum_engine/src/
COPY scpn_quantum_engine/tests/ scpn_quantum_engine/tests/
RUN printf '%s\n' \
    'raise ModuleNotFoundError("compiled scpn_quantum_engine extension is not installed in this image", name="scpn_quantum_engine")' \
    > scpn_quantum_engine/__init__.py
COPY docs/ docs/
COPY paper/ paper/
COPY examples/ examples/
COPY notebooks/ notebooks/
COPY results/ results/
# `data/` holds curated hardware-result JSONs that
# `tests/test_phase1_dla_parity_reproduces.py` asserts against; the
# reproducer ERRORs out without the fixture. `scripts/` holds the
# analysis module the reproducer imports.
COPY data/ data/
COPY figures/ figures/
COPY scripts/ scripts/

RUN mkdir -p /home/sqc/.cache/pytest /home/sqc/.config/matplotlib \
    && chown -R sqc:sqc /home/sqc/.cache /home/sqc/.config

USER sqc

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import scpn_quantum_control; print('OK')"

# Skip slow, hardware, private-corpus, and machine-dependent performance tests by default.
CMD ["pytest", "tests/", "-v", "--tb=short", "-o", "cache_dir=/home/sqc/.cache/pytest", "-m", "not slow and not hardware and not internal_corpus and not performance"]
