# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
#
# PURPOSE: reproduction / CI test image — NOT a production runtime.
# This image exists to run the test suite in a clean, pinned container
# (its default CMD is pytest, and .github/workflows/docker.yml builds it
# and runs the tests inside — it is never pushed to a registry). It
# therefore deliberately ships tests/, docs/, paper/, notebooks/, data/,
# and CI fixtures, and it does NOT install the compiled scpn_quantum_engine
# extension (the module is stubbed to fail loudly): the Python tier runs on
# its pure-Python fallbacks so the image stays free of a Rust toolchain.
# Do NOT slim this into a runtime image — slimming would defeat its only
# job (reproducing the full test run). For a production deployment, install
# the published wheel (`pip install scpn-quantum-control`) into your own
# base image instead of reusing this one.

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

COPY Dockerfile Dockerfile
COPY .pre-commit-config.yaml pyproject.toml mkdocs.yml requirements.txt requirements-dev.txt README.md LICENSE ROADMAP.md ./
# The changelog, public-claim, and rendered-docs-header guards read these
# root documents.
COPY CHANGELOG.md VALIDATION.md RESULTS_SUMMARY.md CAPABILITIES_AND_USECASES.md REUSE.toml ./
COPY ARCHITECTURE.md DEPRECATIONS.md CONTRIBUTORS.md GOVERNANCE.md NOTICE.md SUPPORT.md ./
COPY requirements-ci-cross-platform-smoke.txt requirements-ci-py311-linux.txt requirements-ci-py312-linux.txt requirements-ci-py313-linux.txt requirements-ci-studio-platform.txt ./
COPY src/ src/
COPY oscillatools/src/ oscillatools/src/
# The standalone-package decision and real wheel tests require the complete
# Hatchling metadata pair, including the README declared by its pyproject.
COPY oscillatools/pyproject.toml oscillatools/pyproject.toml
COPY oscillatools/README.md oscillatools/README.md

ENV PYTHONPATH=/app/src:/app/oscillatools/src:/app
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
COPY .github/dependabot.yml .github/dependabot.yml
# The static Rust-inventory, dependency-evidence, and kernel-execution audits
# read the engine crate manifest, its locked dependency graph, and every
# in-tree pyo3-featured member crate the ST-12 program-AD replay extraction
# introduced. Ship those inputs alongside the primary crate.
COPY scpn_quantum_engine/Cargo.toml scpn_quantum_engine/Cargo.toml
COPY scpn_quantum_engine/Cargo.lock scpn_quantum_engine/Cargo.lock
COPY scpn_quantum_engine/src/ scpn_quantum_engine/src/
COPY scpn_quantum_engine/program_ad_replay/src/ scpn_quantum_engine/program_ad_replay/src/
COPY scpn_quantum_engine/tests/ scpn_quantum_engine/tests/
COPY scpn_quantum_engine/fuzz/ scpn_quantum_engine/fuzz/
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
# `benchmarks/` holds the committed regression baselines + threshold policies
# that the tier-benchmark and native-speedup gate guards read live-tree
# (tests/test_tier_benchmark_regression_gate.py fails closed without them).
COPY benchmarks/ benchmarks/

# Git-backed repository-policy audits need an index, but the host .git tree,
# history, remotes, objects, and credentials are deliberately excluded from the
# build context. Apply the repository ignore contract, then create a
# credential-free synthetic Git index over the curated tracked files copied
# above. Ignored fixtures remain readable without becoming policy inputs.
COPY .gitignore .gitignore
RUN git init -q \
    && git add -A \
    && chown sqc:sqc /app /app/.git

RUN mkdir -p /home/sqc/.cache/pytest /home/sqc/.config/matplotlib \
    && chown -R sqc:sqc /home/sqc/.cache /home/sqc/.config

USER sqc

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import scpn_quantum_control; print('OK')"

# Skip slow, hardware, private-corpus, and machine-dependent performance tests by default.
CMD ["pytest", "tests/", "-v", "--tb=short", "-o", "cache_dir=/home/sqc/.cache/pytest", "-m", "not slow and not hardware and not internal_corpus and not performance"]
