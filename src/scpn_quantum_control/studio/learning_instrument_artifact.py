# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed learning, forecasting and domain-benchmark instrument data
"""Project learning, forecasting and domain-benchmark evidence for the Studio.

Each template comes from its source owner: multimodal forecasting with split
custody, the seeded QSNN trainer with an exact checkpoint resume, the committed
JAX NQS baseline, the curated EEG, power-grid and plasma proxies under their
honesty kits, and the free-energy workflow with its covariance refusals. Seeded
work uses local generators only; the caller's global NumPy random state is
compared before and after, and any change refuses the build.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from .committed_data import run_committed_data_cli
from .learning_instrument_templates import (
    domain_proxy_templates,
    fep_template,
    forecasting_template,
    nqs_template,
    qsnn_template,
)

LEARNING_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_learning_instrument_v1"
DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data/studio/learning_instrument_20261114.json"
)


def _global_state() -> bytes:
    # The serialised legacy state (generator name, key, position and cached
    # Gaussian) compares exactly without inspecting its parts.
    return pickle.dumps(np.random.get_state(legacy=True))


def build_learning_instrument_artifact() -> dict[str, object]:
    """Build every learning template and refuse a change to the caller's random state.

    Returns
    -------
    dict[str, object]
        Source-owned forecasting, QSNN, NQS, domain-proxy and free-energy
        templates, with the recorded statement that the global random state was
        left unchanged.

    Raises
    ------
    RuntimeError
        If building the templates changed the legacy global NumPy random state.

    """
    before = _global_state()
    templates: dict[str, object] = {
        "forecasting": forecasting_template(),
        "qsnn": qsnn_template(),
        "nqs": nqs_template(),
        "domain_proxies": domain_proxy_templates(),
        "free_energy": fep_template(),
    }
    if _global_state() != before:
        raise RuntimeError("a learning template changed the caller's global random state")
    return {
        "schema": LEARNING_INSTRUMENT_SCHEMA,
        "artifact_id": "studio-learning-instrument-20261114",
        "generated_by": "python -m scpn_quantum_control.studio.learning_instrument_artifact --write",
        "global_random_state_unchanged": True,
        "templates": templates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed learning instrument data.

    Parameters
    ----------
    argv
        CLI arguments, or the process arguments when omitted.

    Returns
    -------
    int
        The shared committed-data writer's exit status.

    """
    return run_committed_data_cli(
        argv,
        build=build_learning_instrument_artifact,
        default_path=DEFAULT_PATH,
        label="learning instrument data",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
