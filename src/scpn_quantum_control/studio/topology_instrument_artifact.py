# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed data of the symmetry, chimera and topology instrument
"""Build, write and check the source data of the symmetry, chimera and topology lenses.

The Studio shows these values without recomputing them. Run
``python -m scpn_quantum_control.studio.topology_instrument_artifact --write``
after an owner changes, and ``--check`` to compare the committed data with a
fresh build.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from .committed_data import run_committed_data_cli
from .topology_instrument_templates import (
    chimera_template,
    persistence_template,
    relabelling_template,
    sector_template,
    sensitivity_template,
)

TOPOLOGY_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_topology_instrument_v1"
DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data/studio/topology_instrument_20261114.json"
)
LAYOUT_BOUNDARY: Final[str] = (
    "node positions in the 3D view are a display projection computed in the browser; "
    "they carry no physical meaning and change no value"
)


def _global_state() -> bytes:
    # The serialised legacy state (generator name, key, position and cached
    # Gaussian) compares exactly without inspecting its parts.
    return pickle.dumps(np.random.get_state(legacy=True))


def build_topology_instrument_artifact() -> dict[str, object]:
    """Build every topology template and refuse a change to the caller's random state.

    Returns
    -------
    dict[str, object]
        Source-owned relabelling, sector, chimera, persistence and sensitivity
        templates, the layout boundary and the recorded statement that the
        global random state was left unchanged.

    Raises
    ------
    RuntimeError
        If building the templates changed the legacy global NumPy random state.

    """
    before = _global_state()
    templates: dict[str, object] = {
        "relabelling": relabelling_template(),
        "sectors": sector_template(),
        "chimera": chimera_template(),
        "persistence": persistence_template(),
        "sensitivity": sensitivity_template(),
    }
    if _global_state() != before:
        raise RuntimeError("a topology template changed the caller's global random state")
    return {
        "schema": TOPOLOGY_INSTRUMENT_SCHEMA,
        "artifact_id": "studio-topology-instrument-20261114",
        "generated_by": "python -m scpn_quantum_control.studio.topology_instrument_artifact --write",
        "global_random_state_unchanged": True,
        "layout_boundary": LAYOUT_BOUNDARY,
        "templates": templates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed topology instrument data.

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
        build=build_topology_instrument_artifact,
        default_path=DEFAULT_PATH,
        label="topology instrument data",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
