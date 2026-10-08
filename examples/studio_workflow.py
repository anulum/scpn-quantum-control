# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original compiler sweep reproduction
"""Emit an original six-cell graph for the checkpointed workflow command.

python examples/studio_workflow.py > compile-workflow.json
scpn-studio-workflow compile-workflow.json --journal compile-journal.json
scpn-studio-workflow compile-workflow.json --journal compile-journal.json --resume

Both stages emit source/compiler evidence. They do not execute quantum gates.
"""

from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio_workspace.json_transport import write_json


def compiler_workflow() -> dict[str, object]:
    """Compose exact native source-emission and compiler-trace stages over a 2x3 grid.

    Returns
    -------
    dict[str, object]
        Admitted portable graph; seed zero identifies cells and is not a stochastic
        input to this deterministic compiler. Original handler limits still apply.

    """
    programs = [
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];',
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; x q[0];',
    ]
    port = {"schema": "studio.program-source.v1", "dtype": "utf8", "shape": [], "unit": "1"}
    document = {
        "schema": "experiment_workflow.v1",
        "body": {
            "workflow_id": "compile-sweep",
            "stages": [
                {
                    "id": "trace",
                    "adapter": "executive",
                    "verb": "compile",
                    "backend": "python",
                    "parameters": {"compiler_trace": True},
                    "inputs": [
                        {
                            "parameter": "program_source",
                            "source_stage": "source",
                            "source_port": "program",
                            "type": port,
                        }
                    ],
                    "outputs": [],
                    "depends_on": [],
                },
                {
                    "id": "source",
                    "adapter": "executive",
                    "verb": "compile",
                    "backend": "python",
                    "parameters": {"program_source": programs[0]},
                    "inputs": [],
                    "outputs": [
                        {
                            "name": "program",
                            "path": ["result", "outputs", "program", "source"],
                            "type": port,
                        }
                    ],
                    "depends_on": [],
                },
            ],
            "sweep": {
                "axes": [
                    {"stage_id": "source", "parameter": "program_source", "values": programs},
                    {"stage_id": "trace", "parameter": "optimisation_level", "values": [0, 1, 2]},
                ],
                "seeds": ["0"],
                "seed_binding": None,
                "evaluation_budget": 12,
            },
        },
        "extensions": {
            "source_note": "seed identifies the cell; this deterministic compiler does not consume it"
        },
    }
    return parse_workflow(document).to_dict()


if __name__ == "__main__":
    print(write_json(compiler_workflow()))
