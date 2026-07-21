# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer fake-Garnet runner (.venv-iqm side)
"""Fake-Garnet side of the layout-transfer readiness gate (no QPU, no spend).

Runs in the isolated ``.venv-iqm`` (which has ``iqm-client`` but not the full
repository dependency tree, so the calibration adapter is loaded standalone
by file path). Two subcommands:

``dump-calibration``
    Extract the ``IQMFakeGarnet`` calibration into the plain-data JSON
    snapshot consumed by ``scripts/iqm_layout_transfer_harness.py prepare``.

``run``
    Execute the prepared QPY circuit matrix on the ``IQMFakeGarnet`` noisy
    simulator with the preregistered shot counts and write the counts JSON
    for ``analyse-dryrun``. This is the committed-code dry run required by
    the preregistration harness readiness gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
_ADAPTER_PATH = (
    REPO_ROOT / "src" / "scpn_quantum_control" / "hardware" / "iqm_lattice_calibration.py"
)


def _load_adapter() -> ModuleType:
    """Standalone-load the calibration adapter (no package import chain)."""
    spec = importlib.util.spec_from_file_location("iqm_lattice_calibration", _ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_garnet():  # type: ignore[no-untyped-def] — iqm types unavailable in the main venv
    """Return an ``IQMFakeGarnet`` instance (import deferred to ``.venv-iqm``)."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    return IQMFakeGarnet()


def _dump_calibration(args: argparse.Namespace) -> int:
    adapter = _load_adapter()
    backend = _fake_garnet()
    calibration = adapter.lattice_calibration_from_backend(backend)
    payload = {
        "source": "IQMFakeGarnet",
        "date": args.date,
        "calibration": calibration.to_dict(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"calibration: {out_path} "
        f"({calibration.num_qubits} qubits, {len(calibration.edges)} edges)"
    )
    return 0


def _load_qpy_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qpy_artifact_io", REPO_ROOT / "scripts" / "qpy_artifact_io.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(args: argparse.Namespace) -> int:
    from qiskit import transpile

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    circuits = _load_qpy_wrapper().reviewed_qpy_load_circuits(args.circuits)
    if len(labels) != len(circuits):
        raise ValueError(f"{len(labels)} labels but {len(circuits)} circuits")

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    main_shots = int(plan["main_shots"])
    readout_shots = int(plan["readout_shots"])

    backend = _fake_garnet()
    counts: dict[str, dict[str, int]] = {}
    for label, circuit in zip(labels, circuits):
        shots = main_shots if label.startswith("main_") else readout_shots
        # Circuits are already routed to the lattice; this only translates the
        # r/cz basis to the backend's native gate objects (layout preserved).
        native = transpile(circuit, backend=backend, optimization_level=0)
        result = backend.run(native, shots=shots).result()
        counts[label] = {key: int(value) for key, value in result.get_counts().items()}
        print(f"{label}: {shots} shots, {len(counts[label])} outcomes")

    payload = {"backend": "IQMFakeGarnet", "date": args.date, "counts": counts}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"dry-run counts: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it, returning the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump-calibration", help="dump IQMFakeGarnet calibration JSON")
    dump.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    dump.add_argument("--out", required=True, help="output calibration JSON")
    dump.set_defaults(func=_dump_calibration)

    run = sub.add_parser("run", help="run the QPY matrix on the IQMFakeGarnet simulator")
    run.add_argument("--circuits", required=True, help="QPY circuit file")
    run.add_argument("--labels", required=True, help="circuit label JSON")
    run.add_argument("--plan", required=True, help="plan artefact JSON (shot counts)")
    run.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    run.add_argument("--out", required=True, help="output counts JSON")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
