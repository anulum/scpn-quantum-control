# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive dispatch command line interface
"""The ``scpn-studio-run`` executive dispatch command line interface.

Drives the executive action spine (:mod:`scpn_quantum_control.studio.executive`)
from the command line, so the studio's verbs are runnable from a shell or the
SCPN-STUDIO hub, not only from Python. One invocation dispatches one verb::

    scpn-studio-run compile --action-id compile-3node --params '{"K_nm": ...}'
    scpn-studio-run simulate --action-id sim-3node --params-file request.json --preview
    scpn-studio-run execute --action-id deploy --params-file deploy.json --approve

The sealed :class:`~scpn_quantum_control.studio.executive.ExecutiveRecord` (or
the inspectable plan under ``--preview``) is printed to stdout as JSON; the
generated reproduction script is additionally written to ``--script-dir`` when
given. Exit codes are scriptable: ``0`` succeeded (or previewed), ``1`` the
action failed, ``2`` a request/parameter error, ``3`` the action was gated
(a live-hardware or certified verb without ``--approve`` — the fail-closed
deploy contract).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from .executive import (
    ActionRegistry,
    ExecutiveRecord,
    ExecutiveRequest,
    preview_action,
    run_action,
)
from .executive_analyse import AnalyseActionHandler
from .executive_compile import CompileActionHandler
from .executive_differentiate import DifferentiateActionHandler
from .executive_execute import ExecuteActionHandler
from .executive_simulate import SimulateActionHandler

EXIT_SUCCEEDED: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_REQUEST_ERROR: Final[int] = 2
EXIT_GATED: Final[int] = 3


def build_default_registry() -> ActionRegistry:
    """Return a registry with every shipped executive handler registered.

    Returns
    -------
    ActionRegistry
        A fresh registry carrying the ``analyse``, ``compile``,
        ``differentiate``, ``execute``, and ``simulate`` handlers.
    """
    registry = ActionRegistry()
    registry.register(AnalyseActionHandler())
    registry.register(CompileActionHandler())
    registry.register(DifferentiateActionHandler())
    registry.register(ExecuteActionHandler())
    registry.register(SimulateActionHandler())
    return registry


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scpn-studio-run",
        description=(
            "Run one SCPN-QUANTUM-CONTROL studio verb executively and print the "
            "sealed record as JSON."
        ),
    )
    parser.add_argument("verb", help="the studio verb to run (e.g. compile, simulate)")
    parser.add_argument(
        "--action-id",
        required=True,
        help="stable identifier for this action (used in the sealed record)",
    )
    params = parser.add_mutually_exclusive_group(required=True)
    params.add_argument("--params", help="verb parameters as an inline JSON object")
    params.add_argument("--params-file", help="path to a JSON file with the verb parameters")
    parser.add_argument("--backend", help="requested backend (defaults to the handler's choice)")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="explicitly approve a gated (live-hardware or certified) verb",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="print the inspectable plan without executing",
    )
    parser.add_argument(
        "--script-dir",
        help="directory to write the generated reproduction script into on success",
    )
    return parser.parse_args(list(argv))


def _load_parameters(inline: str | None, file_path: str | None) -> dict[str, Any]:
    if inline is not None:
        raw = inline
    else:
        assert file_path is not None  # argparse enforces exactly one source  # nosec B101
        try:
            raw = Path(file_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read --params-file {file_path!r}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"parameters are not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("parameters must be a JSON object")
    return payload


def _write_script(record: ExecutiveRecord, script_dir: str) -> Path:
    script = record.script
    assert script is not None  # only called on succeeded records  # nosec B101
    directory = Path(script_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / script.filename
    target.write_text(script.source, encoding="utf-8")
    return target


def run(argv: Sequence[str] | None = None) -> int:
    """Dispatch one executive action from command line arguments.

    Parameters
    ----------
    argv : Sequence of str or None, optional
        Arguments to parse; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` succeeded (or previewed), ``1`` failed, ``2`` request error,
        ``3`` gated without approval.
    """
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    registry = build_default_registry()
    try:
        parameters = _load_parameters(ns.params, ns.params_file)
        request = ExecutiveRequest(
            verb=ns.verb,
            action_id=ns.action_id,
            parameters=parameters,
            backend=ns.backend,
            approved=ns.approve,
        )
        if ns.preview:
            plan = preview_action(request, registry=registry)
            print(json.dumps(plan.to_dict(), sort_keys=True, indent=2))
            return EXIT_SUCCEEDED
        record = run_action(request, registry=registry)
    except (KeyError, ValueError) as exc:
        print(f"scpn-studio-run: error: {exc}", file=sys.stderr)
        return EXIT_REQUEST_ERROR

    print(json.dumps(record.to_dict(), sort_keys=True, indent=2))
    if record.result.status == "gated":
        print(f"scpn-studio-run: gated: {record.result.error}", file=sys.stderr)
        return EXIT_GATED
    if record.result.status == "failed":
        print(f"scpn-studio-run: failed: {record.result.error}", file=sys.stderr)
        return EXIT_FAILED
    if ns.script_dir is not None:
        target = _write_script(record, ns.script_dir)
        print(f"scpn-studio-run: wrote script {target}", file=sys.stderr)
    return EXIT_SUCCEEDED


def main() -> None:
    """Console entry point for ``scpn-studio-run``."""
    raise SystemExit(run())


__all__ = [
    "EXIT_FAILED",
    "EXIT_GATED",
    "EXIT_REQUEST_ERROR",
    "EXIT_SUCCEEDED",
    "build_default_registry",
    "main",
    "run",
]
