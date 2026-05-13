#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM layout-pinned DLA minimal runner
"""Prepare or execute a layout-pinned IQM DLA/parity minimal tier.

The default mode is a no-submit dry run. Real IQM execution requires
``--execute`` and reads the token from the private vault. Raw job identifiers
are written only to ignored ``docs/internal`` artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "iqm_fake_transpile_payload.py"
VAULT_PATH = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
PUBLIC_DIR = REPO_ROOT / "data" / "iqm_paper_replication"
PRIVATE_DIR = REPO_ROOT / "docs" / "internal" / "iqm_runs"
DEFAULT_LAYOUT = (9, 4, 3, 8)
DEPTHS = (4, 6, 10)
SECTORS = {"even": "0011", "odd": "0001"}


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("iqm_fake_transpile_payload", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM circuit helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_layout(value: str) -> tuple[int, int, int, int]:
    """Parse a four-physical-qubit layout string."""
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 4:
        raise ValueError("layout must contain exactly four comma-separated physical qubits")
    if len(set(parts)) != 4:
        raise ValueError("layout physical qubits must be unique")
    if any(part < 0 for part in parts):
        raise ValueError("layout physical qubits must be non-negative")
    return parts  # type: ignore[return-value]


def build_plan(*, layout: tuple[int, int, int, int], shots: int) -> list[dict[str, Any]]:
    """Build the layout-pinned repeated minimal plan."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        for sector, initial in SECTORS.items():
            rows.append(
                {
                    "tier": "dla_parity_minimal_layout_pinned_repeat",
                    "circuit_name": f"iqm_dla_pinned_n4_d{depth}_{sector}",
                    "kind": "dla_parity",
                    "shots": shots,
                    "requested_initial_layout": list(layout),
                    "meta": {
                        "experiment": "A_dla_parity_n4",
                        "n_qubits": 4,
                        "depth": depth,
                        "sector": sector,
                        "initial": initial,
                        "t_step": 0.3,
                        "paper_source": "phase1_dla_parity",
                    },
                }
            )
    for sector, initial in SECTORS.items():
        rows.append(
            {
                "tier": "dla_readout_layout_pinned_baseline",
                "circuit_name": f"iqm_readout_pinned_{sector}_{initial}",
                "kind": "readout_baseline",
                "shots": shots,
                "requested_initial_layout": list(layout),
                "meta": {
                    "experiment": "C_readout_baseline",
                    "n_qubits": 4,
                    "sector": sector,
                    "initial": initial,
                    "paper_source": "phase1_dla_parity",
                },
            }
        )
    return rows


def bit_parity(bitstring: str) -> int:
    """Return computational-basis parity."""
    return bitstring.replace(" ", "").count("1") % 2


def normalise_key(key: str, width: int) -> str:
    """Normalise Qiskit count keys to zero-padded binary strings."""
    clean = key.replace(" ", "")
    if clean.startswith("0x"):
        return format(int(clean, 16), f"0{width}b")
    return clean.zfill(width)


def analyse_counts(counts: dict[str, int], *, initial: str, n_qubits: int) -> dict[str, Any]:
    """Compute parity leakage and initial-state retention from raw counts."""
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("empty count dictionary")
    expected_parity = bit_parity(initial)
    normalised = {normalise_key(key, n_qubits): int(value) for key, value in counts.items()}
    in_sector = sum(
        value for key, value in normalised.items() if bit_parity(key) == expected_parity
    )
    leakage = total - in_sector
    return {
        "counts": dict(sorted(normalised.items())),
        "total_shots": total,
        "expected_parity": expected_parity,
        "in_sector_counts": in_sector,
        "leakage_counts": leakage,
        "sector_survival_fraction": in_sector / total,
        "parity_leakage": leakage / total,
        "initial_state_retention": normalised.get(initial[::-1], 0) / total,
    }


def _build_circuit(helper: Any, row: dict[str, Any]) -> QuantumCircuit:
    circuit = helper._build_circuit({"circuit_name": row["circuit_name"], "meta": row["meta"]})
    circuit.name = str(row["circuit_name"])
    return circuit


def _load_iqm_credentials() -> tuple[str, str]:
    text = VAULT_PATH.read_text(encoding="utf-8")
    in_section = False
    url = token = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## IQM Resonance"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        if line.lower().startswith("- url:"):
            url = line.split(":", 1)[1].strip()
        if line.lower().startswith("- token:"):
            token = line.split(":", 1)[1].strip()
    if not url or not token:
        raise RuntimeError("missing IQM Resonance URL or token in vault")
    return url, token


def _resolve_backend(*, execute: bool, fake_backend: str, quantum_computer: str) -> Any:
    if execute:
        from iqm.qiskit_iqm.iqm_provider import IQMProvider

        url, token = _load_iqm_credentials()
        return IQMProvider(url, quantum_computer=quantum_computer, token=token).get_backend()

    module_name, class_name = {
        "garnet": ("iqm.qiskit_iqm.fake_backends.fake_garnet", "IQMFakeGarnet"),
        "deneb": ("iqm.qiskit_iqm.fake_backends.fake_deneb", "IQMFakeDeneb"),
        "apollo": ("iqm.qiskit_iqm.fake_backends.fake_apollo", "IQMFakeApollo"),
        "aphrodite": ("iqm.qiskit_iqm.fake_backends.fake_aphrodite", "IQMFakeAphrodite"),
        "adonis": ("iqm.qiskit_iqm.fake_backends.fake_adonis", "IQMFakeAdonis"),
    }[fake_backend]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)()


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if name:
        return str(name)
    return type(backend).__name__


def _job_id(job: Any) -> str:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        return str(job_id())
    if job_id:
        return str(job_id)
    return "iqm_job_id_unavailable"


def _counts(result: Any) -> dict[str, int]:
    raw = result.get_counts()
    if isinstance(raw, list):
        if len(raw) != 1:
            raise RuntimeError("single-circuit IQM run returned multiple count maps")
        raw = raw[0]
    return {str(key): int(value) for key, value in raw.items()}


def run_plan(
    *,
    execute: bool,
    layout: tuple[int, int, int, int],
    shots: int,
    fake_backend: str,
    quantum_computer: str,
) -> dict[str, Any]:
    """Dry-run or execute the layout-pinned plan."""
    helper = _load_helper()
    backend = _resolve_backend(
        execute=execute, fake_backend=fake_backend, quantum_computer=quantum_computer
    )
    rows = build_plan(layout=layout, shots=shots)
    records = []
    started_all = time.time()
    for index, row in enumerate(rows, start=1):
        circuit = _build_circuit(helper, row)
        isa = transpile(
            circuit, backend=backend, initial_layout=list(layout), optimization_level=1
        )
        record = {
            **row,
            "backend_name": _backend_name(backend),
            "transpiled_depth": isa.depth(),
            "transpiled_size": isa.size(),
            "transpiled_ops": {str(key): int(value) for key, value in isa.count_ops().items()},
            "status": "planned" if not execute else "submitted",
        }
        if execute:
            started = time.time()
            job = backend.run([isa], shots=shots)
            raw_job_id = _job_id(job)
            job_hash = hashlib.sha256(raw_job_id.encode("utf-8")).hexdigest()
            try:
                result = job.result(timeout=900)
            except Exception as exc:
                if type(exc).__name__ != "APITimeoutError":
                    raise
                cancel_status = "not_attempted"
                try:
                    cancel_status = "cancelled" if job.cancel() else "not_cancelled"
                except Exception as cancel_exc:
                    cancel_status = f"cancel_error:{type(cancel_exc).__name__}"
                record.update(
                    {
                        "job_id": raw_job_id,
                        "job_id_sha256": job_hash,
                        "wall_time_s": time.time() - started,
                        "status": "timeout_cancelled",
                        "timeout_s": 900,
                        "cancel_status": cancel_status,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                print(
                    f"timeout {index}/{len(rows)} {row['circuit_name']} "
                    f"job_hash={job_hash} cancel_status={cancel_status}"
                )
                records.append(record)
                break
            wall = time.time() - started
            raw_counts = _counts(result)
            stats = analyse_counts(
                raw_counts,
                initial=str(row["meta"]["initial"]),
                n_qubits=int(row["meta"]["n_qubits"]),
            )
            record.update(
                {
                    **stats,
                    "job_id": raw_job_id,
                    "job_id_sha256": job_hash,
                    "wall_time_s": wall,
                    "status": "completed",
                }
            )
            print(
                f"completed {index}/{len(rows)} {row['circuit_name']} "
                f"leakage={record['parity_leakage']:.6f} "
                f"job_hash={record['job_id_sha256']}"
            )
        else:
            print(
                f"planned {index}/{len(rows)} {row['circuit_name']} "
                f"depth={record['transpiled_depth']} ops={record['transpiled_ops']}"
            )
        records.append(record)

    return {
        "schema": (
            "scpn_iqm_dla_layout_pinned_repeat_v1_private"
            if execute
            else "scpn_iqm_dla_layout_pinned_repeat_v1_plan"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "iqm",
        "platform": "IQM Resonance" if execute else f"IQM fake {fake_backend}",
        "quantum_computer": quantum_computer if execute else None,
        "fake_backend": None if execute else fake_backend,
        "tier": "dla_parity_minimal_layout_pinned_repeat",
        "execute": execute,
        "requested_initial_layout": list(layout),
        "total_circuits": len(records),
        "total_shots": sum(int(row["shots"]) for row in records),
        "total_wall_time_s": time.time() - started_all,
        "records": records,
        "claim_boundary": (
            "Layout-pinned repeated minimal tier controls the first IQM run's automatic-layout "
            "confound, but still requires analysis before manuscript claims are upgraded."
        ),
    }


def _public_copy(payload: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(payload))
    for record in clean["records"]:
        record.pop("job_id", None)
    if clean.get("platform") == "IQM Resonance":
        clean["schema"] = "scpn_iqm_dla_layout_pinned_repeat_v1_sanitized"
    return clean


def _write_outputs(payload: dict[str, Any], *, output_dir: Path) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "executed" if payload["execute"] else "plan"
    layout_slug = "q" + "-".join(str(qubit) for qubit in payload["requested_initial_layout"])
    public_path = (
        output_dir / f"iqm_dla_layout_pinned_repeat_{layout_slug}_2026-05-13_{suffix}.json"
    )
    public_path.write_text(
        json.dumps(_public_copy(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    private_path = None
    if payload["execute"]:
        private_path = (
            PRIVATE_DIR / f"iqm_dla_layout_pinned_repeat_{layout_slug}_2026-05-13_private.json"
        )
        private_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return public_path, private_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="submit real IQM jobs")
    parser.add_argument("--layout", default=",".join(map(str, DEFAULT_LAYOUT)))
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--fake-backend", default="garnet")
    parser.add_argument("--quantum-computer", default="garnet")
    parser.add_argument("--output-dir", type=Path, default=PUBLIC_DIR)
    args = parser.parse_args()

    payload = run_plan(
        execute=args.execute,
        layout=parse_layout(args.layout),
        shots=args.shots,
        fake_backend=args.fake_backend,
        quantum_computer=args.quantum_computer,
    )
    public_path, private_path = _write_outputs(payload, output_dir=args.output_dir)
    print(f"wrote_public={public_path}")
    if private_path is not None:
        print(f"wrote_private={private_path}")
    print(f"total_circuits={payload['total_circuits']}")
    print(f"total_shots={payload['total_shots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
