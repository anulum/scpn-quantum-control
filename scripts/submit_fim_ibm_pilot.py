#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- FIM IBM pilot submission
"""Submit the SCPN/FIM n=4 pilot to IBM after live-readiness approval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_fim_ibm_live_readiness import (  # noqa: E402
    DATE,
    OUT_DIR,
    _build_circuit,
    _load_credentials,
)

from scpn_quantum_control.hardware.runner import _extract_counts  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def submit(
    protocol_path: Path,
    backend_name: str,
    channel: str,
    vault: Path,
    vault_token_kind: str,
    shots: int,
    timeout_s: float,
) -> dict[str, object]:
    """Submit the FIM IBM protocol and persist pending and completed artefacts."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    token, instance, credential_source = _load_credentials(vault, channel, vault_token_kind)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    is_repeated_followup = "repeated_followup" in protocol_path.stem
    experiment = "scpn_fim_ibm_repeated_followup" if is_repeated_followup else "scpn_fim_ibm_pilot"
    output_prefix = "fim_ibm_repeated_followup" if is_repeated_followup else "fim_ibm_pilot"
    service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    backend = service.backend(backend_name)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=2)

    circuits = []
    metadata_rows = []
    for index, item in enumerate(protocol["rows"]):
        circuit = _build_circuit(item)
        circuit.name = f"fim_submit_{index:03d}_{item['protocol_arm']}"
        isa = pass_manager.run(circuit)
        circuits.append(isa)
        ops = {gate: int(count) for gate, count in isa.count_ops().items()}
        metadata_rows.append(
            {
                **item,
                "circuit_index": index,
                "circuit_name": circuit.name,
                "live_transpiled_depth": isa.depth(),
                "live_transpiled_size": isa.size(),
                "live_transpiled_ops": ops,
            }
        )

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    submitted_at = datetime.now(timezone.utc).isoformat()
    job = sampler.run(circuits)
    job_id = job.job_id()
    pending = {
        "schema": "scpn_fim_ibm_pilot_submission_v1",
        "date": DATE,
        "experiment": experiment,
        "backend": backend_name,
        "channel": channel,
        "credential_source": credential_source,
        "job_id": job_id,
        "submitted_at_utc": submitted_at,
        "status": "submitted_waiting_for_result",
        "total_circuits": len(circuits),
        "shots_per_circuit": shots,
        "total_shots": len(circuits) * shots,
        "metadata_rows": metadata_rows,
    }
    pending_path = OUT_DIR / f"{output_prefix}_pending_{DATE}_{job_id}.json"
    _write_json(pending_path, pending)
    print(f"job_id={job_id}")
    print(f"pending_file={pending_path}")

    start = time.time()
    result = job.result(timeout=timeout_s)
    wall_time_s = time.time() - start
    result_rows = []
    for index, pub_result in enumerate(result):
        result_rows.append(
            {
                "circuit_index": index,
                "metadata": metadata_rows[index],
                "counts": _extract_counts(pub_result),
            }
        )
    completed = {
        **pending,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wait_wall_time_s": wall_time_s,
        "result_rows": result_rows,
    }
    completed_path = OUT_DIR / f"{output_prefix}_raw_counts_{DATE}_{job_id}.json"
    _write_json(completed_path, completed)
    print(f"completed_file={completed_path}")
    print(f"wait_wall_time_s={wall_time_s:.1f}")
    return completed


def main() -> int:
    """Parse options and submit the FIM IBM pilot or follow-up protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--channel", default="ibm_cloud")
    parser.add_argument(
        "--vault-token-kind", choices=["auto", "api_key", "api_token"], default="api_key"
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=OUT_DIR / f"fim_ibm_candidate_protocol_{DATE}.json",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    ns = parser.parse_args()
    submit(
        ns.protocol,
        ns.backend,
        ns.channel,
        ns.vault,
        ns.vault_token_kind,
        ns.shots,
        ns.timeout_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
