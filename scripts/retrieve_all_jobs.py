#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Result Reprocessor

"""
Job Retrieval Script — Pull real IBM bitstrings from queued jobs
and update all result JSON files with actual measurement counts.
"""

import asyncio
import json
import os
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

from scpn_quantum_control.analysis import (
    DLAParityWitness,
    IntegratedInformationPhi,
    SyncOrderParameter,
)


def _extract_counts(pub_result) -> dict[str, int]:
    data = pub_result.data
    for register_name in ("meas", "c", "cr", "c0", "c1"):
        register = getattr(data, register_name, None)
        if register is not None and hasattr(register, "get_counts"):
            return register.get_counts()
    return {}


async def retrieve_all_jobs():
    token = os.environ.get("SCPN_IBM_TOKEN")
    if not token:
        print("SCPN_IBM_TOKEN environment variable is not set.")
        return

    instance = os.environ.get("SCPN_IBM_INSTANCE")
    service_kwargs = {"channel": "ibm_cloud", "token": token}
    if instance:
        service_kwargs["instance"] = instance
    service = QiskitRuntimeService(**service_kwargs)

    results_dirs = [
        Path("results"),
        Path("results/frontier_campaign"),
        Path("results/sophisticated_campaign"),
        Path("results/primary_campaign"),
    ]

    json_files = []
    for d in results_dirs:
        if d.exists():
            json_files.extend(list(d.glob("*.json")))

    print(f"Found {len(json_files)} result JSON files. Starting retrieval.\n")

    updated_count = 0

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            is_list = isinstance(data, list)
            entries = data if is_list else [data]

            modified = False

            for entry in entries:
                job_id = entry.get("job_id")
                if not job_id or job_id.startswith("simulated") or job_id == "local_simulated":
                    continue

                print(f"Retrieving job {job_id} from {json_file.name}...")

                try:
                    job = service.job(job_id)

                    if job.status().name != "DONE":
                        print(f"   Still {job.status().name}. Skipping for now.")
                        continue

                    result = job.result()
                    counts = _extract_counts(result[0])

                    real_results = {}
                    observables = [
                        DLAParityWitness(),
                        SyncOrderParameter(),
                        IntegratedInformationPhi(),
                    ]
                    for obs in observables:
                        real_results.update(obs(counts=counts))

                    entry.update(real_results)
                    entry["real_counts"] = counts
                    entry["status"] = "DONE_REAL"
                    modified = True

                    print(f"   Success: {len(counts)} measurement outcomes retrieved")

                except Exception as e:
                    print(f"   Failed to retrieve {job_id}: {e}")

            if modified:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                updated_count += 1
                print(f"Updated {json_file.name} with real IBM data\n")

        except Exception as e:
            print(f"Could not process {json_file.name}: {e}")

    print(f"\nRetrieval complete. Updated {updated_count} files with real IBM bitstrings.")


if __name__ == "__main__":
    asyncio.run(retrieve_all_jobs())
