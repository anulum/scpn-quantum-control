<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — quantum Kuramoto audit usage -->

# Quantum Kuramoto audit usage

Generate an import audit, boundary review and API contract from explicitly
selected inputs. Each review and contract records the actual input path and
SHA-256. The date in the output filename identifies the original report format;
it does not establish when the selected source was qualified.

Run from the repository root with the project dependencies installed. Select an
output directory outside the archived `data/s6_quantum_kuramoto_split` reports:

```bash
python scripts/audit_quantum_kuramoto_split.py \
  --source-root src/scpn_quantum_control \
  --out-dir /tmp/quantum-kuramoto-audit \
  --doc-path /tmp/quantum-kuramoto-audit/audit.md
python scripts/export_quantum_kuramoto_boundary_review.py \
  --audit-path /tmp/quantum-kuramoto-audit/quantum_kuramoto_split_audit_2026-05-07.json \
  --out-dir /tmp/quantum-kuramoto-audit \
  --doc-path /tmp/quantum-kuramoto-audit/review.md
python scripts/export_quantum_kuramoto_api_contract.py \
  --review-path /tmp/quantum-kuramoto-audit/quantum_kuramoto_boundary_review_2026-05-07.json \
  --out-dir /tmp/quantum-kuramoto-audit \
  --doc-path /tmp/quantum-kuramoto-audit/contract.md
```

The audit reads the selected physical Python sources, including the root Rust
accelerator entry point. Relative imports are resolved against their containing
module. Existing accelerator compatibility aliases are checked against their
importable canonical provider and record that provider's physical source digest.
This requires the canonical provider to be available in the Python environment.
AST classifications describe source dependencies; they do not establish runtime
correctness, scientific fidelity or package publication readiness.

For Python consumers, `audit_module_source(module, path)` audits one physical
module and `build_split_audit(source_root=...)` collects the source inventory.
`write_split_audit(payload, json_path=..., doc_path=...)` writes JSON and Markdown
and returns their byte digests. `build_boundary_review(audit,
source_audit=path)` and `build_api_contract(review, source_review=path)` bind the
supplied objects to the declared files. A file whose JSON content differs from
the supplied object raises `ValueError`. An in-memory object supplied without a
source path remains explicitly unbound. Omitting the object uses the archived
default input and records its provenance.

The reports retain conservative package and publication holds. Generating them
does not create or authorize a new package. The dated archived reports remain
historical inputs unless explicitly selected as output destinations.
