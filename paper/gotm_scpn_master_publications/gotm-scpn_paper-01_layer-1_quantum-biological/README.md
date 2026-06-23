# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# GOTM-SCPN Paper 1: Layer 1 - Quantum Biological

This package is reserved for the tuned Paper 1 extraction and validation
pipeline. Current available material is legacy/pre-tuned only and must not be
treated as processed.

## Evidence-Class Layout

- `source/` - canonical source dump and source metadata for the tuned rerun.
- `extraction/` - full-fidelity tuned extraction outputs once rerun.
- `source_validation_artifacts/` - stable source ledgers, specs, fixtures,
  promotion gates, and reconciliation outputs once generated.
- `synthesis/` - scientific synthesis derived from the tuned extraction.
- `validation_protocols/` - protocols for testing Paper 1 claims.
- `experiments/` - prepared experiment packages and run plans.
- `results/` - evidence tied to specific protocols.
- `revisions/` - proposed corrections back into the foundational text.
- `legacy_pre_tuned_extraction/` - historical extraction attempts only.

Do not promote anything from `legacy_pre_tuned_extraction/` without rerunning
the tuned methodology.
