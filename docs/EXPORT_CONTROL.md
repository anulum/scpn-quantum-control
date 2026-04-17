# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Export Control Assessment

# Export Control Assessment

**Not legal advice.** This document is the project's own
good-faith analysis of which export-control regimes apply to
`scpn-quantum-control`, performed on 2026-04-17. A consumer
planning to redistribute, embed, or combine this code with
proprietary material in a regulated jurisdiction should obtain
counsel of their own. This file is preserved to make the analysis
traceable, not to provide safe harbour.

## Regime scope

| Regime | Jurisdiction | Relevant category |
| --- | --- | --- |
| **EU Dual-Use Regulation 2021/821** | Every shipment from / via the EU | Annex I, 5A002 / 5D002 (crypto) |
| **US EAR (Export Administration Regulations)** | Any transfer of US-origin code, or mere upload to a US server viewed from abroad | ECCN 5D002 (encryption software) |
| **Swiss Goods Control Act (GKG) + Annex 2 GKV** | Shipment from Switzerland (our primary jurisdiction) | Category 5, Part 2 (crypto), mirrors EU 5A002/5D002 |
| **UK Export Control Act 2002** | UK exporters / re-exports | UK Strategic Export Control Lists, 5A002/5D002 |

We treat the **Swiss GKG / Annex 2 GKV** as authoritative for
ANULUM; the EU Dual-Use Regulation as the strictest applicable
because of near-zero-friction EU re-export; the US EAR as
applicable because GitHub and PyPI physically host the code on US
servers.

## Subpackage-by-subpackage classification

### `crypto/` — NOT CONTROLLED under current Note 3 / TSU / Publicly Available carve-outs

Modules implement the **BB84** protocol (`bb84.py`), **entanglement-based
QKD** (`entanglement_qkd.py`), **Bell-test verification** (`bell_test.py`),
**QKD parameter estimation** (`qkd_parameter.py`), **topology-authenticated
QKD** (`topology_qkd.py`), and **key-hierarchy derivation**
(`key_hierarchy.py`).

**Why not controlled:**

1. **Publicly Available / Public Domain carve-out.** Under EU
   Annex I General Technology Note / US EAR §742.15(b) (the
   "publicly available" or "encryption source code" carve-out
   post-2021 BIS rule) / Swiss Annex 2 GKV Note General Technology
   §GTN, encryption source code made available to the public at
   no cost, in source form, at a verifiable public URL, is not
   subject to the EAR / GTK / GKV except for limited-country
   restrictions (embargoed destinations).
   - We publish on PyPI + GitHub. Free. Source form. Public URL
     persists (Software Heritage + Zenodo DOI). Files include
     SPDX AGPL-3.0 licence.
2. **Note 3 Cryptanalysis Note / TSU** (US). Publicly available
   encryption source code and corresponding object code qualify
   for License Exception TSU (§740.13(e)) and the post-2021 BIS
   rule removes the need for a §742.15(b) notification for
   "publicly available" encryption source code so long as the
   BIS / Enforcement Arm is notified once per re-export path. The
   notification is a one-time email; see `ACTIONS` below.
3. **Research exemption.** Under EU 2021/821 Art. 3(2), basic
   scientific research is outside the scope of the regulation.
   The `crypto/` subpackage is documented for research and
   education (tutorials: `notebooks/05_crypto_and_entanglement.ipynb`,
   `notebooks/06_pec_error_cancellation.ipynb`). No production
   key-management functionality is claimed.

**What this means operationally.**

- Redistribution to standard destinations (EU, EEA, UK, US, CH,
  CA, AU, JP, KR, NZ, and everything not on the relevant
  country-embargo list) — permitted without a licence.
- Redistribution to destinations on the relevant embargo lists
  (e.g. US EAR §746 — Cuba, Iran, North Korea, Syria, Crimea) —
  **not permitted** regardless of the publicly-available
  carve-out. Users in those jurisdictions should not download.
- Incorporation into a **commercial** product that claims
  production-grade encryption functionality would likely move it
  out of the research exemption. The commercial licence
  (`LICENSE`) note covers the licence but explicitly NOT the
  export classification of the derivative work.

### `qec/`, `mitigation/`, `phase/`, `analysis/`, `bridge/`, `control/`, `psi_field/`, `fep/`, `qsnn/`, `hardware/`, `gauge/`, `applications/`, `benchmarks/`, `ssgf/`, `identity/`, `tcbo/`, `pgbo/`, `l16/`

**Not in Category 5 Part 2.** These are general scientific
computing / quantum-simulation modules; none implement
cryptographic primitives.

Potential alternative classification that was considered and
ruled out:

- Annex I **Category 4** (Computers / Digital computers) — does
  not apply; we are software calling a cloud backend, not
  shipping a computer.
- US ECCN **3D001 / 3D002** (Semiconductor design / quantum
  hardware control firmware) — does not apply; we do not produce
  firmware or HDL. Pulse-shaping code (`phase/pulse_shaping.py`)
  operates on abstract control envelopes, not on device-specific
  fabrication data.
- EU **Section 4A005** (intrusion software) — does not apply; no
  module has the capabilities defined there.

### `scpn_quantum_engine/` (Rust crate)

**Not controlled.** No cryptographic kernel. It exposes PyO3
bindings for numerical linear algebra, Kuramoto dynamics, Pauli
observables, and Lindblad operator construction. Same carve-outs
and absence-of-encryption conclusions as the Python subpackages
other than `crypto/`.

## Third-party dependencies (brief)

- **Qiskit** — IBM-published, widely available; IBM handles its
  own classification. We do not redistribute Qiskit.
- **NumPy / SciPy / PyO3 / rayon** — general numerical / systems
  libraries; not controlled.
- **qiskit-ibm-runtime** — cloud client for IBM Quantum Platform;
  not controlled.

## Required actions (one-time)

The publicly-available carve-out under US EAR §742.15(b) requires
a one-time emailed notification to BIS and the NSA's ENC
Encryption Request Coordinator. **Status:** pending. Owner: CEO.
Checklist and template:

```text
To:   crypt@bis.doc.gov, enc@nsa.gov
CC:   protoscience@anulum.li
Subject: Publicly available encryption source code — scpn-quantum-control

Pursuant to 15 CFR §742.15(b), this is notification that the
following source code is publicly available on the Internet:

Package:       scpn-quantum-control
Version:       <current>
URL (source):  https://github.com/anulum/scpn-quantum-control
URL (PyPI):    https://pypi.org/project/scpn-quantum-control/
URL (Zenodo):  https://doi.org/10.5281/zenodo.18821929
Licence:       AGPL-3.0-or-later (commercial licence available)
Maintainer:    Miroslav Šotek, ORCID 0009-0009-3560-0851
Contact:       protoscience@anulum.li

Crypto-relevant modules (subpackage `crypto/`):
  - scpn_quantum_control.crypto.bb84
  - scpn_quantum_control.crypto.bell_test
  - scpn_quantum_control.crypto.entanglement_qkd
  - scpn_quantum_control.crypto.key_hierarchy
  - scpn_quantum_control.crypto.qkd_parameter
  - scpn_quantum_control.crypto.topology_qkd

Each module is documented and intended for research and
educational use only. No production key-management functionality
is claimed. The code is made available at no cost, in source
form, at the URLs above.

We believe this code qualifies for the License Exception TSU
(§740.13(e)) and the publicly-available encryption source code
carve-out introduced by the 2021 BIS rule.
```

## Redistribution checklist for downstream consumers

Before **redistributing** `scpn-quantum-control` or a derivative
thereof from your jurisdiction:

- [ ] Confirm that your destination is not on your jurisdiction's
      embargo list.
- [ ] Keep the `LICENSE` file and the per-file SPDX header intact;
      the publicly-available carve-out relies on the public URL
      remaining the same source.
- [ ] If you embed the `crypto/` subpackage in a product that
      claims production-grade encryption, obtain an independent
      ECCN classification for the product. The research-use
      classification above does not transfer.
- [ ] If you fork and modify, the one-time BIS / ENC notification
      becomes yours, not ours, for your fork's public URL.

## Review cycle

Annual (April). Or sooner if:

- The `crypto/` subpackage gains a new primitive (e.g. a
  post-quantum KEM, an authenticated encryption mode).
- A published vulnerability changes our own risk posture.
- Export-control regulation changes materially (e.g. EU
  Dual-Use Regulation amendment, BIS re-tightening of §742.15(b)).

## References

- **EU Dual-Use Regulation 2021/821** —
  <https://eur-lex.europa.eu/eli/reg/2021/821/oj>
- **US EAR Part 742 §742.15** (encryption items) —
  <https://www.bis.doc.gov/index.php/policy-guidance/encryption>
- **US EAR §740.13(e)** (License Exception TSU — technology and
  software unrestricted) — <https://www.bis.doc.gov/index.php/licensing/exporting-controlled-technology>
- **Swiss Güterkontrollgesetz (GKG) + Annex 2 GKV** —
  <https://www.seco.admin.ch/seco/de/home/Aussenwirtschaftspolitik_Wirtschaftliche_Zusammenarbeit/Wirtschaftsbeziehungen/exportkontrollen-und-sanktionen.html>

Audit item **C4** in the internal gap audit
closes when the BIS / ENC notification is emailed and a receipt
timestamp is recorded here.
