# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Identity continuity analysis: attractor basin, coherence budget, entanglement, fingerprint."""

from scpn_quantum_control.identity import (
    IdentityAttractor,
    coherence_budget,
    disposition_entanglement_map,
    identity_fingerprint,
)

# Reference identity binding spec (6 disposition layers, 18 oscillators)
IDENTITY_BINDING_SPEC = {
    "name": "identity_coherence",
    "version": "0.1.0",
    "layers": [
        {
            "name": "working_style",
            "index": 0,
            "oscillator_ids": [
                "ws_action_first",
                "ws_verify_before_claim",
                "ws_commit_incremental",
            ],
            "natural_frequency": 1.2,
        },
        {
            "name": "reasoning_patterns",
            "index": 1,
            "oscillator_ids": ["rp_simplest_design", "rp_verify_audits", "rp_change_problem"],
            "natural_frequency": 1.5,
        },
        {
            "name": "relationship",
            "index": 2,
            "oscillator_ids": ["rel_autonomous", "rel_report_milestones", "rel_no_questions"],
            "natural_frequency": 0.9,
        },
        {
            "name": "aesthetics",
            "index": 3,
            "oscillator_ids": ["aes_antislop", "aes_honest_naming", "aes_terse_prose"],
            "natural_frequency": 1.1,
        },
        {
            "name": "domain_knowledge",
            "index": 4,
            "oscillator_ids": ["dk_director", "dk_neurocore", "dk_fusion"],
            "natural_frequency": 0.8,
        },
        {
            "name": "cross_project",
            "index": 5,
            "oscillator_ids": ["cp_threshold_halt", "cp_multi_signal", "cp_retrieval_scoring"],
            "natural_frequency": 1.0,
        },
    ],
    "coupling": {
        "base_strength": 0.45,
        "decay_alpha": 0.15,
    },
}

# Use a 4-oscillator subset for tractable quantum simulation
SMALL_SPEC = {
    "layers": [
        {
            "name": "working_style",
            "oscillator_ids": ["ws_action_first", "ws_verify"],
            "natural_frequency": 1.2,
        },
        {
            "name": "aesthetics",
            "oscillator_ids": ["aes_antislop", "aes_naming"],
            "natural_frequency": 1.1,
        },
    ],
    "coupling": {"base_strength": 0.45, "decay_alpha": 0.15},
}


def main():
    print("=" * 60)
    print("IDENTITY CONTINUITY ANALYSIS")
    print("=" * 60)

    # 1. Attractor Basin
    print("\n--- 1. Identity Attractor Basin (VQE) ---")
    attractor = IdentityAttractor.from_binding_spec(SMALL_SPEC, ansatz_reps=2)
    result = attractor.solve(maxiter=100, seed=42)
    print(f"Dispositions:    {result['n_dispositions']}")
    print(f"Ground energy:   {result['ground_energy']:.4f}")
    print(f"Exact energy:    {result['exact_energy']:.4f}")
    print(f"VQE error:       {result['relative_error_pct']:.2f}%")
    print(f"Robustness gap:  {result['robustness_gap']:.4f}")
    print(f"Eigenvalues:     {result['eigenvalues']}")

    if result["robustness_gap"] > 1.0:
        print("  -> Large gap: identity is robust against perturbation")
    else:
        print("  -> Small gap: identity may be fragile to missing context")

    # 2. Coherence Budget
    print("\n--- 2. Coherence Budget (Heron r2 noise model) ---")
    for n_q in [4, 8, 16]:
        budget = coherence_budget(n_q, fidelity_threshold=0.5)
        print(
            f"  {n_q:2d} qubits: max depth = {budget['max_depth']:4d}  "
            f"(F={budget['fidelity_at_max']:.3f})"
        )

    budget_4 = coherence_budget(4, fidelity_threshold=0.5)
    print("\n  Fidelity curve (4 qubits):")
    for depth, fid in sorted(budget_4["fidelity_curve"].items()):
        bar = "#" * int(fid * 30)
        print(f"    depth {depth:4d}: F={fid:.4f}  {bar}")

    # 3. Entanglement Witness
    print("\n--- 3. Disposition Entanglement (CHSH) ---")
    sv = attractor.ground_state()
    labels = ["ws_action", "ws_verify", "aes_antislop", "aes_naming"]
    emap = disposition_entanglement_map(sv, disposition_labels=labels)

    print(f"  Pairs scanned: {emap['n_pairs']}")
    print(f"  Entangled:     {emap['n_entangled']}")
    print(f"  Max S:         {emap['max_S']:.4f}  (classical bound: 2.0)")
    print(f"  Integration:   {emap['integration_metric']:.4f}  (1.0 = Tsirelson)")

    for pair in emap["pairs"]:
        tag = " *ENTANGLED*" if pair["entangled"] else ""
        print(f"    {pair['label_a']:12s} <-> {pair['label_b']:12s}  S={pair['S']:.4f}{tag}")

    # 4. Identity Fingerprint
    print("\n--- 4. Quantum Identity Fingerprint ---")
    fp = identity_fingerprint(attractor.K, attractor.omega, ansatz_reps=2, maxiter=100)
    print(f"  Fiedler value:     {fp['spectral']['fiedler']:.4f}")
    print(f"  Spectral entropy:  {fp['spectral']['spectral_entropy']:.4f}")
    print(f"  Ground energy:     {fp['ground_energy']:.4f}")
    print(f"  Commitment (hex):  {fp['commitment'][:16]}...")
    print(f"  Parameters:        {fp['n_parameters']} independent continuous values")
    print(
        f"  Security:          brute-force requires reconstructing "
        f"{fp['n_parameters']} continuous parameters"
    )

    print("\n" + "=" * 60)
    print("All identity modules exercised successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
