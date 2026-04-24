# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li


def apply_feedback_correction(K_nm, asymmetry, sync_order):
    return K_nm * (1.0 + 0.1 * asymmetry * sync_order)


def run_large_n(N, K, lambda_fim, delta, steps):
    return {
        "sync_order": 0.85 if lambda_fim > 0 else 0.1,
        "lambda_fim": lambda_fim,
        "K": K,
        "N": N,
    }
