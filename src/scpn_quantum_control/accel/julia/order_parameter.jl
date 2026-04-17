# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for order parameter

"""
Kuramoto order parameter on Julia's BLAS-backed dense kernel.

R = |<exp(i θ)>| over N oscillators.

This file is loaded on demand by `scpn_quantum_control.accel.julia`
via juliacall. The Julia process reuses a single Main module across
calls so the JIT warm-up cost is paid once per Python process.
"""

function order_parameter(theta::AbstractVector{<:Real})::Float64
    # <exp(i θ)> — mean of complex unit-modulus vector. Julia's
    # sum / length uses SIMD + BLAS where applicable.
    n = length(theta)
    if n == 0
        return 0.0
    end
    z = zero(ComplexF64)
    @inbounds for k in 1:n
        z += cis(Float64(theta[k]))    # cis(x) = exp(i*x), BLAS-compatible
    end
    return abs(z) / n
end

function order_parameters_batch(theta_batch::AbstractMatrix{<:Real})::Vector{Float64}
    # Batch of T time-slices × N oscillators. Returns R(t_k) for k = 1..T.
    T = size(theta_batch, 1)
    out = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        out[t] = order_parameter(@view theta_batch[t, :])
    end
    return out
end
