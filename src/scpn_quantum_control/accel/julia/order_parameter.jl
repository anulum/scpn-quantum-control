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

function order_parameter_gradient(theta::AbstractVector{<:Real})::Vector{Float64}
    # Gradient ∂r/∂θ_j of the Kuramoto order parameter r = |<exp(i θ)>|.
    # With C = <cos θ>, S = <sin θ> and r = hypot(C, S):
    #     ∂r/∂θ_j = (S cos θ_j - C sin θ_j) / (N r) = (1/N) sin(ψ - θ_j),
    # where ψ = atan2(S, C). The incoherent state r = 0 returns the zero subgradient.
    n = length(theta)
    out = Vector{Float64}(undef, n)
    n == 0 && return out
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    c /= n
    s /= n
    r = sqrt(c * c + s * s)
    if r == 0.0
        fill!(out, 0.0)
        return out
    end
    scale = 1.0 / (n * r)
    @inbounds for j in 1:n
        out[j] = (s * cos(Float64(theta[j])) - c * sin(Float64(theta[j]))) * scale
    end
    return out
end

function order_parameter_hessian(theta::AbstractVector{<:Real})::Matrix{Float64}
    # Hessian ∂²r/∂θ_i∂θ_j of the Kuramoto order parameter r = |<exp(i θ)>|.
    # With C = <cos θ>, S = <sin θ>, r = hypot(C, S) and alignment
    # a_j = cos(ψ − θ_j) = (C cos θ_j + S sin θ_j) / r:
    #     H_ij = a_i a_j / (N² r) − δ_ij a_j / N.
    # Symmetric, row sums zero; the incoherent state r = 0 returns the zero matrix.
    n = length(theta)
    out = Matrix{Float64}(undef, n, n)
    n == 0 && return out
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    c /= n
    s /= n
    r = sqrt(c * c + s * s)
    if r == 0.0
        fill!(out, 0.0)
        return out
    end
    aligned = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        aligned[j] = (c * cos(Float64(theta[j])) + s * sin(Float64(theta[j]))) / r
    end
    scale = 1.0 / (n * n * r)
    @inbounds for i in 1:n, j in 1:n
        out[i, j] = aligned[i] * aligned[j] * scale
    end
    @inbounds for i in 1:n
        out[i, i] -= aligned[i] / n
    end
    return out
end

function mean_phase(theta::AbstractVector{<:Real})::Float64
    # Circular mean phase ψ = atan2(<sin θ>, <cos θ>). The 1/N scaling cancels inside
    # atan2, so the raw sums suffice. Empty input and the incoherent state map to 0.0.
    n = length(theta)
    n == 0 && return 0.0
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    return atan(s, c)
end

function mean_phase_gradient(theta::AbstractVector{<:Real})::Vector{Float64}
    # Gradient ∂ψ/∂θ_j = (C cos θ_j + S sin θ_j) / (N r²) with C = <cos θ>, S = <sin θ>,
    # r = hypot(C, S). The components sum to one; the incoherent state r = 0 returns zeros.
    n = length(theta)
    out = Vector{Float64}(undef, n)
    n == 0 && return out
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    c /= n
    s /= n
    r = sqrt(c * c + s * s)
    if r == 0.0
        fill!(out, 0.0)
        return out
    end
    scale = 1.0 / (n * r * r)
    @inbounds for j in 1:n
        out[j] = (c * cos(Float64(theta[j])) + s * sin(Float64(theta[j]))) * scale
    end
    return out
end
