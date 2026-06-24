# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the Daido m-th Fourier-mode phase and gradient

function daido_mode_phase(theta::AbstractVector{<:Real}, m::Integer)::Float64
    # ψ_m = atan2(Σ sin mθ, Σ cos mθ).
    n = length(theta)
    n == 0 && return 0.0
    re = 0.0
    im = 0.0
    mf = Float64(m)
    @inbounds for k in 1:n
        re += cos(mf * Float64(theta[k]))
        im += sin(mf * Float64(theta[k]))
    end
    return atan(im, re)
end

function daido_mode_phase_gradient(theta::AbstractVector{<:Real}, m::Integer)::Vector{Float64}
    # ∂ψ_m/∂θ_j = (m / (N r_m²)) (C_m cos mθ_j + S_m sin mθ_j).
    n = length(theta)
    out = zeros(Float64, n)
    n == 0 && return out
    re = 0.0
    im = 0.0
    mf = Float64(m)
    @inbounds for k in 1:n
        re += cos(mf * Float64(theta[k]))
        im += sin(mf * Float64(theta[k]))
    end
    cos_mean = re / n
    sin_mean = im / n
    magnitude_squared = cos_mean * cos_mean + sin_mean * sin_mean
    magnitude_squared == 0.0 && return out
    scale = mf / (n * magnitude_squared)
    @inbounds for j in 1:n
        out[j] =
            scale *
            (cos_mean * cos(mf * Float64(theta[j])) + sin_mean * sin(mf * Float64(theta[j])))
    end
    return out
end

function daido_mode_phase_hessian(theta::AbstractVector{<:Real}, m::Integer)::Matrix{Float64}
    # H_ij = m² [δ_ij s_j/(N r_m) − (s_i c_j + c_i s_j)/(N² r_m²)], s_k = sin(ψ_m − m θ_k),
    # c_k = cos(ψ_m − m θ_k). Symmetric, rows sum to zero; the incoherent mode returns zeros.
    n = length(theta)
    out = zeros(Float64, n, n)
    n == 0 && return out
    re = 0.0
    im = 0.0
    mf = Float64(m)
    @inbounds for k in 1:n
        re += cos(mf * Float64(theta[k]))
        im += sin(mf * Float64(theta[k]))
    end
    cos_mean = re / n
    sin_mean = im / n
    magnitude = sqrt(cos_mean * cos_mean + sin_mean * sin_mean)
    magnitude == 0.0 && return out
    s = Vector{Float64}(undef, n)
    c = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        ck = cos(mf * Float64(theta[k]))
        sk = sin(mf * Float64(theta[k]))
        s[k] = (sin_mean * ck - cos_mean * sk) / magnitude
        c[k] = (cos_mean * ck + sin_mean * sk) / magnitude
    end
    m2 = mf * mf
    diagonal_scale = m2 / (n * magnitude)
    off_scale = m2 / (n * n * magnitude * magnitude)
    @inbounds for i in 1:n, j in 1:n
        out[i, j] = -off_scale * (s[i] * c[j] + c[i] * s[j])
    end
    @inbounds for i in 1:n
        out[i, i] += diagonal_scale * s[i]
    end
    return out
end
