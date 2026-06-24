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
