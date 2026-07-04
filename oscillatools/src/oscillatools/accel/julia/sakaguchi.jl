# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the Kuramoto–Sakaguchi force and Jacobian

function sakaguchi_force(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    frustration::Real,
)::Vector{Float64}
    # F_j = Σ_{k≠j} K_jk sin(θ_k − θ_j − α).
    n = length(theta)
    out = zeros(Float64, n)
    alpha = Float64(frustration)
    @inbounds for j in 1:n
        acc = 0.0
        tj = Float64(theta[j])
        for k in 1:n
            k == j && continue
            acc += Float64(coupling[j, k]) * sin(Float64(theta[k]) - tj - alpha)
        end
        out[j] = acc
    end
    return out
end

function sakaguchi_jacobian(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    frustration::Real,
)::Matrix{Float64}
    # J_jl = K_jl cos(θ_l − θ_j − α) for l ≠ j; J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j − α).
    n = length(theta)
    out = zeros(Float64, n, n)
    alpha = Float64(frustration)
    @inbounds for j in 1:n
        diagonal = 0.0
        tj = Float64(theta[j])
        for l in 1:n
            l == j && continue
            entry = Float64(coupling[j, l]) * cos(Float64(theta[l]) - tj - alpha)
            out[j, l] = entry
            diagonal -= entry
        end
        out[j, j] = diagonal
    end
    return out
end

function sakaguchi_mean_field_force(
    theta::AbstractVector{<:Real},
    coupling::Real,
    frustration::Real,
)::Vector{Float64}
    # F_j = K [(S cos θ_j − C sin θ_j) cos α − (C cos θ_j + S sin θ_j) sin α].
    n = length(theta)
    out = zeros(Float64, n)
    n == 0 && return out
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    c /= n
    s /= n
    kf = Float64(coupling)
    cos_a = cos(Float64(frustration))
    sin_a = sin(Float64(frustration))
    @inbounds for j in 1:n
        ct = cos(Float64(theta[j]))
        st = sin(Float64(theta[j]))
        in_phase = s * ct - c * st
        quadrature = c * ct + s * st
        out[j] = kf * (in_phase * cos_a - quadrature * sin_a)
    end
    return out
end

function sakaguchi_mean_field_jacobian(
    theta::AbstractVector{<:Real},
    coupling::Real,
    frustration::Real,
)::Matrix{Float64}
    # J_jl = (K/N) cos(θ_j − θ_l + α) − δ_jl K (C cos(θ_j + α) + S sin(θ_j + α)).
    n = length(theta)
    out = zeros(Float64, n, n)
    n == 0 && return out
    c = 0.0
    s = 0.0
    @inbounds for k in 1:n
        c += cos(Float64(theta[k]))
        s += sin(Float64(theta[k]))
    end
    c /= n
    s /= n
    kf = Float64(coupling)
    af = Float64(frustration)
    scale = kf / n
    @inbounds for i in 1:n, j in 1:n
        out[i, j] = scale * cos(Float64(theta[i]) - Float64(theta[j]) + af)
    end
    @inbounds for i in 1:n
        out[i, i] -= kf * (c * cos(Float64(theta[i]) + af) + s * sin(Float64(theta[i]) + af))
    end
    return out
end
