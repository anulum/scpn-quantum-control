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
