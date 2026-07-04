# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the networked Kuramoto force and Jacobian

function networked_kuramoto_force(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
)::Vector{Float64}
    # F_j = Σ_k K_jk sin(θ_k − θ_j).
    n = length(theta)
    out = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        acc = 0.0
        tj = Float64(theta[j])
        for k in 1:n
            acc += Float64(coupling[j, k]) * sin(Float64(theta[k]) - tj)
        end
        out[j] = acc
    end
    return out
end

function networked_kuramoto_jacobian(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
)::Matrix{Float64}
    # J_jl = K_jl cos(θ_l − θ_j) for l ≠ j; J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j).
    n = length(theta)
    out = zeros(Float64, n, n)
    @inbounds for j in 1:n
        diagonal = 0.0
        tj = Float64(theta[j])
        for l in 1:n
            l == j && continue
            entry = Float64(coupling[j, l]) * cos(Float64(theta[l]) - tj)
            out[j, l] = entry
            diagonal -= entry
        end
        out[j, j] = diagonal
    end
    return out
end
