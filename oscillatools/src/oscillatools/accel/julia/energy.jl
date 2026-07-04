# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the Kuramoto interaction energy and gradient

function kuramoto_interaction_energy(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
)::Float64
    # E = −½ Σ_jk K_jk cos(θ_j − θ_k).
    n = length(theta)
    acc = 0.0
    @inbounds for j in 1:n
        tj = Float64(theta[j])
        for k in 1:n
            acc += Float64(coupling[j, k]) * cos(tj - Float64(theta[k]))
        end
    end
    return -0.5 * acc
end

function kuramoto_interaction_energy_gradient(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
)::Vector{Float64}
    # ∂E/∂θ_j = ½ Σ_k (K_jk + K_kj) sin(θ_j − θ_k).
    n = length(theta)
    out = zeros(Float64, n)
    @inbounds for j in 1:n
        acc = 0.0
        tj = Float64(theta[j])
        for k in 1:n
            acc += (Float64(coupling[j, k]) + Float64(coupling[k, j])) * sin(tj - Float64(theta[k]))
        end
        out[j] = 0.5 * acc
    end
    return out
end

function kuramoto_interaction_energy_hessian(
    theta::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
)::Matrix{Float64}
    # H_il = −½(K_il + K_li) cos(θ_i − θ_l) for l ≠ i; H_ii = −Σ_{l≠i} H_il.
    n = length(theta)
    out = zeros(Float64, n, n)
    @inbounds for i in 1:n
        diagonal = 0.0
        ti = Float64(theta[i])
        for l in 1:n
            l == i && continue
            entry =
                -0.5 *
                (Float64(coupling[i, l]) + Float64(coupling[l, i])) *
                cos(ti - Float64(theta[l]))
            out[i, l] = entry
            diagonal -= entry
        end
        out[i, i] = diagonal
    end
    return out
end
