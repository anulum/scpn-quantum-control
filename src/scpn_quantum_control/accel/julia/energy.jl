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
