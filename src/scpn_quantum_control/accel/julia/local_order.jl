# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the network-local order parameter and Jacobian

function local_order_parameter(
    theta::AbstractVector{<:Real},
    adjacency::AbstractMatrix{<:Real},
)::Vector{Float64}
    # r_j = |Σ_k A_jk e^{iθ_k}| / Σ_k A_jk.
    n = length(theta)
    out = zeros(Float64, n)
    cosv = [cos(Float64(t)) for t in theta]
    sinv = [sin(Float64(t)) for t in theta]
    @inbounds for j in 1:n
        c = 0.0
        s = 0.0
        d = 0.0
        for k in 1:n
            a = Float64(adjacency[j, k])
            c += a * cosv[k]
            s += a * sinv[k]
            d += a
        end
        if d != 0.0
            out[j] = sqrt(c * c + s * s) / d
        end
    end
    return out
end

function local_order_parameter_jacobian(
    theta::AbstractVector{<:Real},
    adjacency::AbstractMatrix{<:Real},
)::Matrix{Float64}
    # ∂r_j/∂θ_l = (A_jl / (d_j |Z_j|)) (S_j cos θ_l − C_j sin θ_l).
    n = length(theta)
    out = zeros(Float64, n, n)
    cosv = [cos(Float64(t)) for t in theta]
    sinv = [sin(Float64(t)) for t in theta]
    @inbounds for j in 1:n
        c = 0.0
        s = 0.0
        d = 0.0
        for k in 1:n
            a = Float64(adjacency[j, k])
            c += a * cosv[k]
            s += a * sinv[k]
            d += a
        end
        magnitude = sqrt(c * c + s * s)
        denominator = d * magnitude
        denominator == 0.0 && continue
        inverse = 1.0 / denominator
        for l in 1:n
            out[j, l] = Float64(adjacency[j, l]) * inverse * (s * cosv[l] - c * sinv[l])
        end
    end
    return out
end
