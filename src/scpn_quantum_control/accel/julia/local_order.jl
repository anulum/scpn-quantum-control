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

function local_mean_phase(
    theta::AbstractVector{<:Real},
    adjacency::AbstractMatrix{<:Real},
)::Vector{Float64}
    # ψ_j = atan2(Σ_k A_jk sin θ_k, Σ_k A_jk cos θ_k). |Z_j| = 0 yields ψ_j = 0.
    n = length(theta)
    out = zeros(Float64, n)
    n == 0 && return out
    cosv = cos.(Float64.(theta))
    sinv = sin.(Float64.(theta))
    @inbounds for j in 1:n
        c = 0.0
        s = 0.0
        for k in 1:n
            a = Float64(adjacency[j, k])
            c += a * cosv[k]
            s += a * sinv[k]
        end
        if c * c + s * s != 0.0
            out[j] = atan(s, c)
        end
    end
    return out
end

function local_mean_phase_jacobian(
    theta::AbstractVector{<:Real},
    adjacency::AbstractMatrix{<:Real},
)::Matrix{Float64}
    # ∂ψ_j/∂θ_l = A_jl (C_j cos θ_l + S_j sin θ_l) / |Z_j|². |Z_j| = 0 yields a zero row.
    n = length(theta)
    out = zeros(Float64, n, n)
    n == 0 && return out
    cosv = cos.(Float64.(theta))
    sinv = sin.(Float64.(theta))
    @inbounds for j in 1:n
        c = 0.0
        s = 0.0
        for k in 1:n
            a = Float64(adjacency[j, k])
            c += a * cosv[k]
            s += a * sinv[k]
        end
        magnitude_squared = c * c + s * s
        magnitude_squared == 0.0 && continue
        inverse = 1.0 / magnitude_squared
        for l in 1:n
            out[j, l] = Float64(adjacency[j, l]) * inverse * (c * cosv[l] + s * sinv[l])
        end
    end
    return out
end
