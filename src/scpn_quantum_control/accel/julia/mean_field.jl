# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the Kuramoto mean-field force and Jacobian

function mean_field_force(theta::AbstractVector{<:Real}, coupling::Real)::Vector{Float64}
    # F_j = K (S cos θ_j − C sin θ_j), with C = <cos θ>, S = <sin θ>.
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
    kf = Float64(coupling)
    @inbounds for j in 1:n
        out[j] = kf * (s * cos(Float64(theta[j])) - c * sin(Float64(theta[j])))
    end
    return out
end

function mean_field_jacobian(theta::AbstractVector{<:Real}, coupling::Real)::Matrix{Float64}
    # J_jk = (K/N) cos(θ_j − θ_k) − K δ_jk (C cos θ_j + S sin θ_j). Symmetric, rows sum to zero.
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
    kf = Float64(coupling)
    scale = kf / n
    @inbounds for i in 1:n, j in 1:n
        out[i, j] = scale * cos(Float64(theta[i]) - Float64(theta[j]))
    end
    @inbounds for i in 1:n
        out[i, i] -= kf * (c * cos(Float64(theta[i])) + s * sin(Float64(theta[i])))
    end
    return out
end
