# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the stochastic (Euler–Maruyama) networked-Kuramoto trajectory

# Seeded Euler–Maruyama of the noisy networked Kuramoto model
# ``dθ_j = (ω_j + Σ_k K_jk sin(θ_k - θ_j)) dt + √(2D) dW_j``, mirroring the Python floor and Rust tier so
# the three agree to tolerance (the coupling-force summation order differs across languages, so the tiers
# are tolerance-parity, not bit-identical). The standard-normal Wiener increments are supplied as an
# (n_steps, N) array by the caller — every tier consumes the same numpy-drawn noise — and the addition
# order ``(θ + drift) + scale·ξ`` mirrors the floor. Returns the order-parameter series and the terminal
# phases.

@inline function _kuramoto_noisy_force!(out, theta, coupling)
    n = length(theta)
    @inbounds for j in 1:n
        acc = 0.0
        theta_j = theta[j]
        for k in 1:n
            acc += coupling[j, k] * sin(theta[k] - theta_j)
        end
        out[j] = acc
    end
end

@inline function _noisy_order_parameter(theta)
    n = length(theta)
    cos_sum = 0.0
    sin_sum = 0.0
    @inbounds for j in 1:n
        cos_sum += cos(theta[j])
        sin_sum += sin(theta[j])
    end
    return sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / n
end

function kuramoto_noisy_trajectory(
    theta0::AbstractVector{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    diffusion::Real,
    dt::Real,
    noise::AbstractMatrix{<:Real},
)
    n = length(theta0)
    n_steps = size(noise, 1)
    theta = collect(Float64, theta0)
    freq = collect(Float64, omega)
    kmat = Matrix{Float64}(coupling)
    force = zeros(Float64, n)
    scale = sqrt(2.0 * diffusion * dt)
    series = zeros(Float64, n_steps)

    for step in 1:n_steps
        _kuramoto_noisy_force!(force, theta, kmat)
        @inbounds for j in 1:n
            drift = (freq[j] + force[j]) * dt
            theta[j] = theta[j] + drift + scale * noise[step, j]
        end
        series[step] = _noisy_order_parameter(theta)
    end
    return series, theta
end
