# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the symplectic inertial networked-Kuramoto trajectory

# Damped velocity-Verlet (leapfrog) of the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` with Strang
# splitting of the linear damping, mirroring the Python and Rust references (tolerance-parity, ~1e-11,
# the coupling-force summation order being the only cross-language difference). Returns the sample
# times, phases and velocities flattened row-major as (M+1)·N.

@inline function _kuramoto_symplectic_force!(out, theta, coupling)
    n = length(theta)
    @inbounds for j in 1:n
        acc = 0.0
        tj = theta[j]
        for k in 1:n
            acc += coupling[j, k] * sin(theta[k] - tj)
        end
        out[j] = acc
    end
end

function kuramoto_symplectic_inertial_trajectory(
    theta0::AbstractVector{<:Real},
    velocities::AbstractVector{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    mass::Real,
    damping::Real,
    dt::Real,
    n_steps::Integer,
)
    n = length(theta0)
    position = collect(Float64, theta0)
    momentum = collect(Float64, velocities)
    freq = collect(Float64, omega)
    kmat = Matrix{Float64}(coupling)
    force_buf = zeros(Float64, n)
    decay = exp(-damping * dt / (2.0 * mass))

    phases = collect(Float64, position)
    vels = collect(Float64, momentum)
    for _ in 1:n_steps
        @inbounds for j in 1:n
            momentum[j] *= decay
        end
        _kuramoto_symplectic_force!(force_buf, position, kmat)
        @inbounds for j in 1:n
            momentum[j] += 0.5 * dt * (freq[j] + force_buf[j]) / mass
        end
        @inbounds for j in 1:n
            position[j] += dt * momentum[j]
        end
        _kuramoto_symplectic_force!(force_buf, position, kmat)
        @inbounds for j in 1:n
            momentum[j] += 0.5 * dt * (freq[j] + force_buf[j]) / mass
        end
        @inbounds for j in 1:n
            momentum[j] *= decay
        end
        append!(phases, position)
        append!(vels, momentum)
    end
    times = Float64[dt * s for s in 0:n_steps]
    return times, phases, vels
end
