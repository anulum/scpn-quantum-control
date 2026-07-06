# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the inertial (second-order) networked-Kuramoto trajectory

# Fixed-step RK4 of the second-order swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` on the ``(θ, v)``
# phase-space state, mirroring the Python and Rust references so the three agree to tolerance
# (the coupling-force summation order differs across languages, so the tiers are tolerance-parity,
# ~1e-11, not bit-identical). Returns the sample times, the phases and the velocities flattened
# row-major as (M+1)·N.

@inline function _kuramoto_inertial_force!(out, theta, coupling)
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

@inline function _kuramoto_inertial_field!(dtheta, dv, theta, v, freq, kmat, mass, damping, force_buf)
    _kuramoto_inertial_force!(force_buf, theta, kmat)
    n = length(theta)
    @inbounds for j in 1:n
        dtheta[j] = v[j]
        dv[j] = (freq[j] + force_buf[j] - damping * v[j]) / mass
    end
end

function kuramoto_inertial_trajectory(
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
    theta = collect(Float64, theta0)
    v = collect(Float64, velocities)
    freq = collect(Float64, omega)
    kmat = Matrix{Float64}(coupling)
    force_buf = zeros(Float64, n)
    k1t = zeros(Float64, n); k1v = zeros(Float64, n)
    k2t = zeros(Float64, n); k2v = zeros(Float64, n)
    k3t = zeros(Float64, n); k3v = zeros(Float64, n)
    k4t = zeros(Float64, n); k4v = zeros(Float64, n)
    st = zeros(Float64, n); sv = zeros(Float64, n)

    phases = collect(Float64, theta)
    vels = collect(Float64, v)
    for _ in 1:n_steps
        _kuramoto_inertial_field!(k1t, k1v, theta, v, freq, kmat, mass, damping, force_buf)
        @inbounds for i in 1:n
            st[i] = theta[i] + 0.5 * dt * k1t[i]
            sv[i] = v[i] + 0.5 * dt * k1v[i]
        end
        _kuramoto_inertial_field!(k2t, k2v, st, sv, freq, kmat, mass, damping, force_buf)
        @inbounds for i in 1:n
            st[i] = theta[i] + 0.5 * dt * k2t[i]
            sv[i] = v[i] + 0.5 * dt * k2v[i]
        end
        _kuramoto_inertial_field!(k3t, k3v, st, sv, freq, kmat, mass, damping, force_buf)
        @inbounds for i in 1:n
            st[i] = theta[i] + dt * k3t[i]
            sv[i] = v[i] + dt * k3v[i]
        end
        _kuramoto_inertial_field!(k4t, k4v, st, sv, freq, kmat, mass, damping, force_buf)
        @inbounds for i in 1:n
            theta[i] += (dt / 6.0) * (k1t[i] + 2.0 * k2t[i] + 2.0 * k3t[i] + k4t[i])
            v[i] += (dt / 6.0) * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i])
        end
        append!(phases, theta)
        append!(vels, v)
    end
    times = Float64[dt * s for s in 0:n_steps]
    return times, phases, vels
end
