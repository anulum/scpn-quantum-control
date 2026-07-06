# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the time-delayed (method-of-steps) networked-Kuramoto trajectory

# Delay-aware fixed-step RK4 of the time-delayed Kuramoto model
# ``θ̇_j(t) = ω_j + Σ_k K_jk sin(θ_k(t-τ) - θ_j(t))``, mirroring the Python floor and Rust tier so the
# three agree to tolerance (the coupling-force summation order differs across languages, so the tiers
# are tolerance-parity, ~1e-11, not bit-identical). The running phase grid (flattened row-major)
# doubles as the history buffer; the delayed argument at a sub-stage grid position is read by linear
# interpolation. Grid positions are 0-based to match the Python reference. Returns the sample times
# and the phases flattened row-major as (M+1)·N.

@inline function _kuramoto_delayed_force!(out, current, lagged, coupling)
    n = length(current)
    @inbounds for j in 1:n
        acc = 0.0
        current_j = current[j]
        for k in 1:n
            acc += coupling[j, k] * sin(lagged[k] - current_j)
        end
        out[j] = acc
    end
end

@inline function _delayed_lagged!(out, buffer, position, n)
    lower = floor(Int, position)
    weight = position - lower
    base = lower * n
    if weight == 0.0
        @inbounds for i in 1:n
            out[i] = buffer[base + i]
        end
    else
        high_base = (lower + 1) * n
        @inbounds for i in 1:n
            out[i] = (1.0 - weight) * buffer[base + i] + weight * buffer[high_base + i]
        end
    end
end

function kuramoto_delayed_trajectory(
    initial_history::AbstractMatrix{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    dt::Real,
    n_steps::Integer,
)
    n = length(omega)
    delay_steps = size(initial_history, 1) - 1
    freq = collect(Float64, omega)
    kmat = Matrix{Float64}(coupling)

    buffer = Float64[]
    sizehint!(buffer, (delay_steps + 1 + n_steps) * n)
    for s in 1:(delay_steps + 1)
        @inbounds for j in 1:n
            push!(buffer, Float64(initial_history[s, j]))
        end
    end

    theta = zeros(Float64, n)
    lagged = zeros(Float64, n)
    k1 = zeros(Float64, n); k2 = zeros(Float64, n); k3 = zeros(Float64, n); k4 = zeros(Float64, n)
    stage = zeros(Float64, n)
    nxt = zeros(Float64, n)

    phases = Float64[]
    sizehint!(phases, (n_steps + 1) * n)
    dbase = delay_steps * n
    @inbounds for j in 1:n
        push!(phases, buffer[dbase + j])
    end

    for step in 0:(n_steps - 1)
        base = (delay_steps + step) * n
        @inbounds for j in 1:n
            theta[j] = buffer[base + j]
        end
        pos = Float64(step)
        _delayed_lagged!(lagged, buffer, pos, n)
        _kuramoto_delayed_force!(k1, theta, lagged, kmat)
        @inbounds for j in 1:n
            k1[j] += freq[j]
            stage[j] = theta[j] + 0.5 * dt * k1[j]
        end
        _delayed_lagged!(lagged, buffer, pos + 0.5, n)
        _kuramoto_delayed_force!(k2, stage, lagged, kmat)
        @inbounds for j in 1:n
            k2[j] += freq[j]
            stage[j] = theta[j] + 0.5 * dt * k2[j]
        end
        _delayed_lagged!(lagged, buffer, pos + 0.5, n)
        _kuramoto_delayed_force!(k3, stage, lagged, kmat)
        @inbounds for j in 1:n
            k3[j] += freq[j]
            stage[j] = theta[j] + dt * k3[j]
        end
        _delayed_lagged!(lagged, buffer, pos + 1.0, n)
        _kuramoto_delayed_force!(k4, stage, lagged, kmat)
        @inbounds for j in 1:n
            k4[j] += freq[j]
            nxt[j] = theta[j] + (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
        end
        append!(buffer, nxt)
        append!(phases, nxt)
    end

    times = Float64[dt * s for s in 0:n_steps]
    return times, phases
end
