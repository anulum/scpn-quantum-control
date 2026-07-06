# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the adaptive Dormand–Prince networked-Kuramoto trajectory

# Error-controlled embedded 4/5 pair (DOPRI5) with the standard elementary step controller,
# mirroring the Python and Rust references so the three agree to the requested tolerance (an
# adaptive scheme's realised grid is tolerance-parity, not bit-identical). Returns the accepted
# times, the phases at those times flattened row-major as (M+1)·N, and the realised step sizes.

const _DOPRI_TABLEAU = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
const _DOPRI_FIFTH = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
const _DOPRI_ERROR = (
    35 / 384 - 5179 / 57600,
    0.0,
    500 / 1113 - 7571 / 16695,
    125 / 192 - 393 / 640,
    -2187 / 6784 + 92097 / 339200,
    11 / 84 - 187 / 2100,
    -1 / 40,
)

@inline function _kuramoto_dopri_force!(out, theta, coupling)
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

function kuramoto_dopri_trajectory(
    theta0::AbstractVector{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    t_end::Real,
    rtol::Real,
    atol::Real,
    safety::Real,
    min_factor::Real,
    max_factor::Real,
    max_steps::Integer,
)
    n = length(theta0)
    y = collect(Float64, theta0)
    freq = collect(Float64, omega)
    kmat = Matrix{Float64}(coupling)
    deriv = [zeros(Float64, n) for _ in 1:7]
    stage = zeros(Float64, n)
    proposed = zeros(Float64, n)
    _kuramoto_dopri_force!(deriv[1], y, kmat)
    deriv[1] .+= freq
    # Initial step mirrors the Python floor's ``t_end / 100`` guess so the adaptive controller
    # walks the same realised grid on well-conditioned problems.
    step = t_end / 100.0
    time = 0.0
    times = Float64[0.0]
    phases = collect(Float64, y)
    steps = Float64[]
    accepted = 0
    while time < t_end && accepted < max_steps
        if step > t_end - time
            step = t_end - time
        end
        for s in 2:7
            @inbounds for i in 1:n
                acc = y[i]
                for p in 1:(s - 1)
                    acc += step * _DOPRI_TABLEAU[s][p] * deriv[p][i]
                end
                stage[i] = acc
            end
            _kuramoto_dopri_force!(deriv[s], stage, kmat)
            deriv[s] .+= freq
        end
        err = 0.0
        @inbounds for i in 1:n
            fifth = 0.0
            embedded = 0.0
            for s in 1:7
                fifth += _DOPRI_FIFTH[s] * deriv[s][i]
                embedded += _DOPRI_ERROR[s] * deriv[s][i]
            end
            proposed[i] = y[i] + step * fifth
            scale = atol + rtol * max(abs(y[i]), abs(proposed[i]))
            scaled = (step * embedded) / scale
            err += scaled * scaled
        end
        err = sqrt(err / n)
        if err <= 1.0
            time += step
            y .= proposed
            deriv[1] .= deriv[7]
            push!(times, time)
            append!(phases, y)
            push!(steps, step)
            accepted += 1
        end
        factor = err == 0.0 ? max_factor : safety * err^(-0.2)
        step *= clamp(factor, min_factor, max_factor)
    end
    return times, phases, steps
end
