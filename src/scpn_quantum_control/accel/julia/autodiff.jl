# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia tier for the differentiable networked-Kuramoto Euler integrator

function kuramoto_euler_trajectory(
    theta0::AbstractVector{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    dt::Real,
    n_steps::Integer,
)::Matrix{Float64}
    # θ_{n+1} = θ_n + dt (ω + F(θ_n)), F_j = Σ_k K_jk sin(θ_k − θ_j).
    n = length(theta0)
    traj = zeros(Float64, n_steps + 1, n)
    current = Float64.(theta0)
    @inbounds for j in 1:n
        traj[1, j] = current[j]
    end
    dtf = Float64(dt)
    @inbounds for step in 1:n_steps
        for j in 1:n
            acc = 0.0
            for k in 1:n
                acc += Float64(coupling[j, k]) * sin(current[k] - current[j])
            end
            traj[step + 1, j] = current[j] + dtf * (Float64(omega[j]) + acc)
        end
        for j in 1:n
            current[j] = traj[step + 1, j]
        end
    end
    return traj
end

function kuramoto_euler_vjp(
    trajectory::AbstractMatrix{<:Real},
    coupling::AbstractMatrix{<:Real},
    dt::Real,
    cotangent::AbstractVector{<:Real},
)::Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}
    # λ_n = λ_{n+1} + dt J(θ_n)ᵀ λ_{n+1}; ∂L/∂ω = dt Σ λ_{n+1};
    # ∂L/∂K_pq = dt Σ λ_{n+1,p} sin(θ_q − θ_p).
    n_steps = size(trajectory, 1) - 1
    n = size(trajectory, 2)
    adjoint = Float64.(cotangent)
    g_omega = zeros(Float64, n)
    g_coupling = zeros(Float64, n, n)
    dtf = Float64(dt)
    jac = zeros(Float64, n, n)
    @inbounds for step in n_steps:-1:1
        phases = Float64[trajectory[step, j] for j in 1:n]
        for p in 1:n
            g_omega[p] += dtf * adjoint[p]
            for q in 1:n
                g_coupling[p, q] += dtf * adjoint[p] * sin(phases[q] - phases[p])
            end
        end
        for j in 1:n
            diag = 0.0
            for l in 1:n
                if l != j
                    e = Float64(coupling[j, l]) * cos(phases[l] - phases[j])
                    jac[j, l] = e
                    diag -= e
                end
            end
            jac[j, j] = diag
        end
        next = copy(adjoint)
        for l in 1:n
            acc = 0.0
            for j in 1:n
                acc += jac[j, l] * adjoint[j]
            end
            next[l] += dtf * acc
        end
        adjoint = next
    end
    return (adjoint, g_omega, g_coupling)
end
