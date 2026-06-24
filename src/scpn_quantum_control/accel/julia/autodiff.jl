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

function _networked_force!(
    theta::AbstractVector{Float64},
    coupling::AbstractMatrix{<:Real},
    n::Integer,
    out::AbstractVector{Float64},
)
    @inbounds for j in 1:n
        acc = 0.0
        tj = theta[j]
        for k in 1:n
            acc += Float64(coupling[j, k]) * sin(theta[k] - tj)
        end
        out[j] = acc
    end
    return out
end

function kuramoto_rk4_trajectory(
    theta0::AbstractVector{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    dt::Real,
    n_steps::Integer,
)::Matrix{Float64}
    n = length(theta0)
    traj = zeros(Float64, n_steps + 1, n)
    current = Float64.(theta0)
    om = Float64.(omega)
    @inbounds for j in 1:n
        traj[1, j] = current[j]
    end
    dtf = Float64(dt)
    half = dtf / 2
    k1 = zeros(Float64, n)
    k2 = zeros(Float64, n)
    k3 = zeros(Float64, n)
    k4 = zeros(Float64, n)
    s = zeros(Float64, n)
    @inbounds for step in 1:n_steps
        _networked_force!(current, coupling, n, k1)
        for j in 1:n
            k1[j] += om[j]
            s[j] = current[j] + half * k1[j]
        end
        _networked_force!(s, coupling, n, k2)
        for j in 1:n
            k2[j] += om[j]
            s[j] = current[j] + half * k2[j]
        end
        _networked_force!(s, coupling, n, k3)
        for j in 1:n
            k3[j] += om[j]
            s[j] = current[j] + dtf * k3[j]
        end
        _networked_force!(s, coupling, n, k4)
        for j in 1:n
            k4[j] += om[j]
            current[j] += (dtf / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            traj[step + 1, j] = current[j]
        end
    end
    return traj
end

function _add_dt_jt_product!(
    theta::AbstractVector{Float64},
    coupling::AbstractMatrix{<:Real},
    n::Integer,
    lambda::AbstractVector{Float64},
    out::AbstractVector{Float64},
)
    @inbounds for j in 1:n
        tj = theta[j]
        lam_j = lambda[j]
        diag = 0.0
        for l in 1:n
            if l != j
                e = Float64(coupling[j, l]) * cos(theta[l] - tj)
                out[l] += e * lam_j
                diag += e
            end
        end
        out[j] -= diag * lam_j
    end
    return out
end

function kuramoto_rk4_vjp(
    trajectory::AbstractMatrix{<:Real},
    omega::AbstractVector{<:Real},
    coupling::AbstractMatrix{<:Real},
    dt::Real,
    cotangent::AbstractVector{<:Real},
)::Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}
    n_steps = size(trajectory, 1) - 1
    n = size(trajectory, 2)
    om = Float64.(omega)
    dtf = Float64(dt)
    half = dtf / 2
    adjoint = Float64.(cotangent)
    g_omega = zeros(Float64, n)
    g_coupling = zeros(Float64, n, n)
    k1 = zeros(Float64, n)
    k2 = zeros(Float64, n)
    k3 = zeros(Float64, n)
    s2 = zeros(Float64, n)
    s3 = zeros(Float64, n)
    s4 = zeros(Float64, n)
    @inbounds for step in n_steps:-1:1
        phases = Float64[trajectory[step, j] for j in 1:n]
        _networked_force!(phases, coupling, n, k1)
        for j in 1:n
            k1[j] += om[j]
            s2[j] = phases[j] + half * k1[j]
        end
        _networked_force!(s2, coupling, n, k2)
        for j in 1:n
            k2[j] += om[j]
            s3[j] = phases[j] + half * k2[j]
        end
        _networked_force!(s3, coupling, n, k3)
        for j in 1:n
            k3[j] += om[j]
            s4[j] = phases[j] + dtf * k3[j]
        end
        gc1 = [(dtf / 6) * adjoint[j] for j in 1:n]
        gc2 = [(dtf / 3) * adjoint[j] for j in 1:n]
        gc3 = [(dtf / 3) * adjoint[j] for j in 1:n]
        gc4 = [(dtf / 6) * adjoint[j] for j in 1:n]
        next = copy(adjoint)
        b = zeros(Float64, n)
        fill!(b, 0.0)
        _add_dt_jt_product!(s4, coupling, n, gc4, b)
        for j in 1:n
            next[j] += b[j]
            gc3[j] += dtf * b[j]
        end
        fill!(b, 0.0)
        _add_dt_jt_product!(s3, coupling, n, gc3, b)
        for j in 1:n
            next[j] += b[j]
            gc2[j] += half * b[j]
        end
        fill!(b, 0.0)
        _add_dt_jt_product!(s2, coupling, n, gc2, b)
        for j in 1:n
            next[j] += b[j]
            gc1[j] += half * b[j]
        end
        fill!(b, 0.0)
        _add_dt_jt_product!(phases, coupling, n, gc1, b)
        for j in 1:n
            next[j] += b[j]
        end
        for j in 1:n
            g_omega[j] += gc1[j] + gc2[j] + gc3[j] + gc4[j]
        end
        for (stage, gc) in ((phases, gc1), (s2, gc2), (s3, gc3), (s4, gc4))
            for p in 1:n
                tp = stage[p]
                lam_p = gc[p]
                for q in 1:n
                    g_coupling[p, q] += lam_p * sin(stage[q] - tp)
                end
            end
        end
        adjoint = next
    end
    return (adjoint, g_omega, g_coupling)
end
