#=
File: compute_psi1.jl
Author: Gemini @ MIT
Date: 2025-07-07
Description: Shared field-building helpers extracted from visualize_STF.jl
             for the semi-analytic STF solution.
=#

using SpecialFunctions # For erf()
# ------------------------------------------------------------------
# 1.  Setup and Grid Definition
# ------------------------------------------------------------------
println("Setting up grid and parameters...")

# Physical domain
x_min, x_max = -4.0, 4.0
z_min, z_max = 0.0, 1.0
Ro = 1.0  # Rossby number
γ = 0.03 # Strain parameter

# Computational grids (moderate resolution; raise if desired)
nx, nz = 401, 101
x = collect(range(x_min, x_max; length = nx))   # <- make it a Vector!
z = collect(range(z_min, z_max; length = nz))   # (z isn’t mutated, but keep symmetrical)
const dx = x[2] - x[1]
const dz = z[2] - z[1]

# ------------------------------------------------------------------
# 2.  Background Buoyancy Profile and Derivatives
#
# B₀(X) is defined using the error function, representing a shear layer.
# We need up to the fourth derivative for the Aᵢ coefficients.
# ------------------------------------------------------------------
println("Defining buoyancy profiles...")
const invs2π = 1 / √(2π)

# --- Case 1: Error Function Profile (Original) ---
B0_erf(X)   = 0.5 * erf(X / √2)
dB0_erf(X)  = @. exp(-0.5 * X^2) * invs2π
d2B0_erf(X) = @. -X * exp(-0.5 * X^2) * invs2π
d3B0_erf(X) = @. (X^2 - 1) * exp(-0.5 * X^2) * invs2π
d4B0_erf(X) = @. (3X - X^3) * exp(-0.5 * X^2) * invs2π

const B_funcs_erf = (dB0_erf, d2B0_erf, d3B0_erf, d4B0_erf)

# --- Case 2: Gaussian Profile (New) ---
B0_gauss(X)   = @. -0.5 * exp(-0.5 * X^2)
dB0_gauss(X)  = @. 0.5 * X * exp(-0.5 * X^2)
d2B0_gauss(X) = @. -0.5 * (X^2 - 1) * exp(-0.5 * X^2)
d3B0_gauss(X) = @. 0.5 * X * (X^2 - 3) * exp(-0.5 * X^2)
d4B0_gauss(X) = @. -0.5 * (X^4 - 6*X^2 + 3) * exp(-0.5 * X^2) # Note: 2*1.5=3, 2*3=6

const B_funcs_gauss = (dB0_gauss, d2B0_gauss, d3B0_gauss, d4B0_gauss)

# ------------------------------------------------------------------
# 3.  Generic ψ¹ Computation
#
# These functions are now generic and take the required buoyancy
# derivatives as arguments, allowing us to reuse the same logic
# for different physical cases.
# ------------------------------------------------------------------

"""
    solve_X0_vec(x_vec, z_val, Ro, dB0, d2B0)

Generic Newton solver for X₀ that accepts buoyancy derivatives.
"""
function solve_X0_vec(x_vec, z_val::Float64, Ro::Float64, dB0::Function, d2B0::Function; maxiter=50, tol=1e-12)
    X = collect(x_vec) # Initial guess for X₀ is x
    Ro_sq_z = Ro^2 * (z_val - 0.5)

    for _ in 1:maxiter
        Bp  = dB0(X)
        Bpp = d2B0(X)
        f   = X .- (x_vec .+ Ro_sq_z .* Bp)
        df  = 1.0 .- Ro_sq_z .* Bpp
        dX  = f ./ df
        X .-= dX
        if maximum(abs, dX) < tol; break; end
    end
    return X
end

"""
    compute_coefficients(X0, Ro, B_funcs; use_new_A_coeffs)

Generic coefficient calculator. A boolean flag switches between Aᵢ sets.
"""
function compute_coefficients(X0, Ro, B_funcs; use_new_A_coeffs = false, return_A = false)
    dB0, d2B0, d3B0, d4B0 = B_funcs

    B_p = dB0(X0); B_pp = d2B0(X0); B_ppp = d3B0(X0); B_pppp = d4B0(X0)

    Ro2 = Ro^2; Ro4 = Ro^4; Bp_sq = B_p^2; Bpp_sq = B_pp^2

    local A0, A1, A2
    if !use_new_A_coeffs
        # Original Aᵢ coefficient definitions
        A0 = 3 * B_p * (2 * Bpp_sq + B_ppp * B_p)
        A1 = -Ro2 * B_p * (12 * B_pp^3 - B_pppp * Bp_sq - 3 * B_ppp * B_p * B_pp)
        A2 = Ro4 * B_p * (6 * Bpp_sq * (Bpp_sq - B_ppp * B_p) - Bp_sq * (B_pppp * B_pp - 3 * B_ppp^2))
    else
        # New Aᵢ coefficient definitions
        A0 = B_ppp
        A1 = Ro2 * (B_ppp * B_pp + B_pppp * B_p)
        A2 = -Ro4 * (2 * B_ppp * Bpp_sq + B_p * (B_pppp * B_pp - 3 * B_ppp^2))
    end

    if return_A
        return A0, A1, A2
    end

    # Compute the Cᵢ coefficients

    Ro2Bp = Ro2 * B_p
    Psi(Z) = 2 + Ro2Bp * (1 - 2Z)
    denom = (Ro2Bp)^4

    if abs(denom) < 1e-20; return ntuple(_ -> 0.0, 5); end

    C3 = A2 / (2 * denom)
    C4 = (A1 * Ro2Bp + 2 * A2) / denom
    C5 = (A0 * Ro2Bp^2 + A1 * Ro2Bp + A2) / denom

    function f(Psi_val)
        if Psi_val <= 0; return 0.0; end
        return C3 * Psi_val * (log(Psi_val / 2) - 1) + C4 * log(Psi_val) + C5 / Psi_val
    end

    C2 = -f(Psi(0.0))
    C1 = -(C2 + f(Psi(1.0)))

    return C1, C2, C3, C4, C5
end

"""
    compute_psi1_field(x, z, Ro, B_funcs)

Top-level function to compute the entire ψ¹ field for a given set of
buoyancy functions.
"""
function compute_psi1_field(x, z, Ro, B_funcs; use_new_A_coeffs::Bool = false, strain=false)
    nx, nz = length(x), length(z)
    psi1_field = zeros(Float64, nx, nz)
    dB0_func, d2B0_func = B_funcs[1], B_funcs[2]

    for j in 1:nz
        z_val = z[j]
        X0_row = solve_X0_vec(x, z_val, Ro, dB0_func, d2B0_func)

        for i in 1:nx
            X0_val = X0_row[i]
            if strain
                psi1_field[i, j] = -Ro * dB0_func(X0_val) * z_val * (z_val-1)
            else
                C1, C2, C3, C4, C5 = compute_coefficients(X0_val, Ro, B_funcs; use_new_A_coeffs)

                Ro2Bp = Ro^2 * dB0_func(X0_val)
                Psi_val = 2 + Ro2Bp * (1 - 2 * z_val)

                if Psi_val <= 0; psi1_field[i, j] = NaN; continue; end

                term3 = C3 * Psi_val * (log(Psi_val / 2) - 1)
                term4 = C4 * log(Psi_val)
                term5 = C5 / Psi_val

                psi1_field[i, j] = Ro * (C1 * z_val + C2 + term3 + term4 + term5)
            end
        end
    end
    return psi1_field
end

function compute_blt_fields(x, z, Ro, B_funcs)
    nx, nz = length(x), length(z)
    ϕzzz_field = zeros(Float64, nx, nz)
    ϕxzz_field = zeros(Float64, nx, nz)
    ϕxxz_field = zeros(Float64, nx, nz)
    ϕxxx_field = zeros(Float64, nx, nz)
    ϕγv_field = zeros(Float64, nx, nz)
    ϕγb_field = zeros(Float64, nx, nz)
    dB0_func, d2B0_func, d3B0_func = B_funcs[1], B_funcs[2], B_funcs[3]

    for j in 1:nz
        z_val = z[j]
        X0_row = solve_X0_vec(x, z_val, Ro, dB0_func, d2B0_func)

        for i in 1:nx
            X0_val = X0_row[i]
            dB0_val, d2B0_val, d3B0_val = dB0_func(X0_val), d2B0_func(X0_val), d3B0_func(X0_val)
            J0_val = 1 / (1 - Ro^2 * d2B0_val * (z_val - 0.5))
            ϕzzz_field[i, j] = Ro^4 * dB0_val^2 * J0_val^2 * (3d2B0_val+Ro^2*dB0_val*d3B0_val*(z_val-0.5)*J0_val)
            ϕxzz_field[i, j] = Ro^2 * dB0_val * J0_val^2 * (2d2B0_val+Ro^2*dB0_val*d3B0_val*(z_val-0.5)*J0_val)
            ϕxxz_field[i, j] = J0_val^2 * (d2B0_val + Ro^2*dB0_val*d3B0_val*(z_val - 0.5)*J0_val)
            ϕxxx_field[i, j] = d3B0_val * J0_val^3 * (z_val - 0.5)
            ϕγv_field[i, j] = - dB0_val*(z_val - 0.5) + x[i]*J0_val*d2B0_val*(z_val - 0.5)
            ϕγb_field[i, j] = x[i]*J0_val*dB0_val
        end
    end

    return ϕzzz_field, ϕxzz_field, ϕxxz_field, ϕxxx_field, ϕγv_field, ϕγb_field
end

function compute_b0v0_field(x, z, Ro, B_funcs; gauss = false)
    nx, nz = length(x), length(z)
    b_field = zeros(Float64, nx, nz)
    v_field = zeros(Float64, nx, nz)
    dB0_func, d2B0_func = B_funcs[1], B_funcs[2]

    for j in 1:nz
        z_val = z[j]
        X0_row = solve_X0_vec(x, z_val, Ro, dB0_func, d2B0_func)

        for i in 1:nx
            X0_val = X0_row[i]
            if gauss
                b_field[i,j] = B0_gauss(X0_val)
            else
                b_field[i,j] = B0_erf(X0_val)
            end
            v_field[i, j] = dB0_func(X0_val) * (z_val - 0.5)
        end
    end

    return b_field, v_field
end

function compute_ϕ12_field(x, z, Ro, B_funcs)
    nx, nz = length(x), length(z)
    ϕxx_field = zeros(Float64, nx, nz)
    ϕxz_field = zeros(Float64, nx, nz)
    ϕzz_field = zeros(Float64, nx, nz)
    dB0_func, d2B0_func = B_funcs[1], B_funcs[2]

    for j in 1:nz
        z_val = z[j]
        X0_row = solve_X0_vec(x, z_val, Ro, dB0_func, d2B0_func)

        for i in 1:nx
            X0_val = X0_row[i]
            dB0_val, d2B0_val = dB0_func(X0_val), d2B0_func(X0_val)
            J0_val = 1 / (1 - Ro^2 * d2B0_val * (z_val - 0.5))
            ϕxx_field[i, j] = J0_val*d2B0_val*(z_val - 0.5)
            ϕxz_field[i, j] = J0_val*dB0_val
            ϕzz_field[i, j] = J0_val*dB0_val^2
        end
    end

    return ϕxx_field, ϕxz_field, ϕzz_field
end

function central_diff(A, ds, dim)
    out = fill(NaN, size(A))
    if dim == 1 # d/dx for (nx, nz) matrix
        C = 2:size(A,1)-1; K = 1:size(A,2)
        @views @. out[C,K] = (A[C.+1,K] - A[C.-1,K]) / (2*ds)
        out[1, :] = (A[2, :] - A[1, :]) / ds
        out[end, :] = (A[end, :] - A[end-1, :]) / ds
    elseif dim == 2 # d/dz for (nx, nz) matrix
        C = 1:size(A,1); K = 2:size(A,2)-1
        @views @. out[C,K] = (A[C,K.+1] - A[C,K.-1]) / (2*ds)
        out[:, 1] = (A[:, 2] - A[:, 1]) / ds
        out[:, end] = (A[:, end] - A[:, end-1]) / ds
    end
    return out
end

function compute_b1v1_field(x, z, Ro, Fb, Fv, ψ¹ₚ, ϕ12s, t)
    dx, dz = x[2] - x[1], z[2] - z[1]
    ϕxx, ϕxz, ϕzz = ϕ12s[1], ϕ12s[2], ϕ12s[3]
    ψ¹ₚz = central_diff(ψ¹ₚ, dz, 2)
    ψ¹ₚx = central_diff(ψ¹ₚ, dx, 1)

    b_field = Fb * t
    v_field = Fv * t
    b_field .-= Ro * (ϕxz.* ψ¹ₚz .- ϕzz.* ψ¹ₚx) * (t - sin(t))
    v_field .-= Ro * ((1/Ro^2 .+ ϕxx).* ψ¹ₚz .- ϕxz.* ψ¹ₚx) * (t - sin(t))
    return b_field, v_field
end

function compute_T1b_field(x, z, Ro, Fb, Fv, ψ¹ₚ, ϕ12s, t; γ̃ = 1, dt = 0.001)
    dx, dz = x[2] - x[1], z[2] - z[1]
    _, ϕxz, _ = ϕ12s[1], ϕ12s[2], ϕ12s[3]
    ψ¹ₚz = central_diff(ψ¹ₚ, dz, 2)
    ψ¹ₚx = central_diff(ψ¹ₚ, dx, 1)
    b1⁺, _ = compute_b1v1_field(x, z, Ro, Fb, Fv, ψ¹ₚ, ϕ12s, t+dt)
    b1⁻, _ = compute_b1v1_field(x, z, Ro, Fb, Fv, ψ¹ₚ, ϕ12s, t-dt)
    b1⁺dx = central_diff(b1⁺, dx, 1)
    b1⁻dx = central_diff(b1⁻, dx, 1)

    T¹b = ϕxz .* (b1⁺dx .- b1⁻dx) / (2*dt) / Ro
    T¹b .+= 0.5*((ψ¹ₚz .- γ̃ / Ro * reshape(x,:,1)) .* central_diff(ϕxz.^2, dx, 1) .- ψ¹ₚx .* central_diff(ϕxz.^2, dz, 2))

    return T¹b
end

function compute_b3_fields(x, z, Ro, b0, ε, Fbs, Fvs, ψ¹ₚs, ϕ12s, t; ψ¹ₚγ = nothing, γ = 0.03)
    if isnothing(ψ¹ₚγ)
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[2], Fvs[2], 0*ψ¹ₚs[1], ϕ12s, t)
        bV = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, 0*Fbs[2], Fvs[2], -ψ¹ₚs[1], ϕ12s, t)
        bvV = b0 .+ ε * b1

        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[2].+Fbs[3], Fvs[2].+Fvs[3], 0*ψ¹ₚs[1], ϕ12s, t)
        bVH = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, (Fbs[2].+Fbs[3])*0.0, Fvs[2].+Fvs[3], - (ψ¹ₚs[1] .+ ψ¹ₚs[2]), ϕ12s, t)
        bvVH = b0 .+ ε * b1
    else
        b1, _ = compute_b1v1_field(x, z, Ro, γ/ε*Fbs[1].+Fbs[2], γ/ε*Fvs[1].+Fvs[2], γ/ε*ψ¹ₚγ, ϕ12s, t)
        bV = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, γ/ε*Fbs[1], γ/ε*Fvs[1].+Fvs[2], γ/ε*ψ¹ₚγ .- ψ¹ₚs[1], ϕ12s, t)
        bvV = b0 .+ ε * b1

        b1, _ = compute_b1v1_field(x, z, Ro, γ/ε*Fbs[1].+Fbs[2].+Fbs[3], γ/ε*Fvs[1].+Fvs[2].+Fvs[3], γ/ε*ψ¹ₚγ, ϕ12s, t)
        bVH = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, γ/ε*Fbs[1], γ/ε*Fvs[1].+Fvs[2].+Fvs[3], γ/ε*ψ¹ₚγ .- (ψ¹ₚs[1] .+ ψ¹ₚs[2]), ϕ12s, t)
        bvVH = b0 .+ ε * b1
    end

    return bV, bvV, bVH, bvVH
end


function compute_bVbH_fields(x, z, Ro, b0, ε, Fbs, Fvs, ψ¹ₚs, ϕ12s, t; ψ¹ₚγ = nothing, γ = 0.03)
    if isnothing(ψ¹ₚγ)
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[1], 0.0*Fvs[1], ψ¹ₚs[1], ϕ12s, t)
        bV = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[2], 0.0*Fvs[2], ψ¹ₚs[2], ϕ12s, t)
        bH = b0 .+ ε * b1

        b1, _ = compute_b1v1_field(x, z, Ro, 0.0*Fbs[1], Fvs[1], -ψ¹ₚs[1], ϕ12s, t)
        bVv = b0 .+ ε * b1
        b1, _ = compute_b1v1_field(x, z, Ro, 0.0*Fbs[2], Fvs[2], -ψ¹ₚs[2], ϕ12s, t)
        bHv = b0 .+ ε * b1
    else
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[1].+ε/γ*Fbs[2], Fvs[1], ψ¹ₚγ .+ ε/γ * ψ¹ₚs[1], ϕ12s, t)
        bV = b0 .+ γ * b1
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[1].+ε/γ*Fbs[3], Fvs[1], ψ¹ₚγ .+ ε/γ * ψ¹ₚs[2], ϕ12s, t)
        bH = b0 .+ γ * b1

        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[1], Fvs[1].+ε/γ*Fvs[2], ψ¹ₚγ .- ε/γ * ψ¹ₚs[1], ϕ12s, t)
        bVv = b0 .+ γ * b1
        b1, _ = compute_b1v1_field(x, z, Ro, Fbs[1], Fvs[1].+ε/γ*Fvs[3], ψ¹ₚγ .- ε/γ * ψ¹ₚs[2], ϕ12s, t)
        bHv = b0 .+ γ * b1
    end
    return bV, bH, bVv, bHv
end
