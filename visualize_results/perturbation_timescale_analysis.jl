#=
Perturbation timescale analysis for the STF visualization fields.

This script reuses the field-building helpers from visualize_STF.jl via the
shared visualize_STF_helpers.jl file. It analytically factors the time
dependence of the velocity perturbation,

    v_case(t) = v0 .+ ε .* (p .* t .+ q .* sin(t)),

and solves for the first threshold-crossing time using only precomputed arrays.
It also reports a simplified comparison timescale obtained by retaining only
the linear `p * t` velocity component.
=#

include("visualize_STF_helpers.jl")

"""
    precompute_velocity_timescale_coefficients(x, z, Ro, v0, Fv, ψ¹ₚ, ϕ12s)

Precompute `(v0, p, q)` for the analytically factored velocity perturbation

```julia
v_case(t) = v0 .+ ε .* (p .* t .+ q .* sin(t))
```

where `p = Fv - C`, `q = C`, and
`C = Ro * ((1 / Ro^2 + ϕxx) * ψ¹ₚz - ϕxz * ψ¹ₚx)`.
"""
function precompute_velocity_timescale_coefficients(x, z, Ro, v0, Fv, ψ¹ₚ, ϕ12s)
    dx, dz = x[2] - x[1], z[2] - z[1]
    ϕxx, ϕxz, _ = ϕ12s
    ψ¹ₚz = central_diff(ψ¹ₚ, dz, 2)
    ψ¹ₚx = central_diff(ψ¹ₚ, dx, 1)

    C = Ro .* ((1 / Ro^2 .+ ϕxx) .* ψ¹ₚz .- ϕxz .* ψ¹ₚx)
    p = Fv .- C
    q = C

    return (v0 = v0, p = p, q = q)
end

function maximum_abs_velocity!(scratch, v0, p, q, ε, t)
    sin_t = sin(t)
    @. scratch = v0 + ε * (p * t + q * sin_t)
    return maximum(abs, scratch)
end

"""
    first_velocity_threshold_crossing(v0, p, q, ε, threshold; kwargs...)

Find the smallest `t ≥ 0` satisfying

```julia
maximum(abs, v0 .+ ε .* (p .* t .+ q .* sin(t))) > threshold
```

The bracketing search uses a Lipschitz-safe adaptive step based on
`L = ε * maximum(abs.(p) .+ abs.(q))`; the first bracket is then refined with
bisection.
"""
function first_velocity_threshold_crossing(v0, p, q, ε, threshold;
                                           dt_min = 1e-6,
                                           dt_max = π / 16,
                                           tolerance = 1e-10,
                                           t_max = 1e6,
                                           max_steps = 10_000_000)
    scratch = similar(v0, Float64)
    g(t) = maximum_abs_velocity!(scratch, v0, p, q, ε, t) - threshold

    g0 = g(0.0)
    if g0 > 0
        return 0.0
    end

    L = ε * maximum(abs.(p) .+ abs.(q))
    if !(isfinite(L)) || L <= 0
        return Inf
    end

    tlo = 0.0
    glo = g0

    for _ in 1:max_steps
        dt = clamp(0.5 * (-glo / L), dt_min, dt_max)
        thi = tlo + dt
        ghi = g(thi)

        if ghi > 0
            while thi - tlo > tolerance
                tm = 0.5 * (tlo + thi)
                gm = g(tm)
                if gm > 0
                    thi = tm
                else
                    tlo = tm
                end
            end

            return thi
        end

        tlo = thi
        glo = ghi

        if tlo >= t_max
            return Inf
        end
    end

    return Inf
end


"""
    first_linear_velocity_threshold_crossing(v0, p, ε, threshold)

Return the threshold-crossing timescale obtained from the simplified linear
model

```julia
v_linear(t) = v0 .+ ε .* p .* t
```

This is computed analytically for each grid point and returns the earliest
positive boundary of `abs(v0 + ε * p * t) == threshold`, matching the boundary
convention used by the bisection-based oscillatory calculation.
"""
function first_linear_velocity_threshold_crossing(v0, p, ε, threshold)
    if maximum(abs, v0) > threshold
        return 0.0
    end

    t_cross = Inf
    for i in eachindex(v0, p)
        slope = ε * p[i]
        if slope == 0
            continue
        end

        t1 = (-threshold - v0[i]) / slope
        t2 = ( threshold - v0[i]) / slope
        t_exit = max(t1, t2)

        if t_exit >= 0
            t_cross = min(t_cross, t_exit)
        end
    end

    return t_cross
end

function recompute_timescale_fields(x, z, Ro)
    psi1_erf_orig = compute_psi1_field(x, z, Ro, B_funcs_erf)
    psi1_gauss_orig = compute_psi1_field(x, z, Ro, B_funcs_gauss)
    psi1_erf_new = compute_psi1_field(x, z, Ro, B_funcs_erf; use_new_A_coeffs = true)
    psi1_gauss_new = compute_psi1_field(x, z, Ro, B_funcs_gauss; use_new_A_coeffs = true)

    _, v_erf = compute_b0v0_field(x, z, Ro, B_funcs_erf)
    _, v_gauss = compute_b0v0_field(x, z, Ro, B_funcs_gauss; gauss = true)

    ϕxx_erf, ϕxz_erf, ϕzz_erf = compute_ϕ12_field(x, z, Ro, B_funcs_erf)
    ϕxx_gauss, ϕxz_gauss, ϕzz_gauss = compute_ϕ12_field(x, z, Ro, B_funcs_gauss)
    _, ϕxzz_erf, _, ϕxxx_erf, _, _ = compute_blt_fields(x, z, Ro, B_funcs_erf)
    _, ϕxzz_gauss, _, ϕxxx_gauss, _, _ = compute_blt_fields(x, z, Ro, B_funcs_gauss)

    return (; psi1_erf_orig, psi1_gauss_orig, psi1_erf_new, psi1_gauss_new,
            v_erf, v_gauss,
            ϕxx_erf, ϕxz_erf, ϕzz_erf, ϕxzz_erf, ϕxxx_erf,
            ϕxx_gauss, ϕxz_gauss, ϕzz_gauss, ϕxzz_gauss, ϕxxx_gauss)
end

function build_perturbation_cases(x, z, Ro, fields)
    erf_ϕ12s = (fields.ϕxx_erf, fields.ϕxz_erf, fields.ϕzz_erf)
    gauss_ϕ12s = (fields.ϕxx_gauss, fields.ϕxz_gauss, fields.ϕzz_gauss)

    return [
        (profile = :erf, perturbation = :V,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_erf, 0.0 .* fields.ϕxzz_erf, fields.psi1_erf_orig, erf_ϕ12s)),
        (profile = :erf, perturbation = :H,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_erf, 0.0 .* fields.ϕxxx_erf, fields.psi1_erf_new, erf_ϕ12s)),
        (profile = :erf, perturbation = :Vv,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_erf, fields.ϕxzz_erf, -fields.psi1_erf_orig, erf_ϕ12s)),
        (profile = :erf, perturbation = :Hv,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_erf, fields.ϕxxx_erf, -fields.psi1_erf_new, erf_ϕ12s)),
        (profile = :gauss, perturbation = :V,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_gauss, 0.0 .* fields.ϕxzz_gauss, fields.psi1_gauss_orig, gauss_ϕ12s)),
        (profile = :gauss, perturbation = :H,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_gauss, 0.0 .* fields.ϕxxx_gauss, fields.psi1_gauss_new, gauss_ϕ12s)),
        (profile = :gauss, perturbation = :Vv,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_gauss, fields.ϕxzz_gauss, -fields.psi1_gauss_orig, gauss_ϕ12s)),
        (profile = :gauss, perturbation = :Hv,
         coefficients = precompute_velocity_timescale_coefficients(x, z, Ro, fields.v_gauss, fields.ϕxxx_gauss, -fields.psi1_gauss_new, gauss_ϕ12s)),
    ]
end

"""
    perturbation_timescale_analysis(; ε = 0.03, Ro = 1.0)

Recompute all Ro-dependent STF fields, build the eight requested perturbation
cases, and print/return the 16 threshold-crossing rows for multipliers
`1 + ε` and `1 + 10ε`. Each row includes `t`, the full oscillatory
threshold time, and `t_linear`, the comparison time from `v0 + ε * p * t`.
"""
function perturbation_timescale_analysis(; ε = 0.03, Ro = 1.0)
    ε = Float64(ε)
    Ro = Float64(Ro)

    fields = recompute_timescale_fields(x, z, Ro)
    cases = build_perturbation_cases(x, z, Ro, fields)
    vmax0 = Dict(:erf => maximum(abs, fields.v_erf),
                 :gauss => maximum(abs, fields.v_gauss))
    thresholds = ((multiplier = Symbol("1+ε"), factor = 1 + ε),
                  (multiplier = Symbol("1+10ε"), factor = 1 + 10ε))

    rows = NamedTuple[]
    for case in cases
        v0, p, q = case.coefficients
        for threshold in thresholds
            threshold_value = threshold.factor * vmax0[case.profile]
            t_cross = first_velocity_threshold_crossing(v0, p, q, ε, threshold_value)
            t_linear = first_linear_velocity_threshold_crossing(v0, p, ε, threshold_value)
            row = (profile = case.profile,
                   perturbation = case.perturbation,
                   multiplier = threshold.multiplier,
                   ε = ε,
                   Ro = Ro,
                   t = t_cross,
                   t_linear = t_linear)
            push!(rows, row)
            println(row)
        end
    end

    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    perturbation_timescale_analysis()
end
