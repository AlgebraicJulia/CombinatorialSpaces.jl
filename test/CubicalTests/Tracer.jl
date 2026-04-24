using CairoMakie
using Distributions
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays
using CUDA
using CUDA.CUSPARSE
using OrdinaryDiffEqSSPRK

CUDA.allowscalar(false)

const USE_CUDA = CUDA.functional()
println("CUDA is functional: $USE_CUDA")

to_device(arr::AbstractVector{FT}) where FT <: AbstractFloat = USE_CUDA ? CuVector{FT}(arr) : arr
to_device(arr::AbstractVector{T}) where T = USE_CUDA ? CuVector{T}(arr) : arr
to_device(mat::AbstractMatrix{FT}) where FT <: AbstractFloat = USE_CUDA ? CuSparseMatrixCSC{FT}(mat) : SparseMatrixCSC{FT}(mat)

include("../../src/CubicalCode/UniformDEC.jl")

#######################
### Grid Setup      ###
#######################

const default_n = 257
const nx_env = get(ENV, "CS_TRACER_N", string(default_n))
const nx_ = parse(Int, nx_env)
const ny_ = nx_

const lx_ = ly_ = 1.0
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_, halo_x = 4, halo_y = 4)

const kappa   = 1e-5
const te      = 2.0
const dt      = min(edge_len(s, X_ALIGN), edge_len(s, Y_ALIGN)) / 2
const saveat  = max(1, floor(Int64, 0.05 / dt))

#####################
### DEC Operators ###
#####################

println("Constructing DEC operators...")

const cpu_d0 = exterior_derivative(Val(0), s);
const cpu_d1 = exterior_derivative(Val(1), s);

# No-flux dual derivative: prevents tracer from leaving the domain
const cpu_dual_d0 = no_flux_dual_derivative(Val(0), s);

const cpu_hdg_1 = hodge_star(Val(1), s);
const cpu_hdg_2 = hodge_star(Val(2), s);

const cpu_inv_hdg_1 = inv_hodge_star(Val(1), s);
const cpu_inv_hdg_2 = inv_hodge_star(Val(2), s);

const cpu_dual_δ1 = dual_codifferential(Val(1), s);

# Dual Laplacian on 0-forms: used for tracer diffusion
const cpu_dΔ0 = cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1 * cpu_dual_d0;

const d0 = to_device(cpu_d0);
const d1 = to_device(cpu_d1);
const dual_d0 = to_device(cpu_dual_d0);
const hdg_1 = to_device(cpu_hdg_1);
const hdg_2 = to_device(cpu_hdg_2);
const inv_hdg_1 = to_device(cpu_inv_hdg_1);
const inv_hdg_2 = to_device(cpu_inv_hdg_2);
const dual_δ1 = to_device(cpu_dual_δ1);
const dΔ0 = to_device(cpu_dΔ0);

println("DEC operators constructed.")
println("Loading simulation files...")

simname = get(ENV, "CS_TRACER_SIM", "Diagonal") # "Diagonal", "Stretch", "Rotate", "CircularVortex", "ReversedVortex"
println("Tracer configuration: sim=$(simname), n=$(nx_)x$(ny_)")
include("Tracer_Files/Adv_Tests.jl")

###############################
### Physical Tracer Operator ##
###############################

# Tracer continuity equation driven by a prescribed velocity.
#
# Given the dual 1-form velocity u and primal 1-form v (its primal counterpart),
# the tracer φ (as a dual 2-form) evolves according to:
#
#   ∂φ/∂t = -φ·(δu) - L(u, φ) + κ·Δφ
#
# where:
#   -φ·(δu)   is the divergence/creation term (zero for incompressible u)
#   L(u, φ)   is the Lie derivative (advection)
#   κ·Δφ      is the dual Laplacian diffusion

function tracer_continuity(
    u::AbstractVector{Float64},
    v::AbstractVector{Float64},
    phi_star::AbstractVector{Float64},
    kappa::Float64,
)
    phi = hdg_2 * phi_star

    # -φ·(δu): divergence/creation term (numerically zero for incompressible u)
    tracer_divergence = phi .* (dual_δ1 * u)

    # L(u, φ): Lie derivative advection term
    tracer_advection = hdg_2 * wedge_product(Val(1), Val(1), WENO5(), s, v, inv_hdg_1 * dual_d0 * phi)

    # κ·Δφ: diffusion via dual Laplacian on the scalar φ
    tracer_diffusion = kappa * dΔ0 * (inv_hdg_2 * phi)

    return inv_hdg_2 * (-tracer_divergence - tracer_advection + tracer_diffusion)
end

######################
### Time Integration ##
######################

# SSP-RK3 integrator.  The velocity (u, v) is held fixed throughout.
# We are solving ∂φ/∂t = F(φ) where F(φ) = -φ·(δu) - L(u, φ) + κ·Δφ
# In vector calculus this is ∂φ/∂t + ∇·(uφ) = κΔφ, so the tracer is advected by u and diffused by κ.
function run_tracer(
    phi_star_0::AbstractVector{Float64},
    te::Float64,
    dt::Float64,
    kappa::Float64;
    saveat::Int = 100,
    mass_every::Int = saveat,
)
    steps = ceil(Int64, te / dt)
    save_every = max(1, saveat)
    mass_every = max(1, mass_every)

    u0 = deepcopy(phi_star_0)
    m₀ = sum(hdg_2 * u0)

    function rhs!(dphi_star, phi_star, p, t)
        set_periodic!(phi_star, Val(2), s, ALL)

        X = generate_X(s, t, te)
        Y = generate_Y(s, t, te)
        u = flat_dd(s, X, Y)
        v = flat_dp(s, X, Y)

        dphi_star .= tracer_continuity(u, v, phi_star, kappa)
        return nothing
    end

    mass_condition(u, t, integrator) = integrator.iter > 0 && (integrator.iter % mass_every == 0)

    function mass_affect!(integrator)
        snap = hdg_2 * integrator.u
        m = sum(snap)
        println(@sprintf("Step %6d/%d | Relative mass: %.4f%%", integrator.iter, steps, 100 * m / m₀))
        flush(stdout)
        return nothing
    end

    callbacks = CallbackSet(
        DiscreteCallback(mass_condition, mass_affect!; save_positions = (false, false))
    )

    prob = ODEProblem(rhs!, u0, (0.0, te))
    sol = solve(
        prob,
        SSPRK33();
        dt = dt,
        adaptive = false,
        saveat = dt * save_every,
        save_start = true,
        save_end = true,
        callback = callbacks,
    )

    phis = [Array(hdg_2 * u) for u in sol.u]

    final_step = Int(round(sol.t[end] / dt))
    final_mass = sum(hdg_2 * sol.u[end])
    println(@sprintf("Final step %6d/%d | Relative mass: %.4f%%", final_step, steps, 100 * final_mass / m₀))

    return phis
end

############
### Run  ###
############

println("Starting tracer simulation (te=$te, dt=$dt, steps=$(ceil(Int64, te/dt)))...")

phis = run_tracer(to_device(phi_star_0), te, dt, kappa; saveat = saveat)

println("Simulation complete. $(length(phis)) snapshots saved.")

####################
### Plot Results ###
####################

# Animate all snapshots
save_path = joinpath(@__DIR__, "imgs", "Tracer", simname)
mkpath(save_path)

fig_init = plot_twoform(s, phis[1])
save(joinpath(save_path, "initial_n=$(nx_).png"), fig_init)

fig_final = plot_twoform(s, phis[end])
save(joinpath(save_path, "final_n=$(nx_).png"), fig_final)

fig_diff = plot_twoform(s, phis[end] - phis[1])
save(joinpath(save_path, "difference_n=$(nx_).png"), fig_diff)

fig_anim = Figure()
ax_anim  = CairoMakie.Axis(fig_anim[1, 1]; title = "Tracer", xlabel = "x", ylabel = "y")

interdps = interior(Val(2), dual_points(s), s)
xs = map(p -> p[1], interdps)
ys = map(p -> p[2], interdps)

phi_range = extrema(phis[1])
hm = CairoMakie.heatmap!(ax_anim, xs, ys, interior(Val(2), phis[1], s);
    colormap = :jet, colorrange = phi_range)
Colorbar(fig_anim[1, 2], hm)

CairoMakie.record(fig_anim, joinpath(save_path, "animation_n=$(nx_).mp4"), 1:length(phis); framerate = 15) do i
    hm[3] = interior(Val(2), phis[i], s)
end

println("Plots and animation saved to $save_path")
