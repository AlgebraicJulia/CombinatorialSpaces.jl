using CairoMakie
using Distributions
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays
using CUDA
using CUDA.CUSPARSE
# using OrdinaryDiffEq

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

const lx_ = ly_ = 1.0
const nx_ = ny_ = 513 # 129, 257, 513 for higher resolution tests (requires more memory and time)
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

simname = "Diagonal" # "Diagonal", "Stretch", "Rotate", "CircularVortex", "ReversedVortex"
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
)
    phi_star      = deepcopy(phi_star_0)
    phi_star_1    = similar(phi_star)
    phi_star_2    = similar(phi_star)
    phi_star_full = similar(phi_star)

    steps = ceil(Int64, te / dt)

    phis = [Array(hdg_2 * phi_star_0)]
    m₀   = sum(phis[1])

    error_encountered = false

    for step in 1:steps

        t = step * dt

        # Stage 1
        X = generate_X(s, t, te)
        Y = generate_Y(s, t, te)
        u = flat_dd(s, X, Y)
        v = flat_dp(s, X, Y)

        phi_star_1    .= phi_star .+ dt .* tracer_continuity(u, v, phi_star, kappa)
        set_periodic!(phi_star_1, Val(2), s, ALL)

        # Stage 2
        X = generate_X(s, t + 0.25 * dt, te)
        Y = generate_Y(s, t + 0.25 * dt, te)
        u = flat_dd(s, X, Y)
        v = flat_dp(s, X, Y)

        phi_star_2    .= 0.75 .* phi_star .+ 0.25 .* phi_star_1 .+
                         0.25 .* dt .* tracer_continuity(u, v, phi_star_1, kappa)
        set_periodic!(phi_star_2, Val(2), s, ALL)

        # Stage 3
        X = generate_X(s, t + (2/3) * dt, te)
        Y = generate_Y(s, t + (2/3) * dt, te)
        u = flat_dd(s, X, Y)
        v = flat_dp(s, X, Y)

        phi_star_full .= (1/3) .* phi_star .+ (2/3) .* phi_star_2 .+
                         (2/3) .* dt .* tracer_continuity(u, v, phi_star_2, kappa)
        set_periodic!(phi_star_full, Val(2), s, ALL)

        phi_star .= phi_star_full

        if any(isnan, phi_star)
            println("Warning: NaN detected at step $step")
            error_encountered = true
        elseif any(isinf, phi_star)
            println("Warning: Inf detected at step $step")
            error_encountered = true
        end

        if step % saveat == 0 || step == steps || error_encountered
            snap = hdg_2 * phi_star
            push!(phis, Array(snap))
            m = sum(snap)
            println(@sprintf("Step %6d/%d | Relative mass: %.4f%%", step, steps, 100 * m / m₀))
            flush(stdout)
        end

        error_encountered && break
    end

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
