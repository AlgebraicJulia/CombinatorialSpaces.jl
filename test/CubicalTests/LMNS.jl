using Test
using CairoMakie
using LinearAlgebra
using Printf
using SparseArrays
using JLD2
using ComponentArrays
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks

include(joinpath(@__DIR__, "..", "..", "src", "CubicalCode", "UniformDEC.jl"))
include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Harness.jl"))
include(joinpath(@__DIR__, "LMNS_Helpers", "DEC_Operators.jl"))
include(joinpath(@__DIR__, "LMNS_Helpers", "CUDA_Init.jl"))

#################################
### Simulation Initialization ###
#################################

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics.jl"))

simfile = "Taylor_Vortices_Sim.jl"

# Test case parameters for lid-driven cavity flow at different Reynolds numbers
include(joinpath(@__DIR__, "Sim_Files", simfile))

#####################
### DEC Operators ###
#####################

const dec_ops = build_dec_operators(LMNSModel(), s, to_device)

const d0 = dec_ops.d0
const d1 = dec_ops.d1
const dual_d0 = dec_ops.dual_d0
const dual_d1 = dec_ops.dual_d1
const d_beta = dec_ops.d_beta
const hdg_1 = dec_ops.hdg_1
const hdg_2 = dec_ops.hdg_2
const inv_hdg_0 = dec_ops.inv_hdg_0
const inv_hdg_1 = dec_ops.inv_hdg_1
const inv_hdg_2 = dec_ops.inv_hdg_2
const δ1 = dec_ops.delta1
const dual_δ1 = dec_ops.dual_delta1
const dΔ1 = dec_ops.dlap1
const dΔ1_V = dec_ops.dlap1_v
const dd0_h2 = dec_ops.dd0_h2
const ih0_dd1 = dec_ops.ih0_dd1
const ih0_db = dec_ops.ih0_db
const smoothing = dec_ops.smoothing

#######################################
### Plotting and Analysis Functions ###
#######################################

println("Setting up plotting and analysis functions...")

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics_Plotting.jl"))

##########################
### Physical Operators ###
##########################

println("Setting up physical operators...")

# TODO: Precompute some matrix products to speed up the loop
function momentum_continuity(s::UniformCubicalComplex2D{FT}, state::ComponentVector{FT}, p::Dict{Symbol, FT}, periodic_side::Union{Nothing, GridSide}=nothing) where FT
  U_star = state.U_star
  rho_star = state.rho_star

  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  mu = p[:mu]

  # Compute velocity u from momentum U and density rho
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  vX, vY = sharp_dd(s, u)
  v = flat_dp(s, vX, vY)

  VX, VY = sharp_dd(s, U)
  V = flat_dp(s, VX, VY)

  # TODO: Enforce conditions on V as well?
  enforce_bc_v!(v)
  enforce_bc_V!(V)

  # U ∧ δu
  div_term = wedge_product_dd(Val(0), Val(1), s, dual_δ1 * u, U)

  # L(u, U)
  L_term = dd0_h2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(0), Val(1), s, ih0_dd1 * U + ih0_db * V, v)

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dd0_h2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 .* wedge_product_dd(Val(0), Val(1), s, rho, diff_norm_u)

  # dP, P = κρ
  diff_p = dual_d0 * pressure_same_theta(rho)

  # muΔu
  lap_term = mu * (dΔ1 * u + dΔ1_V * v)

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term)
end

function mass_continuity(s::UniformCubicalComplex2D{FT}, state::ComponentVector{FT}, periodic_side::Union{Nothing, GridSide}=nothing) where FT <: AbstractFloat
  U_star = state.U_star

  return d1 * U_star
end

build_saved_value_type(::LMNSModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho), Tuple{Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::LMNSModel) = :LMNS
checkpoint_field_names(::LMNSModel) = (:U_star, :rho_star)

function regularized_state(::LMNSModel, u, context)
  return (
    U = Array(hdg_1 * u.U_star),
    rho = Array(hdg_2 * u.rho_star),
  )
end

function progress_reference(::LMNSModel, u0, context)
  return (
    m0 = sum(interior(Val(2), Array(u0.rho_star), context.s)),
  )
end

function log_progress!(::LMNSModel, integrator, refs, cfg::CallbackConfig, context)
  progress = integrator.t / max(cfg.te, eps(typeof(cfg.te)))
  println("Loading simulation results: $(progress * 100)%")
  println("Relative mass is : $((sum(interior(Val(2), Array(integrator.u.rho_star), context.s)) / refs.m0) * 100)%")
  println("-----")
  flush(stdout)

  return nothing
end

model_has_periodic_prestep(::LMNSModel, context) =
  hasproperty(context, :periodic_side) && context.periodic_side !== nothing

function apply_periodic_prestep!(::LMNSModel, integrator, context)
  periodic_side = context.periodic_side
  set_periodic!(integrator.u.U_star, Val(1), context.s, periodic_side)
  set_periodic!(integrator.u.rho_star, Val(2), context.s, periodic_side)
  return nothing
end

function run_checkpoint_outputs!(::LMNSModel, regular_save_values::SavedValues, step::Int, checkpoint_t::AbstractFloat, cfg::CallbackConfig, context)
  println("Checkpoint saved at step: $(step)")
  println("Generating plots and mp4 for checkpoint...")

  state_end = regular_save_values.saveval[end]
  time = @sprintf("%.6f", checkpoint_t)
  file_end = "$(context.simspec)_t=$(time)"

  plot_vorticity(context.s, state_end, file_end, time)
  plot_density(context.s, state_end, file_end, time)
  plot_momentum_components(context.s, state_end, file_end, time)
  plot_momentum_magnitude(context.s, state_end, file_end, time)

  create_mp4(
    LMNSModel(),
    regular_save_values,
    file_end;
    records = max(1, div(cfg.checkpoint_at, cfg.saveat)),
  )

  println("Plots and mp4 generated for checkpoint at step: $(step)")
  println("-----")
  flush(stdout)

  return nothing
end

function run_compressible_ns(u0::ComponentVector{FT}, te::FT, dt::FT, p::Dict{Symbol, FT}; saveat::Int=500, checkpoint_at::Int=10_000, full_periodic::Bool=false, periodic_left_right::Bool=false, periodic_top_bottom::Bool=false) where FT <: AbstractFloat

  u = deepcopy(u0)

  periodic_side = periodic_side_selection(full_periodic, periodic_left_right, periodic_top_bottom)

  function rhs!(du, u, p_rhs, t)
    du.U_star .= momentum_continuity(s, u, p_rhs, periodic_side)
    du.rho_star .= mass_continuity(s, u, periodic_side)

    return nothing
  end

  cfg = CallbackConfig{FT}(
    te = te,
    dt = dt,
    saveat = saveat,
    checkpoint_at = checkpoint_at,
  )

  context = (
    s = s,
    save_path = save_path,
    simspec = simspec,
    periodic_side = periodic_side,
  )

  run_with_model_callbacks(
    LMNSModel(),
    u,
    rhs!,
    p,
    cfg;
    context = context,
  )

  return nothing
end

println("Starting simulation...")

run_compressible_ns(to_device(u0),
  te, dt, p;
  saveat=saveat,
  checkpoint_at=checkpoint_at,
  full_periodic=full_periodic,
  periodic_left_right=periodic_left_right,
  periodic_top_bottom=periodic_top_bottom,
);

println("Simulation complete.")
