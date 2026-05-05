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

const lx_ = ly_ = 6.0
const nx_ = ny_ = 151
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_; halo_x = 5, halo_y = 5)

const simname = "Alfven_Wave"
const save_path = joinpath(@__DIR__, "imgs", "MHD")
mkpath(save_path)

#####################
### DEC Operators ###
#####################

println("Constructing DEC operators...")

const dec_ops = build_dec_operators(MHDModel(), s)

const d1 = dec_ops.d1
const dual_d0 = dec_ops.dual_d0
const dual_d1 = dec_ops.dual_d1
const hdg_1 = dec_ops.hdg_1
const hdg_2 = dec_ops.hdg_2
const inv_hdg_0 = dec_ops.inv_hdg_0
const inv_hdg_1 = dec_ops.inv_hdg_1
const inv_hdg_2 = dec_ops.inv_hdg_2
const dual_δ1 = dec_ops.dual_delta1
const dΔ1 = dec_ops.dlap1

println("DEC operators constructed.")

#####################
### MHD Constants ###
#####################

const μ₀ = 4π * 1e-7  # Vacuum permeability
const V_s = 1.0       # Sound speed

#######################################
### Plotting and Analysis Functions ###
#######################################

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics_Plotting.jl"))

##########################
### Initial Conditions ###
##########################

function initialize_alfven_wave(s::UniformCubicalComplex2D, lx::Float64)
  ρ₀ = 1.0
  B₀ = 1.0
  v_A = B₀ / sqrt(μ₀ * ρ₀)
  k_wave = 2π / lx
  δu = 1e-4 * v_A
  δB = δu * sqrt(μ₀ * ρ₀)

  rho_0 = ρ₀ * ones(nquads(s))
  rho_star_0 = inv_hdg_2 * rho_0

  U_star_0 = zeros(ne(s))
  for e in (nxedges(s) + 1):ne(s)
    x, _ = edge_to_coord(s, e)
    x_pos = (x - 1) * dx(s)
    U_star_0[e] = ρ₀ * δu * sin(k_wave * x_pos) * dy(s)
  end

  B_star_0 = zeros(ne(s))
  for e in 1:nxedges(s)
    B_star_0[e] = B₀ * dx(s)
  end
  for e in (nxedges(s) + 1):ne(s)
    x, _ = edge_to_coord(s, e)
    x_pos = (x - 1) * dx(s)
    B_star_0[e] = -δB * sin(k_wave * x_pos) * dy(s)
  end

  return U_star_0, rho_star_0, B_star_0
end

const U_star_0, rho_star_0, B_star_0 = initialize_alfven_wave(s, lx_)

save(joinpath(save_path, "InitialDensity.png"), plot_twoform(s, hdg_2 * rho_star_0))
save(joinpath(save_path, "InitialVelocityComponents.png"), plot_xy_oneform(s, hdg_1 * U_star_0))
save(joinpath(save_path, "InitialMagneticFieldComponents.png"), plot_xy_oneform(s, hdg_1 * B_star_0))

##########################
### Physical Operators ###
##########################

# Momentum equation (eq 23 from derivation):
# ∂U/∂t = -L_{u♯}U + 1/2 ρm ∧ di_{u♯}u - U ∧ δ(u) - V²s ∧ dρm + (1/μ₀)i_{B♯}dB + μΔu
function momentum_mhd(state::ComponentVector{FT}, p::NamedTuple, periodic_side::Union{Nothing, GridSide}=nothing) where FT <: AbstractFloat
  (; μ, V_s, μ₀) = p

  U = hdg_1 * state.U_star
  rho = hdg_2 * state.rho_star
  B_star = state.B_star

  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)
  v = interpolate_dp(Val(1), s, u)

  div_term = wedge_product_dd(Val(1), Val(0), s, U, dual_δ1 * u)

  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(1), Val(0), s, v, inv_hdg_0 * dual_d1 * U)

  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 * wedge_product_dd(Val(1), Val(0), s, diff_norm_u, rho)

  diff_p = V_s^2 * dual_d0 * rho

  curl_B = hdg_2 * d1 * B_star
  lorentz = (1 / μ₀) * wedge_product_dd(Val(0), Val(1), s, curl_B, hdg_1 * B_star)

  lap_term = μ * dΔ1 * u

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term + lorentz)
end

# Magnetic induction equation (eq 24 from derivation):
# ∂⋆B/∂t = d(i_{u♯}(⋆B)) + ηΔB.
function magnetic_induction(state::ComponentVector{FT}, p::NamedTuple, periodic_side::Union{Nothing, GridSide}=nothing) where FT <: AbstractFloat
  (; η) = p

  U = hdg_1 * state.U_star
  rho = hdg_2 * state.rho_star
  B_star = state.B_star

  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)
  v = interpolate_dp(Val(1), s, u)

  contraction = -hdg_2 * wedge_product(Val(1), Val(1), s, v, B_star)
  induction = dual_d0 * contraction

  diffusion = η * dΔ1 * (hdg_1 * B_star)

  return inv_hdg_1 * (induction + diffusion)
end

function mass_continuity(state::ComponentVector{FT}, periodic_side::Union{Nothing, GridSide}=nothing) where FT <: AbstractFloat
  return d1 * state.U_star
end

build_saved_value_type(::MHDModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho, :B), Tuple{Vector{FT}, Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::MHDModel) = :MHD
checkpoint_field_names(::MHDModel) = (:U_star, :rho_star, :B_star)

function regularized_state(::MHDModel, u, context)
  return (
    U = Array(hdg_1 * u.U_star),
    rho = Array(hdg_2 * u.rho_star),
    B = Array(hdg_1 * u.B_star),
  )
end

function progress_reference(::MHDModel, u0, context)
  return (
    m0 = sum(interior(Val(2), Array(u0.rho_star), context.s)),
  )
end

function log_progress!(::MHDModel, integrator, refs, cfg::CallbackConfig, context)
  progress = integrator.t / max(cfg.te, eps(typeof(cfg.te)))
  println("Loading simulation results: $(progress * 100)%")
  println("Relative mass is : $((sum(interior(Val(2), Array(integrator.u.rho_star), context.s)) / refs.m0) * 100)%")
  println("-----")
  flush(stdout)

  return nothing
end

model_has_periodic_prestep(::MHDModel, context) =
  hasproperty(context, :periodic_side) && context.periodic_side !== nothing

function apply_periodic_prestep!(::MHDModel, integrator, context)
  periodic_side = context.periodic_side
  set_periodic!(integrator.u.U_star, Val(1), context.s, periodic_side)
  set_periodic!(integrator.u.rho_star, Val(2), context.s, periodic_side)
  set_periodic!(integrator.u.B_star, Val(1), context.s, periodic_side)
  return nothing
end

model_has_smoothing(::MHDModel) = true

function apply_smoothing!(::MHDModel, integrator, context)
  integrator.u.rho_star .= rho_smoothing * integrator.u.rho_star
  return nothing
end

function run_checkpoint_outputs!(::MHDModel, regular_save_values::SavedValues, step::Int, checkpoint_t::AbstractFloat, cfg::CallbackConfig, context)
  println("Checkpoint saved at step: $(step)")
  println("Generating plots and mp4 for checkpoint...")

  suffix = "step_$(step)"
  save_mhd_diagnostics(
    regular_save_values;
    suffix = suffix,
    records = max(1, div(cfg.checkpoint_at, cfg.saveat)),
  )

  println("Plots and mp4 generated for checkpoint at step: $(step)")
  println("-----")
  flush(stdout)

  return nothing
end

#########################
### Smoothing Helpers ###
#########################

paired_smoothing_dual0(s::UniformCubicalComplex2D, c_smooth::Float64) =
  smoothing_dual0(s, -c_smooth) * smoothing_dual0(s, c_smooth)

const c_smooth = 0.2
const rho_smoothing = paired_smoothing_dual0(s, c_smooth)

######################
### Time Integrator ###
######################

function run_compressible_mhd(
  U_star_0::AbstractVector{Float64},
  rho_star_0::AbstractVector{Float64},
  B_star_0::AbstractVector{Float64},
  te::Float64,
  dt::Float64,
  p::NamedTuple;
  saveat::Int = 500,
  checkpoint_at::Int = 10_000,
  full_periodic::Bool = true,
  periodic_left_right::Bool = true,
  periodic_top_bottom::Bool = true,
)
  state = ComponentArray(
    U_star = deepcopy(U_star_0),
    rho_star = deepcopy(rho_star_0),
    B_star = deepcopy(B_star_0),
  )

  periodic_side = periodic_side_selection(full_periodic, periodic_left_right, periodic_top_bottom)

  function rhs!(du, u, p_rhs, t)
    du.U_star .= momentum_mhd(u, p_rhs, periodic_side)
    du.rho_star .= mass_continuity(u, periodic_side)
    du.B_star .= magnetic_induction(u, p_rhs, periodic_side)

    return nothing
  end

  cfg = CallbackConfig{Float64}(
    te = te,
    dt = dt,
    saveat = saveat,
    checkpoint_at = checkpoint_at,
  )

  context = (
    s = s,
    save_path = save_path,
    periodic_side = periodic_side,
  )

  run_with_model_callbacks(
    MHDModel(),
    state,
    rhs!,
    p,
    cfg;
    context = context,
  )

  return nothing
end

######################
### Run Simulation ###
######################

const μ = 1e-3
const η = 0.0
const p = (
  μ = μ,
  η = η,
  V_s = V_s,
  μ₀ = μ₀,
)

const te = 2e-2
const dt = 5e-5
const saveat = 2
const checkpoint_at = 100

println("Starting simulation...")
run_compressible_mhd(U_star_0, rho_star_0, B_star_0, te, dt, p; saveat = saveat, checkpoint_at = checkpoint_at)
println("Simulation complete.")
