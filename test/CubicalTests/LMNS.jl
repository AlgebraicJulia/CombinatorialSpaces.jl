include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

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

# Pressure 2-form for the isothermal closure: P = Rθᵣ ρ
momentum_pressure(::LMNSModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat =
  pressure_same_theta(fields.rho)

# Viscous diffusion: μ Δu  (returns nothing when μ = 0)
function momentum_diffusion(::LMNSModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  p.mu == 0 && return nothing
  return p.mu * (dec_ops.dlap1 * fields.u + dec_ops.dlap1_v * fields.v)
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

############################
### Context and Run       ###
############################

build_sim_context(::LMNSModel, periodic_side) = (; simspec = simspec)

function rhs!(du, u, p_rhs, t)
  du.U_star .= momentum_conservation(LMNSModel(), s, u, p_rhs, dec_ops)
  du.rho_star .= mass_continuity(s, u, dec_ops)
  return nothing
end

######################
### Run Simulation  ###
######################

println("Starting simulation...")
run_simulation(
  LMNSModel(), to_device(u0), rhs!, p;
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  full_periodic = full_periodic,
  periodic_left_right = periodic_left_right,
  periodic_top_bottom = periodic_top_bottom,
)
println("Simulation complete.")
