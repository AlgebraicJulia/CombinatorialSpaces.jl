include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

simfile = "AlfvenWaves.jl"
include(joinpath(@__DIR__, "Sim_Files", simfile))

#####################
### DEC Operators ###
#####################

println("Constructing DEC operators...")

const dec_ops = build_dec_operators(MHDModel(), s)

println("DEC operators constructed.")

#######################################
### Plotting and Analysis Functions ###
#######################################

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics_Plotting.jl"))

initial_state = (
  U = dec_ops.hdg_1 * U_star_0,
  rho = dec_ops.hdg_2 * rho_star_0,
  B = dec_ops.hdg_1 * B_star_0,
)

plot_mhd_density(initial_state; suffix = "initial")
plot_mhd_velocity_components(initial_state; suffix = "initial")
plot_mhd_magnetic_components(initial_state; suffix = "initial")

##########################
### Physical Operators ###
##########################

# Pressure 2-form for the isothermal MHD closure: P = V_s² ρ
function momentum_pressure(::MHDModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  return p.V_s^2 * fields.rho
end

# Viscous diffusion: μ Δu  (returns nothing when μ = 0)
function momentum_diffusion(::MHDModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  p.μ == 0 && return nothing
  (; dlap1, dlap1_v) = dec_ops
  return p.μ * (dlap1 * fields.u + dlap1_v * fields.v)
end

# Lorentz body force: (1/μ₀) i_{B♯} dB
function momentum_body_forces(::MHDModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  curl_B = dec_ops.hdg_2 * dec_ops.d1 * state.B_star
  return (1 / p.μ₀) * wedge_product_dd(Val(0), Val(1), s, curl_B, dec_ops.hdg_1 * state.B_star)
end

# Magnetic induction equation (eq 24 from derivation):
# ∂⋆B/∂t = d(i_{u♯}(⋆B)) + ηΔB.
function magnetic_induction(
  s::UniformCubicalComplex2D,
  state::ComponentVector{FT},
  p::NamedTuple,
  dec_ops::NamedTuple,
) where FT <: AbstractFloat
  (; η) = p
  (; hdg_1, hdg_2, dual_d0, inv_hdg_1, dlap1) = dec_ops

  U = hdg_1 * state.U_star
  rho = hdg_2 * state.rho_star
  B_star = state.B_star

  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)
  v = interpolate_dp(Val(1), s, u)

  contraction = -hdg_2 * wedge_product(Val(1), Val(1), s, v, B_star)
  induction = dual_d0 * contraction

  diffusion = η * dlap1 * (hdg_1 * B_star)

  return inv_hdg_1 * (induction + diffusion)
end

build_saved_value_type(::MHDModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho, :B), Tuple{Vector{FT}, Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::MHDModel) = :MHD
checkpoint_field_names(::MHDModel) = (:U_star, :rho_star, :B_star)

function regularized_state(::MHDModel, u, context)
  return (
    U = Array(dec_ops.hdg_1 * u.U_star),
    rho = Array(dec_ops.hdg_2 * u.rho_star),
    B = Array(dec_ops.hdg_1 * u.B_star),
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

const rho_smoothing = smoothing_dual0(s, -c_smooth) * smoothing_dual0(s, c_smooth)

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

############################
### Context and Run       ###
############################

const simspec = ""
build_sim_context(::MHDModel, periodic_side) = (; simspec = simspec, dec_ops = dec_ops)

function rhs!(du, u, p, t)
  du.U_star .= momentum_conservation(MHDModel(), s, u, p, dec_ops)
  du.rho_star .= mass_continuity(s, u, dec_ops)
  du.B_star .= magnetic_induction(s, u, p, dec_ops)
  return nothing
end

######################
### Run Simulation ###
######################

state = ComponentArray(
  U_star = deepcopy(U_star_0),
  rho_star = deepcopy(rho_star_0),
  B_star = deepcopy(B_star_0),
)

println("Starting simulation...")
run_simulation(
  MHDModel(), state, rhs!, p;
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  full_periodic = full_periodic,
  periodic_left_right = periodic_left_right,
  periodic_top_bottom = periodic_top_bottom,
)
println("Simulation complete.")
