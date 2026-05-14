include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

# simfile = "Kelvin-Helmholtz_Sim.jl"
simfile = "Thermal_Bubble_Sim.jl"
# simfile = "RTI_Sim.jl"

# Test case parameters for lid-driven cavity flow at different Reynolds numbers
include(joinpath(@__DIR__, "Sim_Files", simfile))

#####################
### DEC Operators ###
#####################
println("Constructing DEC operators...")

println("DEC operators constructed. Moving to device...")

const dec_ops = build_dec_operators(
  LMNSHModel(),
  s,
  to_device;
  smoothing_coefficients = (
    rho = r_smooth_constant,
    Theta = T_smooth_constant,
  ),
)

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
const dual_δ1 = dec_ops.dual_delta1
const dΔ0 = dec_ops.dlap0
const dΔ1 = dec_ops.dlap1
const dΔ1_V = dec_ops.dlap1_v

#######################################
### Plotting and Analysis Functions ###
#######################################

println("Setting up plotting and analysis functions...")

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics_Plotting.jl"))

##########################
### Physical Operators ###
##########################

println("Setting up physical operators...")

# Pressure 2-form from potential temperature: P = pressure(Θ)
momentum_pressure(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat =
  pressure(dec_ops.hdg_2 * state.Theta_star)

# Viscous diffusion: μ Δu  (returns nothing when μ = 0)
function momentum_diffusion(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  p.mu == 0 && return nothing
  return p.mu * (dec_ops.dlap1 * fields.u + dec_ops.dlap1_v * fields.v)
end

# Gravity body force: ρ g  (only when use_gravity is active in the sim case)
function momentum_body_forces(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  use_gravity || return nothing
  return wedge_product_dd(Val(0), Val(1), s, fields.rho, g_dual)
end

function potential_temperature_continuity(state::ComponentVector{FT}, p::NamedTuple) where FT <: AbstractFloat
  U = hdg_1 * state.U_star
  rho = hdg_2 * state.rho_star
  Theta = hdg_2 * state.Theta_star

  kappa = p[:kappa]

  # Velocity from momentum/density
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  vX, vY = sharp_dd(s, u)
  v = flat_dp(s, vX, vY)

  theta = Theta ./ rho

  # -Θ * (δu) creation/divergence term
  temperature_creation = Theta .* (dual_δ1 * u)

  # L(u, Θ) advection: dual_d0(Theta) → dual 1-form, convert to primal, wedge with v, back to dual
  temperature_advection = hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * dual_d0 * Theta)

  # κΔθ thermal diffusion
  temperature_diffusion = kappa * (dΔ0 * theta)

  return inv_hdg_2 * (-temperature_creation - temperature_advection + temperature_diffusion)
end

build_saved_value_type(::LMNSHModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho, :Theta), Tuple{Vector{FT}, Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::LMNSHModel) = :LMNSH
checkpoint_field_names(::LMNSHModel) = (:U_star, :rho_star, :Theta_star)

function regularized_state(::LMNSHModel, u, context)
  return (
    U = Array(hdg_1 * u.U_star),
    rho = Array(hdg_2 * u.rho_star),
    Theta = Array(hdg_2 * u.Theta_star),
  )
end

function progress_reference(::LMNSHModel, u0, context)
  return (
    m0 = sum(interior(Val(2), Array(u0.rho_star), context.s)),
    E0 = sum(interior(Val(2), Array(u0.Theta_star), context.s)),
  )
end

function log_progress!(::LMNSHModel, integrator, refs, cfg::CallbackConfig, context)
  progress = integrator.t / max(cfg.te, eps(typeof(cfg.te)))
  println("Loading simulation results: $(progress * 100)%")
  println("Relative mass is : $((sum(interior(Val(2), Array(integrator.u.rho_star), context.s)) / refs.m0) * 100)%")
  println("Relative energy is : $((sum(interior(Val(2), Array(integrator.u.Theta_star), context.s)) / refs.E0) * 100)%")

  # mem_before = CUDA.used_memory()

  # GC.gc(true)
  # GC.gc(true)                # second pass: drain finalizer thread queue
  # CUDA.synchronize()         # drain the CUDA stream — makes all cudaFreeAsync complete
  # CUDA.reclaim()             # now the pool actually knows what's free

  # mem_after = CUDA.used_memory()
  # println("GPU live: $(round(mem_before / 1024^2, digits=1)) MiB before GC → $(round(mem_after / 1024^2, digits=1)) MiB after")

  println("-----")
  flush(stdout)

  return nothing
end

model_has_periodic_prestep(::LMNSHModel, context) =
  hasproperty(context, :periodic_side) && context.periodic_side !== nothing

function apply_periodic_prestep!(::LMNSHModel, integrator, context)
  periodic_side = context.periodic_side
  set_periodic!(integrator.u.U_star, Val(1), context.s, periodic_side)
  set_periodic!(integrator.u.rho_star, Val(2), context.s, periodic_side)
  set_periodic!(integrator.u.Theta_star, Val(2), context.s, periodic_side)
  return nothing
end

model_has_smoothing(::LMNSHModel) = true

const tmp_rho_smooth_buf = to_device(zeros(Float64, nquads(s)))
const tmp_Theta_smooth_buf = to_device(zeros(Float64, nquads(s)))

function apply_smoothing!(::LMNSHModel, integrator, context)
  mul!(tmp_rho_smooth_buf, context.dec_ops.smoothing.rho, integrator.u.rho_star)
  integrator.u.rho_star .= tmp_rho_smooth_buf

  mul!(tmp_Theta_smooth_buf, context.dec_ops.smoothing.Theta, integrator.u.Theta_star)
  integrator.u.Theta_star .= tmp_Theta_smooth_buf
  return nothing
end



function run_checkpoint_outputs!(::LMNSHModel, regular_save_values::SavedValues, step::Int, checkpoint_t::AbstractFloat, cfg::CallbackConfig, context)
  println("Checkpoint saved at step: $(step)")
  println("Generating plots and mp4 for checkpoint...")

  state_end = regular_save_values.saveval[end]
  time = @sprintf("%.6f", checkpoint_t)
  file_end = "$(context.simspec)_t=$(time)"

  plot_vorticity(context.s, state_end, file_end, time)
  plot_density(context.s, state_end, file_end, time)
  # plot_momentum_magnitude(context.s, state_end, file_end, time)
  plot_pressure(context.s, state_end, file_end, time)
  plot_momentum_components(context.s, state_end, file_end, time)

  create_mp4(
    LMNSHModel(),
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

build_sim_context(::LMNSHModel, periodic_side) = (; simspec = simspec, dec_ops = dec_ops)

function rhs!(du, u, p_rhs, t)
  du.U_star .= momentum_conservation(LMNSHModel(), s, u, p_rhs, dec_ops)
  du.Theta_star .= potential_temperature_continuity(u, p_rhs)
  du.rho_star .= mass_continuity(s, u, dec_ops)
  return nothing
end

######################
### Run Simulation  ###
######################

println("Starting simulation...")

initial_state = ComponentArray(
  U_star = to_device(U_star_0),
  rho_star = to_device(rho_star_0),
  Theta_star = to_device(Theta_star_0),
)

run_simulation(
  LMNSHModel(), initial_state, rhs!, p;
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  full_periodic = full_periodic,
  periodic_left_right = periodic_left_right,
  periodic_top_bottom = periodic_top_bottom,
)
println("Simulation complete.")
