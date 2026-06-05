abstract type AbstractSimulationModel end

struct LMNSModel <: AbstractSimulationModel end
struct LMNSHModel <: AbstractSimulationModel end
struct MHDModel <: AbstractSimulationModel end

Base.@kwdef struct CallbackConfig{FT <: AbstractFloat}
  te::FT
  dt::FT
  saveat::Int
  checkpoint_at::Int
  start_step::Int = 1
  start_time::FT = zero(FT)
end

step_from_iter(cfg::CallbackConfig, iter::Int) = cfg.start_step - 1 + iter

# Required model-dispatched hooks.
build_saved_value_type(::AbstractSimulationModel, ::Type{FT}) where FT <: AbstractFloat =
  error("build_saved_value_type is not implemented for this model")

regularized_state(::AbstractSimulationModel, u, context) =
  error("regularized_state is not implemented for this model")

model_has_smoothing(::AbstractSimulationModel) = false
apply_smoothing!(::AbstractSimulationModel, integrator, context) = nothing

apply_periodic_prestep!(::AbstractSimulationModel, integrator, context) = nothing

###################################
### Shared Momentum Conservation ##
###################################

# Model-dispatched hook: returns the pressure 2-form used to build the pressure-gradient
# term `dd0 * momentum_pressure(...)` in the momentum equation.
# Each model must define a method for its own type.
# `fields` is a NamedTuple (U, rho, u, v, V) pre-computed by momentum_conservation.
momentum_pressure(::AbstractSimulationModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat =
  error("momentum_pressure not implemented for $(typeof(model))")

# Model-dispatched hook: returns an optional dual-1-form body force (e.g. gravity for
# LMNSHModel, Lorentz force for MHDModel).  Default: no body forces.
# `fields` is a NamedTuple (U, rho, u, v, V) pre-computed by momentum_conservation.
momentum_body_forces(::AbstractSimulationModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat = 
  nothing

# Model-dispatched hook: returns an optional dual-1-form viscous diffusion term.
# Return nothing for inviscid models; otherwise return the fully assembled diffusion
# dual-1-form (e.g. μ*(dlap1*u + dlap1_v*v)).  Default: no diffusion.
# `fields` is a NamedTuple (U, rho, u, v, V) pre-computed by momentum_conservation.
momentum_diffusion(::AbstractSimulationModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat = 
  nothing

# Shared compressible-fluid momentum conservation kernel.
#
# Implements:
#   ∂U/∂t = -L_{u♯}U + ½ρ d‖u‖² − U ∧ δu − d P(state) + μΔu + F_body
#
# Arguments
#   model    – simulation model tag (dispatches pressure and body-force hooks)
#   s        – the mesh (UniformCubicalComplex2D)
#   state    – ComponentVector with at least U_star and rho_star fields
#   p        – NamedTuple of physical parameters (must include p.mu)
#   dec_ops  – NamedTuple of pre-built DEC operators for the mesh
#
# Boundary-condition hooks `enforce_bc_v!(v)`, `enforce_bc_V!(V)`, and `enforce_bc_U!(result)`
# are called on the primal-edge velocity, momentum interpolants, and the momentum RHS result
# respectively; define them in the sim-case file for non-trivial BCs.
function momentum_conservation(
  model::AbstractSimulationModel,
  s::UniformCubicalComplex2D,
  state::ComponentVector{FT},
  p::NamedTuple,
  dec_ops::NamedTuple,
) where FT <: AbstractFloat

  (; hdg_1, hdg_2, wdg_dd_01, wdg_dd_01, inv_hdg_1, wdg_11, dd0,
  dd1, inv_hdg_0, wdg_01, d_beta, dcd_1, interp_dp_1) = dec_ops

  U   = hdg_1(state.U_star)
  rho = hdg_2(state.rho_star)

  # Velocity u = U / ρ (dual 1-form)
  u = wdg_dd_01(1 ./ rho, U)

  # Interpolate to primal edges for advection and diffusion
  v = interp_dp_1(u)
  V = interp_dp_1(U)

  # Boundary-condition hooks (no-op by default; defined per sim-case)
  enforce_bc_v!(v)
  enforce_bc_V!(V)

  # Bundle pre-computed fields for dispatch hooks to avoid redundant work.
  fields = (; U, rho, u, v, V)

  # U ∧ δu  (divergence / continuity coupling)
  div_term = wdg_dd_01(dcd_1(u), U)

  # Lie derivative  L_{u♯}U
  L_term = dd0(hdg_2(wdg_11(v, inv_hdg_1(U)))) + hdg_1(wdg_01(inv_hdg_0(dd1(U) + d_beta(V)), v))

  # ½ ρ d‖u‖²
  diff_norm_u = dd0(hdg_2(wdg_11(v, inv_hdg_1(u))))
  energy = 0.5 .* wdg_dd_01(rho, diff_norm_u)

  # Pressure gradient  d P
  diff_p = dd0(momentum_pressure(model, s, state, p, dec_ops, fields))

  rhs = .- div_term .- L_term .+ energy .- diff_p

  # Optional viscous diffusion
  diff = momentum_diffusion(model, s, state, p, dec_ops, fields)
  if diff !== nothing
    rhs .+= diff
  end

  # Optional body forces (gravity, Lorentz force, …)
  body = momentum_body_forces(model, s, state, p, dec_ops, fields)
  if body !== nothing
    rhs .+= body
  end

  result = -inv_hdg_1(rhs)
  enforce_bc_U!(result) # TODO: This was added because gravity force was non-zero through boundary
  return result
end

###############################
### Shared Mass Continuity  ###
###############################

# ∂ρ★/∂t = d₁ U★
# Identical across all compressible models: the primal exterior derivative of the
# momentum 1-form gives the rate of change of the dual density 2-form coefficient.
function mass_continuity(
  s::UniformCubicalComplex2D,
  state::ComponentVector{FT},
  dec_ops::NamedTuple,
) where FT <: AbstractFloat
  return dec_ops.d1(state.U_star)
end

############################
### Shared Run Entrypoint ###
############################

# Model-dispatched hook: return a NamedTuple of model-specific context fields that will
# be merged with the common fields (s, savepath, periodic) before being passed
# to run_with_model_callbacks.  Default: no extra fields.
build_sim_context(::AbstractSimulationModel, periodic) = (;)

# Top-level simulation runner.
#
# Arguments
#   model         – simulation model tag (e.g. MHDModel(), LMNSModel(), LMNSHModel())
#   initial_state – ComponentVector containing the initial field values
#   rhs!          – in-place RHS function with signature (du, u, p, t)
#   p             – physical parameters passed through to rhs! and callbacks
#
# Keyword arguments
#   te, dt           – end time and fixed timestep (required)
#   saveat           – save every N steps          (default 500)
#   checkpoint_at    – checkpoint every N steps    (default 10_000)
#   full_periodic,
#   periodic_left_right,
#   periodic_top_bottom – periodic boundary toggles (default false)
#
# The context visible to all model-dispatch hooks is:
#   (s, savepath, periodic, <fields from build_sim_context>)
# where s and savepath are expected globals in the simulation file.
function run_simulation(
  model::AbstractSimulationModel,
  s::UniformCubicalComplex2D,
  initial_state::ComponentVector{FT},
  rhs!::Function,
  p;
  dec_ops::NamedTuple,
  te::FT,
  dt::FT,
  saveat::Int = 500,
  checkpoint_at::Int = 10_000,
  periodic::GridSide,
  savepath::String
) where FT <: AbstractFloat

  cfg = CallbackConfig{FT}(
    te = te,
    dt = dt,
    saveat = saveat,
    checkpoint_at = checkpoint_at,
  )

  # Common context fields available to all models and callbacks
  # These are derived from global variables and the dec_ops, which is built from the mesh and shared across all models.
  common_context = (
    s = s,
    savepath = savepath,
    simspec = simspec,
    periodic = periodic,
    dec_ops = dec_ops,
  )
  model_context = build_sim_context(model, periodic)
  context = merge(common_context, model_context)

  println("Starting simulation...")

  run_with_model_callbacks(model, initial_state, rhs!, p, cfg; context = context)

  println("Simulation complete.")
  return nothing
end

function run_with_model_callbacks(
  model::AbstractSimulationModel,
  u0,
  rhs!,
  params,
  cfg::CallbackConfig{FT};
  context = nothing,
) where FT <: AbstractFloat

  println("Setting up callbacks...")
  regular_save_values = SavedValues(FT, build_saved_value_type(model, FT),)

  regular_state_cb = SavingCallback(
    (u, t, integrator) -> regularized_state(model, u, context),
    regular_save_values;
    saveat = cfg.saveat * cfg.dt,
    save_start = false, # TODO: Check this is saving all the right data
    save_end = true,
  )

  println("Regularized saving callbacks configured.")

  function smoothing_condition(u, t, integrator)
    integrator.iter > 0 || return false
    return model_has_smoothing(model)
  end

  function smoothing_affect!(integrator)
    apply_smoothing!(model, integrator, context)
    return nothing
  end

  println("Smoothing callbacks configured.")

  function save_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(cfg, integrator.iter)
    return step % cfg.saveat == 0
  end

  # TODO: Might add hook for model to print diagnostics
  function save_affect(integrator)
    progress = integrator.t / max(cfg.te, eps(typeof(cfg.te)))
    println("Loading simulation results: $(progress * 100)%")
    println("Saving variables at current step...")

    save_step = step_from_iter(cfg, integrator.iter)
  
    jldopen(savedata_filepath, "a+") do file
      for (var, val) in pairs(only(regular_save_values.saveval))
        file["$(string(save_step))/$(string(var))"] = Float32.(val)
      end
    end

    println("Saving complete.")
    println("-----")
    flush(stdout)

    empty!(regular_save_values.t)
    empty!(regular_save_values.saveval)

    return nothing
  end

  println("Save callbacks configured.")

  function checkpoint_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(cfg, integrator.iter)
    return step % cfg.checkpoint_at == 0
  end

  function checkpoint_affect!(integrator)
    println("Generating checkpoint file...")

    step = step_from_iter(cfg, integrator.iter)
    #TODO: Change keys to better names
    jldsave(joinpath(savepath, "checkpoint_$(step).jld2"); t=only(regular_save_values.t), val=only(regular_save_values.saveval))

    println("Checkpoint file saved.")

    return nothing
  end

  println("Checkpoint callbacks configured.")

  callback_items = Any[]
  if hasproperty(context, :periodic) && context.periodic !== nothing
    function periodic_prestep_affect!(integrator)
      apply_periodic_prestep!(model, integrator, context)
      return nothing
    end

    push!(callback_items, PeriodicCallback(periodic_prestep_affect!, cfg.dt; initial_affect = true, final_affect = false))
  end

  if model_has_smoothing(model)
    push!(callback_items, DiscreteCallback(smoothing_condition, smoothing_affect!; save_positions = (false, false)))
  end
  push!(callback_items, regular_state_cb)
  push!(callback_items, DiscreteCallback(checkpoint_condition, checkpoint_affect!; save_positions = (false, false)))
  push!(callback_items, DiscreteCallback(save_condition, save_affect; save_positions = (false, false)))

  callbacks = CallbackSet(callback_items...)

  println("All callbacks configured. Setting up and starting the solver...")

  dec_ops = context.dec_ops

  prob = ODEProblem(rhs!, u0, (cfg.start_time, cfg.te), params)
  solve(
    prob,
    SSPRK33();
    dt = cfg.dt,
    adaptive = false,
    save_everystep = false,
    save_start = false,
    save_end = false,
    save_on = false,
    callback = callbacks,
    dense = false,
  )

  return nothing
end
