include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

sim = "Taylor_Vortices"

const config = TOML.parsefile("Sim_Files/$(sim)_Sim.toml")

const simspec = config["Metadata"]["simspec"]
const savepath = config["Metadata"]["savepath"]

println("Running simulation: $(config["Metadata"]["name"])")
println("Given simulation specification is: $(simspec)")
println("Saving outputs at: $(savepath)")

mkpath(savepath)

const lx_ = config["Mesh"]["lx"]
const ly_ = config["Mesh"]["ly"]
const nx_ = config["Mesh"]["nx"]
const ny_ = config["Mesh"]["ny"]
const halo_x = config["Mesh"]["halo_x"]
const halo_y = config["Mesh"]["halo_y"]

const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_, halo_x = halo_x, halo_y = halo_y)

const Re = config["Physics"]["Re"]
const p = (mu = 1 / Re,) # μ

const te = config["Simulation"]["te"]
const dt = config["Simulation"]["dt"]

const full_periodic = config["Simulation"]["full_periodic"]

if full_periodic
  const periodic_left_right = true
  const periodic_top_bottom = true
else
  const periodic_left_right = config["Simulation"]["periodic_left_right"]
  const periodic_top_bottom = config["Simulation"]["periodic_top_bottom"]
end

@assert !periodic_left_right || halo_x > 0
@assert !periodic_top_bottom || halo_y > 0

const saveat = floor(Int64, config["Simulation"]["savetime"] / dt)
const checkpoint_every_saveat = config["Simulation"]["checkpoint_every_savetime"]
const checkpoint_at = saveat * checkpoint_every_saveat

include(joinpath(@__DIR__, "Sim_Files", "$(sim)_Sim.jl"))

@isdefined(u0) || error("Initial condition not defined properly, please assign to variable \"u0\"")
# TODO: Are these checks working?
# @isdefined(enforce_bc_u!) || error("Please define boundary conditions for dual velocity (u), can be trivial")
# @isdefined(enforce_bc_U!) || error("Please define boundary conditions for dual momentum (U), can be trivial")
# @isdefined(enforce_bc_v!) || error("Please define boundary conditions for primal velocity (v), can be trivial")
# @isdefined(enforce_bc_V!) || error("Please define boundary conditions for primal momentum (v), can be trivial")


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
  s = context.s
  periodic_side = context.periodic_side
  set_periodic!(integrator.u.U_star, Val(1), s, periodic_side)
  set_periodic!(integrator.u.rho_star, Val(2), s, periodic_side)
  return nothing
end

# TODO: Add more plotting options later
# TODO: Upstream into physics plotting
function parse_plotting_config(config::Dict)
  map(config["Plotting"]["imgs"]) do img
    if img == "Vorticity" return plot_vorticity
    elseif img == "Density" return plot_density
    elseif img == "Momentum_Components" return plot_momentum_components
    elseif img == "Momentum_Magnitude" return plot_momentum_magnitude
    end
  end
end

function run_checkpoint_outputs!(::LMNSModel, regular_save_values::SavedValues, step::Int, checkpoint_t::AbstractFloat, cfg::CallbackConfig, context)
  println("Checkpoint saved at step: $(step)")
  println("Generating plots and mp4 for checkpoint...")

  state_end = regular_save_values.saveval[end]
  time = @sprintf("%.6f", checkpoint_t)
  file_end = "$(context.simspec)_t=$(time)"

  # TODO: Add this to context?
  required_plots = parse_plotting_config(config)
  map(plot -> plot(context.s, state_end, savepath, file_end, time), required_plots)

  if config["Plotting"]["create_mp4"]
    create_mp4(
      LMNSModel(),
      regular_save_values,
      file_end;
      records = max(1, div(cfg.checkpoint_at, cfg.saveat)),
    )
  end

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
