include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

const sim = "Kelvin-Helmholtz"

const config_filepath = joinpath(@__DIR__, "Sim_Files", "$(sim)_Sim.toml")
const sim_filepath = joinpath(@__DIR__, "Sim_Files", "$(sim)_Sim.jl")

const config = TOML.parsefile(config_filepath)

const simspec = config["Metadata"]["simspec"]
const savepath = config["Metadata"]["savepath"]

println("Running simulation: $(config["Metadata"]["name"])")
println("Given simulation specification is: $(simspec)")
println("Saving outputs at: $(savepath)")

mkpath(savepath)
cp(config_filepath, joinpath(savepath, "$(sim)_Sim.toml"); force=true)

const savedata_filepath = joinpath(savepath, "savedata.jld2")
jldopen(savedata_filepath, "w") do file end # Create empty JLD2 file to store results in later

const lx_ = config["Mesh"]["lx"]
const ly_ = config["Mesh"]["ly"]
const nx_ = config["Mesh"]["nx"]
const ny_ = config["Mesh"]["ny"]
const halo_x = config["Mesh"]["halo_x"]
const halo_y = config["Mesh"]["halo_y"]

const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_, halo_x = halo_x, halo_y = halo_y)

const Re = config["Physics"]["Re"]
const Pr = config["Physics"]["Pr"]
const p = (mu = 1 / Re, kappa = 1 / (Re * Pr)) # μ and κ

const use_gravity = config["Physics"]["use_gravity"]

if use_gravity
  const g_dual = to_device(map(edges(s)) do e
    is_X_aligned(e, s) ? gₐ * dual_edge_len(s, e) : 0.0 end)
end

const te = config["Simulation"]["te"]
const dt = config["Simulation"]["dt"]

function parse_periodic(config)
  entry = config["Simulation"]["periodic"]
  if entry == "ALL"
    return ALL
  elseif entry == "NORTHSOUTH"
    return NORTHSOUTH
  elseif entry == "EASTWEST"
    return EASTWEST
  else
    error("Valid periodic settings are: ALL, NORTHSOUTH, EASTWEST")
  end
end

const periodic = parse_periodic(config)

const saveat = floor(Int64, config["Simulation"]["savetime"] / dt)
const checkpoint_every_saveat = config["Simulation"]["checkpoint_every_savetime"]
const checkpoint_at = saveat * checkpoint_every_saveat

# Load in initial and boundary conditions
include(sim_filepath)

@isdefined(u0) || error("Initial condition not defined properly, please assign to variable \"u0\"")

#####################
### DEC Operators ###
#####################
println("Constructing DEC operators...")

println("DEC operators constructed. Moving to device...")

const dec_ops = build_dec_kernels(s)

rho_smooth_cache = Adapt.adapt(USE_CUDA ? CUDABackend() : CPU(), SmoothingCache(s, config["Smoothing"]["rho_smooth_constant"]))
Theta_smooth_cache = Adapt.adapt(USE_CUDA ? CUDABackend() : CPU(), SmoothingCache(s, config["Smoothing"]["T_smooth_constant"]))

rho_smoothing(x) = smooth_dual0_fused(rho_smooth_cache, x)
Theta_smoothing(x) = smooth_dual0_fused(Theta_smooth_cache, x)

const dec_and_smooth_ops = merge(dec_ops, (; smoothing = (; rho = rho_smoothing, Theta = Theta_smoothing) ))

##########################
### Physical Operators ###
##########################

println("Setting up physical operators...")

# Pressure 2-form from potential temperature: P = pressure(Θ)
momentum_pressure(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat =
  pressure(dec_ops.hdg_2(state.Theta_star))

# Viscous diffusion: μ Δu  (returns nothing when μ = 0)
function momentum_diffusion(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  p.mu == 0 && return nothing
  return p.mu * (dec_ops.dlap_1(fields.u) + dec_ops.dlap_1_v(fields.v))
end

# Gravity body force: ρ g  (only when use_gravity is active in the sim case)
function momentum_body_forces(::LMNSHModel, s::UniformCubicalComplex2D, state::ComponentVector{FT}, p::NamedTuple, dec_ops::NamedTuple, fields::NamedTuple) where FT <: AbstractFloat
  use_gravity || return nothing
  return dec_ops.wdg_dd_01(fields.rho, g_dual)
end

function potential_temperature_continuity(state::ComponentVector{FT}, p::NamedTuple) where FT <: AbstractFloat

  (; hdg_1, hdg_2, inv_hdg_1, inv_hdg_2, dd0, wdg_dd_01, wdg_11, dcd_1, dlap_0, interp_dp_1) = dec_ops

  U = hdg_1(state.U_star)
  rho = hdg_2(state.rho_star)
  Theta = hdg_2(state.Theta_star)

  kappa = p[:kappa]

  # Velocity from momentum/density
  u = wdg_dd_01(1 ./ rho, U)

  # Interpolate u to primal edges
  v = interp_dp_1(u)

  theta = Theta ./ rho

  # -Θ * (δu) creation/divergence term
  temperature_creation = Theta .* (dcd_1(u))

  # L(u, Θ) advection: dual_d0(Theta) → dual 1-form, convert to primal, wedge with v, back to dual
  temperature_advection = hdg_2(wdg_11(v, inv_hdg_1(dd0(Theta))))

  # κΔθ thermal diffusion
  temperature_diffusion = kappa * (dlap_0(theta))

  return inv_hdg_2(.-temperature_creation .- temperature_advection .+ temperature_diffusion)
end

build_saved_value_type(::LMNSHModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho, :Theta), Tuple{Vector{FT}, Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::LMNSHModel) = :LMNSH
checkpoint_field_names(::LMNSHModel) = (:U_star, :rho_star, :Theta_star)

function regularized_state(::LMNSHModel, u, context)
  return (
    U = Array(context.dec_ops.hdg_1(u.U_star)),
    rho = Array(context.dec_ops.hdg_2(u.rho_star)),
    Theta = Array(context.dec_ops.hdg_2(u.Theta_star)),
  )
end

function apply_periodic_prestep!(::LMNSHModel, integrator, context)
  s = context.s
  periodic = context.periodic
  set_periodic!(integrator.u.U_star, Val(1), s, periodic)
  set_periodic!(integrator.u.rho_star, Val(2), s, periodic)
  set_periodic!(integrator.u.Theta_star, Val(2), s, periodic)
  return nothing
end

model_has_smoothing(::LMNSHModel) = true

function apply_smoothing!(::LMNSHModel, integrator, context)
  integrator.u.rho_star .= context.dec_ops.smoothing.rho(integrator.u.rho_star)
  integrator.u.Theta_star .= context.dec_ops.smoothing.Theta(integrator.u.Theta_star)
  return nothing
end

############################
### Context and Run       ###
############################

build_sim_context(::LMNSHModel, periodic_side) = (; dec_ops = dec_and_smooth_ops, periodic = periodic_side)

function rhs!(du, u, p_rhs, t)
  du.U_star .= momentum_conservation(LMNSHModel(), s, u, p_rhs, dec_ops)
  du.Theta_star .= potential_temperature_continuity(u, p_rhs)
  du.rho_star .= mass_continuity(s, u, dec_ops)
  return nothing
end

######################
### Run Simulation  ###
######################

# Warm up all kernel specialisations so JIT compilation happens before the
# solver starts rather than stalling the first ODE step.
println("Warming up kernels (JIT compilation)...")
let _u0 = to_device(u0), _du = to_device(zero(u0))
  rhs!(_du, _u0, p, zero(Float64))
end
println("Kernel warmup complete.")

run_simulation(
  LMNSHModel(), s,
  to_device(u0), rhs!, p;
  dec_ops = dec_and_smooth_ops,
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  periodic=periodic,
  savepath=savepath)
