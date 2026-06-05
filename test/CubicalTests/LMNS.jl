include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

const sim = "Taylor_Vortices"

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
jldopen(savedata_filepath, "w") do file end

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

dec_ops = build_dec_kernels(s)

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
  return p.mu * (dec_ops.dlap_1(fields.u) + dec_ops.dlap_1_v(fields.v))
end

build_saved_value_type(::LMNSModel, ::Type{FT}) where FT <: AbstractFloat =
  NamedTuple{(:U, :rho), Tuple{Vector{FT}, Vector{FT}}}

checkpoint_model_kind(::LMNSModel) = :LMNS
checkpoint_field_names(::LMNSModel) = (:U_star, :rho_star)

function regularized_state(::LMNSModel, u, context)
  return (
    U = Array(context.dec_ops.hdg_1(u.U_star)),
    rho = Array(context.dec_ops.hdg_2(u.rho_star)),
  )
end

function apply_periodic_prestep!(::LMNSModel, integrator, context)
  s = context.s
  periodic = context.periodic
  set_periodic!(integrator.u.U_star, Val(1), s, periodic)
  set_periodic!(integrator.u.rho_star, Val(2), s, periodic)
  return nothing
end

############################
### Context and Run      ###
############################

build_sim_context(::LMNSModel, periodic) = (;)

function rhs!(du, u, p_rhs, t)
  du.U_star .= momentum_conservation(LMNSModel(), s, u, p_rhs, dec_ops)
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
  LMNSModel(), s,
  to_device(u0), rhs!, p;
  dec_ops = dec_ops,
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  periodic=periodic,
  savepath=savepath)
