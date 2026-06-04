include(joinpath(@__DIR__, "LMNS_Helpers", "Simulation_Header.jl"))

#################################
### Simulation Initialization ###
#################################

sim = "Taylor_Vortices"

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
# TODO: Are these checks working?
# @isdefined(enforce_bc_u!) || error("Please define boundary conditions for dual velocity (u), can be trivial")
# @isdefined(enforce_bc_U!) || error("Please define boundary conditions for dual momentum (U), can be trivial")
# @isdefined(enforce_bc_v!) || error("Please define boundary conditions for primal velocity (v), can be trivial")
# @isdefined(enforce_bc_V!) || error("Please define boundary conditions for primal momentum (v), can be trivial")


#####################
### DEC Operators ###
#####################

function build_dec_kernels(s::UniformCubicalComplex2D{FT}) where FT <: AbstractFloat
  cache = UniformDECCache(s);

  d0(x) = exterior_derivative(Val(1), cache, x)
  d1(x) = exterior_derivative(Val(1), cache, x)

  dd0(x) = no_flux_dual_derivative(Val(0), cache, x)
  dd1(x) = dual_derivative(Val(1), cache, x)

  hdg_1(x) = hodge_star(Val(1), cache, x)
  hdg_2(x) = hodge_star(Val(2), cache, x)

  inv_hdg_0(x) = inv_hodge_star(Val(0), cache, x)
  inv_hdg_1(x) = inv_hodge_star(Val(1), cache, x)

  d_beta(x) = d_beta_mul(cache, x)

  wdg_01(f, a) = wedge_product(Val(0), Val(1), cache, f, a)
  wdg_11(a, b) = wedge_product(Val(1), Val(1), cache, a, b)
  wdg_dd_01(f, a) = wedge_product_dd(Val(0), Val(1), cache, f, a)

  dcd_1(x) = dual_codifferential(Val(1), cache, x)
  dcd_2(x) = dual_codifferential(Val(2), cache, x)

  # TODO: Replace with the fused kernels
  dlap_1(x) = dd0(dcd_1(x)) + dcd_2(dd1(x))
  dlap_1_v(x) = dcd_2(d_beta(x))

  interp_dp_1(x) = interpolate_dp(Val(1), cache, x)

  return (; d0=d0, d1=d1, dd0=dd0, dd1=dd1, 
  hdg_1=hdg_1, hdg_2=hdg_2, inv_hdg_0=inv_hdg_0, inv_hdg_1=inv_hdg_1, 
  d_beta=d_beta, wdg_01=wdg_01, wdg_11=wdg_11, wdg_dd_01=wdg_dd_01, 
  dcd_1=dcd_1, dcd_2=dcd_2, dlap_1=dlap_1, dlap_1_v=dlap_1_v,
  interp_dp_1)
end


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
    U = Array(dec_ops.hdg_1(u.U_star)),
    rho = Array(dec_ops.hdg_2(u.rho_star)),
  )
end

model_has_periodic_prestep(::LMNSModel, context) = hasproperty(context, :periodic) && context.periodic !== nothing

function apply_periodic_prestep!(::LMNSModel, integrator, context)
  s = context.s
  periodic = context.periodic
  set_periodic!(integrator.u.U_star, Val(1), s, periodic)
  set_periodic!(integrator.u.rho_star, Val(2), s, periodic)
  return nothing
end

############################
### Context and Run       ###
############################

build_sim_context(::LMNSModel, periodic) = (; simspec = simspec)

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
  LMNSModel(), s,
  to_device(u0), rhs!, p;
  te = te, dt = dt,
  saveat = saveat,
  checkpoint_at = checkpoint_at,
  periodic=periodic,
  savepath=savepath)
println("Simulation complete.")
