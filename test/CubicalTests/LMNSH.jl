using Test
using CairoMakie
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using Printf
using SparseArrays
using JLD2
using ComponentArrays
using Distributions
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks

include("../../src/CubicalCode/UniformDEC.jl")

CUDA.allowscalar(false)

const USE_CUDA = CUDA.functional()
println("CUDA is functional: $USE_CUDA")

# Toggle this to start from latest checkpoint instead of initial conditions.
const RESUME_FROM_CHECKPOINT = false

to_device(arr::AbstractVector{FT}) where FT <: AbstractFloat = USE_CUDA ? CuVector{FT}(arr) : arr
to_device(arr::AbstractVector{T}) where T = USE_CUDA ? CuVector{T}(arr) : arr
to_device(mat::AbstractMatrix{FT}) where FT <: AbstractFloat = USE_CUDA ? CuSparseMatrixCSC{FT}(mat) : SparseMatrixCSC{FT}(mat)

#################################
### Simulation Initialization ###
#################################

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics.jl"))

simfile = "Kelvin-Helmholtz_Sim.jl"
# simfile = "Thermal_Bubble_Sim.jl"

# Test case parameters for lid-driven cavity flow at different Reynolds numbers
include(joinpath(@__DIR__, "Sim_Files", simfile))

#####################
### DEC Operators ###
#####################
println("Constructing DEC operators...")

const cpu_d0 = exterior_derivative(Val(0), s);
const cpu_d1 = exterior_derivative(Val(1), s);

const cpu_dual_d0 = no_flux_dual_derivative(Val(0), s) # Enforce no-flux boundary condition on density
const cpu_dual_d1 = dual_derivative(Val(1), s);

const cpu_d_beta = dual_derivative_beta(Val(1), s); # Closing the dual cells

const cpu_hdg_1 = hodge_star(Val(1), s);
const cpu_hdg_2 = hodge_star(Val(2), s);

const cpu_inv_hdg_0 = inv_hodge_star(Val(0), s);
const cpu_inv_hdg_1 = inv_hodge_star(Val(1), s);
const cpu_inv_hdg_2 = inv_hodge_star(Val(2), s);

const cpu_δ1 = codifferential(Val(1), s);
const cpu_dual_δ1 = dual_codifferential(Val(1), s);

# No gradient through boundary edges
const cpu_dΔ0 = cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1 * cpu_dual_d0

# Enforce closure of dual cells
const cpu_dΔ1 = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 * cpu_dual_d1 + cpu_dual_d0 * cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1
const cpu_dΔ1_V = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 *	cpu_d_beta

println("DEC operators constructed. Moving to device...")

const d0 = to_device(cpu_d0);
const d1 = to_device(cpu_d1);
const dual_d0 = to_device(cpu_dual_d0);
const dual_d1 = to_device(cpu_dual_d1);
const d_beta = to_device(cpu_d_beta);
const hdg_1 = to_device(cpu_hdg_1);
const hdg_2 = to_device(cpu_hdg_2);
const inv_hdg_0 = to_device(cpu_inv_hdg_0);
const inv_hdg_1 = to_device(cpu_inv_hdg_1);
const inv_hdg_2 = to_device(cpu_inv_hdg_2);
const dual_δ1 = to_device(cpu_dual_δ1);
const dΔ0 = to_device(cpu_dΔ0);
const dΔ1 = to_device(cpu_dΔ1);
const dΔ1_V = to_device(cpu_dΔ1_V);

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
function momentum_continuity(state::ComponentVector{Float64}, p::NamedTuple, use_gravity::Bool=false)
  U_star = state.U_star
  rho_star = state.rho_star
  Theta_star = state.Theta_star

  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  Theta = hdg_2 * Theta_star

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

  # U ∧ δu
  div_term = wedge_product_dd(Val(0), Val(1), s, dual_δ1 * u, U)

  # L(u, U)
  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(0), Val(1), s, inv_hdg_0 * (dual_d1 * U + d_beta * V), v)

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 .* wedge_product_dd(Val(0), Val(1), s, rho, diff_norm_u)

  # dP, P = κρ
  diff_p = dual_d0 * pressure(Theta)

  # muΔu
  lap_term = mu * (dΔ1 * u + dΔ1_V * v)

  # ρg body force
  # body_forces = wedge_product_dd(Val(0), Val(1), s, rho, g_dual) * use_gravity

  # return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term + body_forces)
  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term)
end

function potential_temperature_continuity(state::ComponentVector{Float64}, p::NamedTuple)
  U_star = state.U_star
  rho_star = state.rho_star
  Theta_star = state.Theta_star

  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  Theta = hdg_2 * Theta_star

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
  temperature_diffusion = kappa * dΔ0 * theta

  return inv_hdg_2 * (-temperature_creation - temperature_advection + temperature_diffusion)
end

function latest_checkpoint_path(checkpoint_dir::AbstractString)
  isdir(checkpoint_dir) || return nothing

  re = r"^checkpoint_step_(\d+)\.jld2$"
  latest_step = -1
  latest_file = nothing

  for file in readdir(checkpoint_dir)
    m = match(re, file)
    m === nothing && continue

    step = parse(Int, m.captures[1])
    if step > latest_step
      latest_step = step
      latest_file = joinpath(checkpoint_dir, file)
    end
  end

  return latest_file
end

function split_flat_state(state_flat::AbstractVector, nU::Int, nrho::Int, nTheta::Int)
  U_star = state_flat[1:nU]
  rho_star = state_flat[(nU + 1):(nU + nrho)]
  Theta_star = state_flat[(nU + nrho + 1):(nU + nrho + nTheta)]
  return U_star, rho_star, Theta_star
end

function extract_state_triplet(state_like, nU::Int, nrho::Int, nTheta::Int)
  if hasproperty(state_like, :U_star) && hasproperty(state_like, :rho_star) && hasproperty(state_like, :Theta_star)
    return state_like.U_star, state_like.rho_star, state_like.Theta_star
  end
  return split_flat_state(Array(state_like), nU, nrho, nTheta)
end

# TODO: Check this function for correctness
function load_checkpoint_state(checkpoint_path::AbstractString, nU::Int, nrho::Int, nTheta::Int)
  data = JLD2.load(checkpoint_path)

  checkpoint_t = Float64(data["checkpoint_t"])
  checkpoint_step = Int(data["checkpoint_step"])
  checkpoint_state = data["checkpoint_state"]

  if checkpoint_state isa AbstractVector && !isempty(checkpoint_state)
    U_star, rho_star, Theta_star = extract_state_triplet(checkpoint_state[end], nU, nrho, nTheta)
  elseif hasproperty(checkpoint_state, :u)
    checkpoint_u = checkpoint_state.u[end]
    U_star, rho_star, Theta_star = extract_state_triplet(checkpoint_u, nU, nrho, nTheta)
  else
    U_star, rho_star, Theta_star = extract_state_triplet(checkpoint_state, nU, nrho, nTheta)
  end

  return (
    state = ComponentArray(
      U_star = to_device(U_star),
      rho_star = to_device(rho_star),
      Theta_star = to_device(Theta_star),
    ),
    checkpoint_t = checkpoint_t,
    checkpoint_step = checkpoint_step,
  )
end

function initialize_state(U_star_0::AbstractVector{Float64}, rho_star_0::AbstractVector{Float64}, Theta_star_0::AbstractVector{Float64}; resume::Bool=false, checkpoint_dir::AbstractString=save_path)
  nU = length(U_star_0)
  nrho = length(rho_star_0)
  nTheta = length(Theta_star_0)

  if resume
    checkpoint_path = latest_checkpoint_path(checkpoint_dir)
    if checkpoint_path !== nothing
      loaded = load_checkpoint_state(checkpoint_path, nU, nrho, nTheta)
      start_step = loaded.checkpoint_step + 1
      start_time = loaded.checkpoint_t
      println("Resuming from checkpoint $(checkpoint_path)")
      println("Resuming from t=$(start_time), step=$(start_step)")
      return loaded.state, start_time, start_step
    end
    println("No checkpoint found in $(checkpoint_dir); starting from initial condition.")
  end

  state = ComponentArray(
    U_star = deepcopy(U_star_0),
    rho_star = deepcopy(rho_star_0),
    Theta_star = deepcopy(Theta_star_0),
  )

  return state, 0.0, 1
end

# TODO: Type stability for this function
function run_compressible_ns(u0::ComponentVector{Float64}, te::Float64, dt::Float64, p; saveat::Int=500, checkpoint_at::Int=10_000, start_step::Int=1, start_time::Float64=0.0, use_gravity::Bool=false)
  
  m₀ = sum(interior(Val(2), Array(u0.rho_star), s))
  E₀ = sum(interior(Val(2), Array(u0.Theta_star), s))

  function rhs!(du, u, p_rhs, t)
    du.U_star .= momentum_continuity(u, p_rhs, use_gravity)
    du.Theta_star .= potential_temperature_continuity(u, p_rhs)
    du.rho_star .= d1 * u.U_star

    set_periodic!(du.U_star, Val(1), s, ALL)
    set_periodic!(du.Theta_star, Val(2), s, ALL)
    set_periodic!(du.rho_star, Val(2), s, ALL)

    return nothing
  end

  function regularized_state(u)
    return (
      U = Array(hdg_1 * u.U_star),
      rho = Array(hdg_2 * u.rho_star),
      Theta = Array(hdg_2 * u.Theta_star),
    )
  end

  regular_save_values = SavedValues(
    Float64,
    NamedTuple{(:U, :rho, :Theta), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}},
  )

  regular_state_cb = SavingCallback(
    (u, t, integrator) -> regularized_state(u),
    regular_save_values;
    saveat = saveat * dt,
    save_start = true,
    save_end = false,
  )

  step_from_iter(iter) = start_step - 1 + iter

  function save_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(integrator.iter)
    return step % saveat == 0
  end

  function save_affect!(integrator)
    u = integrator.u

    println("Loading simulation results: $((integrator.t / te) * 100)%")
    println("Relative mass is : $((sum(interior(Val(2), Array(u.rho_star), s)) / m₀) * 100)%")
    println("Relative energy is : $((sum(interior(Val(2), Array(u.Theta_star), s)) / E₀) * 100)%")
    println("-----")
    flush(stdout)

    return nothing
  end

  function checkpoint_condition(u, t, integrator)
    integrator.iter > 0 || return false
    step = step_from_iter(integrator.iter)
    return step % checkpoint_at == 0
  end

  function checkpoint_affect!(integrator)
    step = step_from_iter(integrator.iter)

    u = integrator.u

    checkpoint_t = integrator.t
    checkpoint_step = step
    checkpoint_regular_t = regular_save_values.t
    checkpoint_regular_state = regular_save_values.saveval

    @save joinpath(save_path, "checkpoint_step_$(step).jld2") checkpoint_t checkpoint_step checkpoint_regular_t checkpoint_regular_state

    println("Checkpoint saved at step: $(step)")
    println("Generating plots and mp4 for checkpoint...")

    if isempty(checkpoint_regular_state)
      push!(checkpoint_regular_t, checkpoint_t)
      push!(checkpoint_regular_state, regularized_state(u))
    end

    time = @sprintf("%.6f", step * dt)
    file_end = "$(simspec)_t=$(time)"

    plot_vorticity(s, regular_save_values.saveval[end].U, file_end, time)
    plot_density(s, regular_save_values.saveval[end].rho, file_end, time)
    # plot_velocity_magnitude(s, fin_U, file_end, time)
    plot_pressure(s, regular_save_values.saveval[end].Theta, file_end, time)
    plot_velocity_components(s, regular_save_values.saveval[end].U, file_end, time)

    create_mp4(regular_save_values, file_end; records = div(checkpoint_at, saveat))

    # Start a fresh interval buffer after checkpoint while keeping continuity at the boundary.
    empty!(regular_save_values.t)
    empty!(regular_save_values.saveval)
    push!(regular_save_values.t, checkpoint_t)
    push!(regular_save_values.saveval, regularized_state(u))

    println("Plots and mp4 generated for checkpoint at step: $(step)")
    println("-----")
    flush(stdout)

    return nothing
  end

  callbacks = CallbackSet(
    regular_state_cb,
    DiscreteCallback(save_condition, save_affect!; save_positions = (false, false)),
    DiscreteCallback(checkpoint_condition, checkpoint_affect!; save_positions = (false, false)),
  )

  tspan = (start_time, te)
  prob = ODEProblem(rhs!, u0, tspan, p)
  soln = solve(
    prob,
    SSPRK33();
    dt = dt,
    adaptive = false,
    save_everystep = false,
    save_start = false,
    save_end = false,
    callback = callbacks,
    dense = false,
  )

  return soln
end

println("Starting simulation...")

initial_state, start_time, start_step = initialize_state(
  to_device(U_star_0),
  to_device(rho_star_0),
  to_device(Theta_star_0);
  resume = RESUME_FROM_CHECKPOINT,
)

run_compressible_ns(initial_state, te, dt, p; saveat=saveat, checkpoint_at=checkpoint_at, start_step=start_step, start_time=start_time, use_gravity=use_gravity);

println("Simulation complete.")
