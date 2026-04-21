using Test
using CairoMakie
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using Printf
using SparseArrays
using JLD2

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

CUDA.allowscalar(false)

const USE_CUDA = CUDA.functional()
println("CUDA is functional: $USE_CUDA")

to_device(arr::AbstractVector) = USE_CUDA ? CuVector{Float64}(arr) : arr
to_device(mat::AbstractMatrix) = USE_CUDA ? CuSparseMatrixCSC{Float64}(mat) : SparseMatrixCSC{Float64}(mat)

#################################
### Simulation Initialization ###
#################################

include(joinpath(@__DIR__, "LMNS_Helpers", "Physics.jl"))

simfile = "Taylor_Vortices_Sim.jl"

# Test case parameters for lid-driven cavity flow at different Reynolds numbers
include(joinpath(@__DIR__, "Sim_Files", simfile))

#####################
### DEC Operators ###
#####################

const cpu_d0 = exterior_derivative(Val(0), s);
const cpu_d1 = exterior_derivative(Val(1), s);

const cpu_dual_d0 = no_flux_dual_derivative(Val(0), s); # Enforce no-flux boundary condition on density
const cpu_dual_d1 = dual_derivative(Val(1), s);

const cpu_d_beta = dual_derivative_beta(Val(1), s); # Closing the dual cells

const cpu_hdg_1 = hodge_star(Val(1), s);
const cpu_hdg_2 = hodge_star(Val(2), s);

const cpu_inv_hdg_0 = inv_hodge_star(Val(0), s);
const cpu_inv_hdg_1 = inv_hodge_star(Val(1), s);
const cpu_inv_hdg_2 = inv_hodge_star(Val(2), s);

const cpu_δ1 = codifferential(Val(1), s);
const cpu_dual_δ1 = dual_codifferential(Val(1), s);

# Enforce closure of dual cells
const cpu_dΔ1 = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 * cpu_dual_d1 + cpu_dual_d0 * cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1;
const cpu_dΔ1_V = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 * cpu_d_beta;

const cpu_dd0_h2 = cpu_dual_d0 * cpu_hdg_2;
const cpu_ih0_dd1 = cpu_inv_hdg_0 * cpu_dual_d1;
const cpu_ih0_db = cpu_inv_hdg_0 * cpu_d_beta;

### Convert operators to device arrays if CUDA is available ###
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

const δ1 = to_device(cpu_δ1);
const dual_δ1 = to_device(cpu_dual_δ1);

const dΔ1 = to_device(cpu_dΔ1);
const dΔ1_V = to_device(cpu_dΔ1_V);

const dd0_h2 = to_device(cpu_dd0_h2);
const ih0_dd1 = to_device(cpu_ih0_dd1);
const ih0_db = to_device(cpu_ih0_db);

const smoothing = to_device(smoothing_dual0(s, -0.2) * smoothing_dual0(s, 0.2))

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
function momentum_continuity(s::UniformCubicalComplex2D{FT}, U_star::AbstractVector{FT}, rho_star::AbstractVector{FT}, p::Dict{Symbol, FT}) where FT
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
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
  L_term = dd0_h2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(0), Val(1), s, ih0_dd1 * U + ih0_db * V, v)

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dd0_h2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 .* wedge_product_dd(Val(0), Val(1), s, rho, diff_norm_u)

  # dP, P = κρ
  diff_p = dual_d0 * pressure_same_theta(rho)

  # muΔu
  lap_term = mu * (dΔ1 * u + dΔ1_V * v)

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term)
end

function run_compressible_ns(s::UniformCubicalComplex2D{FT}, U_star_0::AbstractVector{FT}, rho_star_0::AbstractVector{FT}, te::FT, dt::FT, p::Dict{Symbol, FT}; saveat::Int=500, checkpoint_at::Int=10_000, start_step::Int=1) where FT

  #############
  ### Setup ###
  #############

  U_star = deepcopy(U_star_0)
  rho_star = deepcopy(rho_star_0)

  U_star_half = similar(U_star)

  rho_star_half = similar(rho_star)
  rho_star_full = similar(rho_star)

  steps = ceil(Int64, te / dt)

  Us = [Array(hdg_1 * U_star_0)]
  rhos = [Array(hdg_2 * rho_star_0)]

  m₀ = sum(interior(Val(2), Array(rho_star_0), s))

  error_encounted = false

  #################
  ### Main Loop ###
  #################

  for step in start_step:steps

    set_periodic!(U_star, Val(1), s, ALL) # REMOVEME
    set_periodic!(rho_star, Val(2), s, ALL) # REMOVEME

    U_star_half .= U_star .+ 0.5 * dt * momentum_continuity(s, U_star, rho_star, p)

    set_periodic!(U_star_half, Val(1), s, ALL) # REMOVEME
    enforce_bc_U!(U_star_half)

    rho_star_full .= smoothing * (rho_star + dt * d1 * U_star_half) # Mass changes by momentum flux
    set_periodic!(rho_star_full, Val(2), s, ALL) # REMOVEME

    rho_star_half .= 0.5 .* (rho_star .+ rho_star_full)

    U_star .= U_star .+ dt * momentum_continuity(s, U_star_half, rho_star_half, p)
    enforce_bc_U!(U_star)
    

    rho_star .= rho_star_full

    if any(isnan, U_star)
      println("Warning, NAN result in U at step: $(step)")
      error_encounted = true
    elseif any(isinf, U_star)
      println("Warning, INF result in U at step: $(step)")
      error_encounted = true
    elseif any(isnan, rho_star)
      println("Warning, NAN result in rho_star at step: $(step)")
      error_encounted = true
    elseif any(isinf, rho_star)
      println("Warning, INF result in rho_star at step: $(step)")
      error_encounted = true
    end

    if step % saveat == 0 || step == steps || error_encounted
      push!(Us, Array(hdg_1 * U_star))
      push!(rhos, Array(hdg_2 * rho_star))
      println("Loading simulation results: $((step / steps) * 100)%")
      println("Relative mass is : $((sum(interior(Val(2), Array(rho_star), s)) / m₀) * 100)%")
      println("-----")

      flush(stdout)
    end

    if step % checkpoint_at == 0 || step == steps || error_encounted
      @save joinpath(save_path, "checkpoint_step_$(step).jld2") Us rhos
      println("Checkpoint saved at step: $(step)")

      fin_U = Us[end]
      fin_rho = rhos[end]

      time = @sprintf("%.6f", step * dt)
      file_end = "$(simspec)_t=$(time)"

      plot_vorticity(s, fin_U, file_end, time)
      plot_density(s, fin_rho, file_end, time)
      plot_velocity_components(s, fin_U, file_end, time)
      plot_velocity_magnitude(s, fin_U, file_end, time)

      create_mp4(Us, rhos, file_end; records = div(checkpoint_at, saveat))

      # To prevent memory overflow, we can clear the saved states after checkpointing
      Us = [fin_U]
      rhos = [fin_rho]
    end

    error_encounted && break
  end
end

function run_compressible_ns(s::UniformCubicalComplex2D{FT}, U_star_0::AbstractVector{FT}, rho_star_0::AbstractVector{FT}, te::FT, dt::FT, p::Dict{Symbol, FT}; saveat::Int=500, checkpoint_at::Int=10_000, start_step::Int=1) where FT

  #############
  ### Setup ###
  #############

  U_star = deepcopy(U_star_0)
  rho_star = deepcopy(rho_star_0)

  U_star_1 = similar(U_star)
  rho_star_1 = similar(rho_star)

  U_star_2 = similar(U_star)
  rho_star_2 = similar(rho_star)

  U_star_full = similar(U_star)
  rho_star_full = similar(rho_star)

  steps = ceil(Int64, te / dt)

  Us = [Array(hdg_1 * U_star_0)]
  rhos = [Array(hdg_2 * rho_star_0)]

  m₀ = sum(interior(Val(2), Array(rho_star_0), s))

  error_encounted = false

  #################
  ### Main Loop ###
  #################

  for step in start_step:steps

    set_periodic!(U_star, Val(1), s, ALL) # REMOVEME
    set_periodic!(rho_star, Val(2), s, ALL) # REMOVEME

    U_star_1 .= U_star .+ dt * momentum_continuity(s, U_star, rho_star, p)
    rho_star_1 .= rho_star .+ dt * d1 * U_star_1

    set_periodic!(U_star_1, Val(1), s, ALL) # REMOVEME
    set_periodic!(rho_star_1, Val(2), s, ALL) # REMOVEME

    U_star_2 .= 0.75 .* U_star .+ 0.25 .* U_star_1 + 0.25 * dt * momentum_continuity(s, U_star_1, rho_star_1, p)
    rho_star_2 .= 0.75 .* rho_star .+ 0.25 .* rho_star_1 + 0.25 * dt * d1 * U_star_1 # Mass changes by momentum flux

    set_periodic!(U_star_2, Val(1), s, ALL) # REMOVEME
    set_periodic!(rho_star_2, Val(2), s, ALL) # REMOVEME

    U_star_full .= (1/3) .* U_star .+ (2/3) .* U_star_2 + (2/3) * dt * momentum_continuity(s, U_star_2, rho_star_2, p)
    rho_star_full .= (1/3) .* rho_star .+ (2/3) .* rho_star_2 + (2/3) * dt * d1 * U_star_2

    U_star .= U_star_full
    rho_star .= rho_star_full

    if any(isnan, U_star)
      println("Warning, NAN result in U at step: $(step)")
      error_encounted = true
    elseif any(isinf, U_star)
      println("Warning, INF result in U at step: $(step)")
      error_encounted = true
    elseif any(isnan, rho_star)
      println("Warning, NAN result in rho_star at step: $(step)")
      error_encounted = true
    elseif any(isinf, rho_star)
      println("Warning, INF result in rho_star at step: $(step)")
      error_encounted = true
    end

    if step % saveat == 0 || step == steps || error_encounted
      push!(Us, Array(hdg_1 * U_star))
      push!(rhos, Array(hdg_2 * rho_star))
      println("Loading simulation results: $((step / steps) * 100)%")
      println("Relative mass is : $((sum(interior(Val(2), Array(rho_star), s)) / m₀) * 100)%")
      println("-----")

      flush(stdout)
    end

    if step % checkpoint_at == 0 || step == steps || error_encounted
      @save joinpath(save_path, "checkpoint_step_$(step).jld2") Us rhos
      println("Checkpoint saved at step: $(step)")

      fin_U = Us[end]
      fin_rho = rhos[end]

      time = @sprintf("%.6f", step * dt)
      file_end = "$(simspec)_t=$(time)"

      plot_vorticity(s, fin_U, file_end, time)
      plot_density(s, fin_rho, file_end, time)
      plot_velocity_components(s, fin_U, file_end, time)
      plot_velocity_magnitude(s, fin_U, file_end, time)

      create_mp4(Us, rhos, file_end; records = div(checkpoint_at, saveat))

      # To prevent memory overflow, we can clear the saved states after checkpointing
      Us = [fin_U]
      rhos = [fin_rho]
    end

    error_encounted && break
  end
end


println("Starting simulation...")

run_compressible_ns(s, to_device(U_star_0), to_device(rho_star_0), te, dt, p; saveat=saveat, checkpoint_at=checkpoint_at);

println("Simulation complete.")