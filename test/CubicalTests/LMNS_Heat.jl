using CairoMakie
using Distributions
using LinearAlgebra
using Printf
using SparseArrays

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

lx_ = ly_ = 6.0;
nx_ = ny_ = 151
s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_; halo_x = 5, halo_y = 5)

# DEC Operators
Δ0 = laplacian(Val(0), s);
dΔ0 = dual_laplacian(Val(0), s);
dΔ1 = dual_laplacian(Val(1), s);

d0 = exterior_derivative(Val(0), s);
d1 = exterior_derivative(Val(1), s);

dual_d0 = dual_derivative(Val(0), s);
dual_d1 = dual_derivative(Val(1), s);

hdg_1 = hodge_star(Val(1), s);
hdg_2 = hodge_star(Val(2), s);

inv_hdg_0 = inv_hodge_star(Val(0), s);
inv_hdg_1 = inv_hodge_star(Val(1), s);
inv_hdg_2 = inv_hodge_star(Val(2), s);

δ1 = codifferential(Val(1), s);
dual_δ1 = dual_codifferential(Val(1), s);

##############################
##### THERMODYNAMIC CONSTANTS #
##############################

P₀ = 1e5 # Reference pressure for potential temperature
Cₚ = 1006 # Specific heat
R_gas = 287 # Specific gas constant
ρᵣ = 1 # Reference density
θᵣ = 300 # Reference potential temperature, temperature at P₀
Pᵣ = ρᵣ * R_gas * θᵣ
gₐ = -9.81 # Acceleration due to gravity

# Gravity as a primal 1-form: x-edges = 0, y-edges = gₐ * dy
g_primal = zeros(ne(s))
g_primal[nxedges(s)+1:end] .= gₐ * dy(s)
g_dual = hdg_1 * g_primal

# From the ideal gas law: P = (Θ R (P₀^(-R/Cₚ)))^(Cₚ/(Cₚ-R))
function pressure(Theta)
  R_Cₚ = R_gas / Cₚ
  return (Theta .* R_gas .* (P₀ .^ -R_Cₚ)) .^ (1 / (1 - R_Cₚ))
end

# Assumes theta is constant in z
function hydrostatic_pressure(theta, h)
  return (1 / (Cₚ * theta) * (P₀)^(R_gas / Cₚ) * gₐ * h + Pᵣ^(R_gas / Cₚ))^(Cₚ / R_gas)
end

function hydrostatic_density(theta, h)
  Pₕ = hydrostatic_pressure(theta, h)
  return (Pₕ / (R_gas * theta)) * (P₀ / Pₕ)^(R_gas / Cₚ)
end

##############################
##### INITIAL CONDITIONS #####
##############################

dps = dual_points(s)

simname = "LMNS_Heat_Warm_Bubble"

if simname == "LMNS_Heat_Constant"
  rho_0 = ones(nquads(s))
  U_star_0 = zeros(ne(s))
  theta_0 = 300 * ones(nquads(s))
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Hydrostatic"
  thetaₕ = 300
  rho_0 = map(p -> hydrostatic_density(thetaₕ, p[2]), dps)
  U_star_0 = zeros(ne(s))
  theta_0 = thetaₕ * ones(nquads(s))
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Warm_Bubble"
  rho_0 = ones(nquads(s))
  U_star_0 = zeros(ne(s))
  theta_dist = MvNormal([lx_ / 2, ly_ / 2 ], 0.1)
  theta_0 = 300 .+ 0.5 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Warm_Bubble_Gradient"
  thetaₕ = 300
  rho_0 = map(p -> hydrostatic_density(thetaₕ, p[2]), dps)
  U_star_0 = zeros(ne(s))
  theta_dist = MvNormal([lx_ / 2, ly_ / 2], 0.1)
  theta_0 = thetaₕ * ones(nquads(s)) .+ 5 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Cold_Bubble"
  rho_0 = ones(nquads(s))
  U_star_0 = zeros(ne(s))
  theta_dist = MvNormal([lx_ / 2, ly_ / 2], 0.1)
  theta_0 = 300 .- 0.05 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Cold_Bubble_Wind"
  rho_0 = ones(nquads(s))
  # Constant velocity (100, 0) as a primal 1-form: x-edges get 100*dx, y-edges get 0
  U_star_0 = zeros(ne(s))
  U_star_0[1:nxedges(s)] .= 100 * dx(s)
  theta_dist = MvNormal([lx_ / 2, ly_ / 2], 0.1)
  theta_0 = 300 .- 0.05 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_Light_Bubble_Gradient"
  thetaₕ = 300
  rho_0 = map(p -> hydrostatic_density(thetaₕ, p[2]), dps)
  U_star_0 = zeros(ne(s))
  theta_dist = MvNormal([lx_ / 2, ly_ / 4], 0.1)
  theta_0 = thetaₕ * ones(nquads(s)) .+ 5 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
elseif simname == "LMNS_Heat_RTI"
  rho_0 = ones(nquads(s))
  U_star_0 = zeros(ne(s))
  theta_0 = zeros(nquads(s))

  upper_quads = findall(p -> p[2] >= lx_ * 0.75, dps)
  lower_quads = findall(p -> p[2] < lx_ * 0.75, dps)

  theta_0[upper_quads] .= 302
  theta_0[lower_quads] .= 300
  theta_0 .+= 1e-6 * ((2 * rand(nquads(s))) .- 1)
  Theta_0 = rho_0 .* theta_0
  rho_star_0 = inv_hdg_2 * rho_0
  Theta_star_0 = inv_hdg_2 * Theta_0
else
  error("Unknown simname: $(simname)")
end

##################################
##### END INITIAL CONDITIONS #####
##################################

save("imgs/LMNSH/InitialDensity.png", plot_twoform(s, hdg_2 * rho_star_0))
save("imgs/LMNSH/InitialTheta.png", plot_twoform(s, hdg_2 * Theta_star_0))

##############################
##### EQUATIONS OF MOTION ####
##############################

function momentum_continuity(U_star, rho_star, Theta_star, μ, use_gravity=false)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  Theta = hdg_2 * Theta_star

  # Compute velocity u from momentum U and density rho
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  v = interpolate_dp(Val(1), s, u)

  # U ∧ δu
  div_term = wedge_product_dd(Val(1), Val(0), s, U, dual_δ1 * u)

  # L(u, U)
  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(1), Val(0), s, v, inv_hdg_0 * dual_d1 * U)

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 * wedge_product_dd(Val(1), Val(0), s, diff_norm_u, rho)

  # dP from ideal gas law
  diff_p = dual_d0 * pressure(Theta)

  # μΔu
  lap_term = μ * dΔ1 * u

  # ρg body force
  body_forces = wedge_product_dd(Val(0), Val(1), s, rho, g_dual) * use_gravity

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term + body_forces)
end

function potential_temperature_continuity(U_star, rho_star, Theta_star, κ)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  Theta = hdg_2 * Theta_star

  # Velocity from momentum/density
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  v = interpolate_dp(Val(1), s, u)

  theta = Theta ./ rho

  # -Θ * (δu) creation/divergence term
  temperature_creation = Theta .* (dual_δ1 * u)

  # L(u, Θ) advection: dual_d0(Theta) → dual 1-form, convert to primal, wedge with v, back to dual
  temperature_advection = hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * dual_d0 * Theta)

  # κΔθ thermal diffusion
  temperature_diffusion = κ * dΔ0 * theta

  return inv_hdg_2 * (-temperature_creation - temperature_advection + temperature_diffusion)
end

##############################
##### SMOOTHING          #####
##############################

# Smoothing for dual 0-forms (which live on quads/faces).
# Each quad is averaged with its face-adjacent neighbors (sharing an edge),
# weighted by the inverse distance between quad centers.
function smoothing_dual0(s::UniformCubicalComplex2D, c_smooth)
  n = nquads(s)
  mat = spzeros(n, n)

  for q in quads(s)
    x, y = quad_to_coord(s, q)
    # Left neighbor (separated by dx)
    x > 1 && (mat[q, coord_to_quad(s, x - 1, y)] = 1 / dx(s))
    # Right neighbor (separated by dx)
    x < nxquads(s) && (mat[q, coord_to_quad(s, x + 1, y)] = 1 / dx(s))
    # Bottom neighbor (separated by dy)
    y > 1 && (mat[q, coord_to_quad(s, x, y - 1)] = 1 / dy(s))
    # Top neighbor (separated by dy)
    y < nyquads(s) && (mat[q, coord_to_quad(s, x, y + 1)] = 1 / dy(s))
  end

  c = c_smooth / 2

  for q in quads(s)
    row = mat[q, :]
    tot_w = sum(row)
    if tot_w > 0
      for i in row.nzind
        mat[q, i] = c * row[i] / tot_w
      end
    end
    mat[q, q] = (1 - c)
  end
  return mat
end

c_smooth = 0.2
forward_smooth = smoothing_dual0(s, c_smooth)
backward_smooth = smoothing_dual0(s, -c_smooth)
smoothing_mat = backward_smooth * forward_smooth

##############################
##### TIME INTEGRATION   #####
##############################

function run_compressible_ns(U_star_0, rho_star_0, Theta_star_0, tₑ, Δt; saveat=500, use_gravity=false)

  U_star = deepcopy(U_star_0)
  rho_star = deepcopy(rho_star_0)
  Theta_star = deepcopy(Theta_star_0)

  U_star_half = zeros(Float64, ne(s))
  Theta_star_half = zeros(Float64, nquads(s))

  rho_star_half = zeros(Float64, nquads(s))
  rho_star_full = zeros(Float64, nquads(s))

  steps = ceil(Int64, tₑ / Δt)

  Us = [Array(hdg_1 * U_star_0)]
  rhos = [Array(hdg_2 * rho_star_0)]
  Thetas = [Array(hdg_2 * Theta_star_0)]

  m₀ = sum(interior(Val(2), rho_star_0, s))
  E₀ = sum(interior(Val(2), Theta_star_0, s))

  for step in 1:steps

    set_periodic!(U_star, Val(1), s, ALL)
    set_periodic!(rho_star, Val(2), s, ALL)
    set_periodic!(Theta_star, Val(2), s, ALL)

    # Half step for momentum and potential temperature
    U_star_half .= U_star .+ 0.5 * Δt * momentum_continuity(U_star, rho_star, Theta_star, μ, use_gravity)
    Theta_star_half .= Theta_star .+ 0.5 * Δt * potential_temperature_continuity(U_star, rho_star, Theta_star, κ)

    set_periodic!(U_star_half, Val(1), s, ALL)
    set_periodic!(Theta_star_half, Val(2), s, ALL)

    # Full step for density (mass flux)
    rho_star_full .= smoothing_mat * (rho_star + Δt * d1 * U_star_half)
    set_periodic!(rho_star_full, Val(2), s, ALL)

    rho_star_half .= 0.5 .* (rho_star .+ rho_star_full)

    # Full step for momentum and potential temperature
    U_star .= U_star .+ Δt * momentum_continuity(U_star_half, rho_star_half, Theta_star_half, μ, use_gravity)
    Theta_star .= Theta_star .+ Δt * potential_temperature_continuity(U_star_half, rho_star_half, Theta_star_half, κ)
    rho_star .= rho_star_full

    if any(isnan.(U_star))
      println("Warning, NAN result in U at step: $(step)")
      break
    elseif any(isinf.(U_star))
      println("Warning, INF result in U at step: $(step)")
      break
    elseif any(isnan.(rho_star))
      println("Warning, NAN result in rho at step: $(step)")
      break
    elseif any(isinf.(rho_star))
      println("Warning, INF result in rho at step: $(step)")
      break
    elseif any(isnan.(Theta_star))
      println("Warning, NAN result in Theta at step: $(step)")
      break
    elseif any(isinf.(Theta_star))
      println("Warning, INF result in Theta at step: $(step)")
      break
    end

    if step % saveat == 0
      push!(Us, hdg_1 * U_star)
      push!(rhos, hdg_2 * rho_star)
      push!(Thetas, hdg_2 * Theta_star)
      println("Step $(step)/$(steps) ($(@sprintf("%.1f", (step / steps) * 100))%)")
      # TODO: Is something mutating the Thetas arrays here?
      # println("  Relative mass:   $(@sprintf("%.6f", (sum(interior(Val(2), rho_star, s)) / m₀) * 100))%")
      # println("  Relative energy: $(@sprintf("%.6f", (sum(interior(Val(2), Theta_star, s)) / E₀) * 100))%")
      # println("-----")
    end
  end

  return Thetas, Us, rhos
end

##############################
##### RUN SIMULATION     #####
##############################

# For dry air
Pr = 0.7
μ = 1e-3 # Momentum diffusivity
κ = μ / Pr # Thermal diffusivity

tₑ = 2e-2
Δt = 1e-5
Thetas, Us, rhos = run_compressible_ns(U_star_0, rho_star_0, Theta_star_0, tₑ, Δt; saveat=1, use_gravity=false);

##############################
##### VISUALIZATION      #####
##############################

timestep = length(Us)

# Vorticity
ω_end = inv_hdg_0 * dual_d1 * Us[timestep];
save("imgs/LMNSH/Vorticity.png", plot_zeroform(s, ω_end))

# Density
save("imgs/LMNSH/FinalDensity.png", plot_twoform(s, rhos[timestep]))

# Potential Temperature (Θ/ρ)
theta_end = Thetas[timestep] ./ rhos[timestep];
save("imgs/LMNSH/PotentialTemperature.png", plot_twoform(s, theta_end))

# Velocity Magnitude
u_end = wedge_product_dd(Val(0), Val(1), s, 1 ./ rhos[timestep], Us[timestep]);
X_end = zeros(nquads(s)); Y_end = zeros(nquads(s));
sharp_dd!(X_end, Y_end, s, u_end);
vel_mag_end = sqrt.(X_end.^2 .+ Y_end.^2);
save("imgs/LMNSH/VelocityMagnitude.png", plot_twoform(s, vel_mag_end))

# Density-Coupled Potential Temperature
save("imgs/LMNSH/DensityCoupledTheta.png", plot_twoform(s, Thetas[timestep]))

# Pressure Field
save("imgs/LMNSH/PressureField.png", plot_twoform(s, pressure(Thetas[timestep])))

# Momentum Divergence
U_div = inv_hdg_0 * dual_d1 * Us[timestep];
save("imgs/LMNSH/MomentumDivergence.png", plot_zeroform(s, U_div))

##############################
##### ANIMATION          #####
##############################

function create_mp4(filename, Us, rhos, Thetas; frames=length(Us), framerate=15)

  nqx = nxquads(s) - 2 * hx(s)
  nqy = nyquads(s) - 2 * hy(s)

  dps = interior(Val(2), dual_points(s), s)
  unique_x = sort(unique(map(a -> a[1], dps)))
  unique_y = sort(unique(map(a -> a[2], dps)))

  fig = Figure(size=(1200, 800))
  ax1 = Axis(fig[1, 1], title="Density")
  ax2 = Axis(fig[1, 3], title="Density-Coupled Potential Temperature")
  ax3 = Axis(fig[2, 1], title="Velocity Magnitude")
  ax4 = Axis(fig[2, 3], title="Pressure")

  step = Observable(1)

  rho = @lift(reshape(interior(Val(2), rhos[$step], s), nqx, nqy))
  theta = @lift(reshape(interior(Val(2), Thetas[$step], s), nqx, nqy))
  vel_mag = @lift(begin
    u_step = wedge_product_dd(Val(0), Val(1), s, 1 ./ rhos[$step], Us[$step])
    X_step = zeros(nquads(s)); Y_step = zeros(nquads(s))
    sharp_dd!(X_step, Y_step, s, u_step)
    reshape(interior(Val(2), sqrt.(X_step.^2 .+ Y_step.^2), s), nqx, nqy)
  end)
  P = @lift(reshape(interior(Val(2), pressure(Thetas[$step]), s), nqx, nqy))

  h1 = heatmap!(ax1, unique_x, unique_y, rho, colormap=Reverse(:oslo))
  Colorbar(fig[1, 2], h1)

  h2 = heatmap!(ax2, unique_x, unique_y, theta, colormap=:thermal)
  Colorbar(fig[1, 4], h2)

  h3 = heatmap!(ax3, unique_x, unique_y, vel_mag, colormap=:viridis)
  Colorbar(fig[2, 2], h3)

  h4 = heatmap!(ax4, unique_x, unique_y, P, colormap=Reverse(:acton))
  Colorbar(fig[2, 4], h4)

  colsize!(fig.layout, 1, Aspect(1, 1.0))
  colsize!(fig.layout, 3, Aspect(1, 1.0))
  resize_to_layout!(fig)

  CairoMakie.record(fig, filename, 1:10:frames; framerate=framerate) do i
    step[] = i
  end
end

create_mp4("imgs/LMNSH/$(simname)_mu=$(μ).mp4", Us, rhos, Thetas; framerate=15)

##############################
##### CONSERVATION CHECKS ####
##############################

# m₀ = sum(interior(Val(2), rho_star_0, s))
# E₀ = sum(interior(Val(2), Theta_star_0, s))

# fig = Figure();
# ax = CairoMakie.Axis(fig[1, 1]; title="Relative Error in Mass", xlabel="Saved Step", ylabel="Relative Error")
# # rhos are dual 0-forms (hdg_2 * rho_star), convert back to primal 2-form for interior sum
# rho_data = [(sum(interior(Val(2), inv_hdg_2 * r, s)) - m₀) / m₀ for r in rhos]
# CairoMakie.lines!(ax, rho_data)
# save("imgs/LMNSH/$(simname)_MassError.png", fig)

# fig = Figure();
# ax = CairoMakie.Axis(fig[1, 1]; title="Relative Error in Thermal Energy", xlabel="Saved Step", ylabel="Relative Error")
# Theta_data = [(sum(interior(Val(2), inv_hdg_2 * Θ, s)) - E₀) / E₀ for Θ in Thetas]
# CairoMakie.lines!(ax, Theta_data)
# save("imgs/LMNSH/$(simname)_EnergyError.png", fig)
