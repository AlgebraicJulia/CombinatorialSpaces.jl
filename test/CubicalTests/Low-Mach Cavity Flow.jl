using Test
using CairoMakie
using LinearAlgebra
using Printf

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

lx_ = ly_ = 1.0;
nx_ = ny_ = 129
s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_)

# DEC Operators
Δ0 = laplacian(Val(0), s);

d0 = exterior_derivative(Val(0), s);
d1 = exterior_derivative(Val(1), s);

dual_d0 = dual_derivative(Val(0), s);
dual_d1 = dual_derivative(Val(1), s);

# TODO: Use this if we don't want periodic boundaries
d_beta = 0.5 * abs.(dual_d1) * spdiagm(dual_d0 * ones(nquads(s)));

boundary_idxs = findall(x -> x != 0, dual_d0 * ones(nquads(s)));
top = top_edges(s);

dual_d0[boundary_idxs, :] .= 0.0 # Enforce no-flux boundary condition on density

hdg_1 = hodge_star(Val(1), s);
hdg_2 = hodge_star(Val(2), s);

inv_hdg_0 = inv_hodge_star(Val(0), s);
inv_hdg_1 = inv_hodge_star(Val(1), s);
inv_hdg_2 = inv_hodge_star(Val(2), s);

δ1 = codifferential(Val(1), s);
dual_δ1 = dual_codifferential(Val(1), s);

# Enforce closure of dual cells and no symmetric boundary conditions on velocity
dΔ1 = hdg_1 * d0 * inv_hdg_0 * dual_d1 + dual_d0 * hdg_2 * d1 * inv_hdg_1
dΔ1_V = hdg_1 * d0 * inv_hdg_0 * d_beta

u_star_0 = zeros(ne(s))

function momentum_continuity(U_star, rho_star, p)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  κ, μ = p

  U[boundary_idxs] .= 0.0 # Enforce no-flux boundary condition on momentum

  # Compute velocity u from momentum U and density rho
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  v = interpolate_dp(Val(1), s, u)
  V = interpolate_dp(Val(1), s, U)

  v[boundary_idxs] .= 0.0 # Enforce no-slip boundary condition on velocity
  v[top] .= edge_len(s, X_ALIGN) # Lid velocity

  # U ∧ δu
  div_term = wedge_product_dd(Val(1), Val(0), s, U, dual_δ1 * u)

  # L(u, U)
  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(1), Val(0), s, v, inv_hdg_0 * (dual_d1 * U + d_beta * V))

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 * wedge_product_dd(Val(1), Val(0), s, diff_norm_u, rho)

  # dP, P = κρ
  diff_p = dual_d0 * (κ * rho)

  # μΔu
  lap_term = μ * (dΔ1 * u + dΔ1_V * v)

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term)
end

function run_compressible_ns(U_star_0, rho_star_0, tₑ, Δt, p; saveat=500)

  U_star = deepcopy(U_star_0)
  rho_star = deepcopy(rho_star_0)

  U_star_half = zeros(Float64, ne(s))

  rho_star_half = zeros(Float64, nquads(s))
  rho_star_full = zeros(Float64, nquads(s))

  steps = ceil(Int64, tₑ / Δt)

  Us = [Array(hdg_1 * U_star_0)]
  rhos = [Array(hdg_2 * rho_star_0)]

  m₀ = sum(interior(Val(2), rho_star_0, s))

  for step in 1:steps
    U_star_half .= U_star .+ 0.5 * Δt * momentum_continuity(U_star, rho_star, p)

    rho_star_full .= rho_star + Δt * d1 * U_star_half # Mass changes by momentum flux
    rho_star_half .= 0.5 .* (rho_star .+ rho_star_full)

    U_star .= U_star .+ Δt * momentum_continuity(U_star_half, rho_star_half, p)
    rho_star .= rho_star_full

    if any(isnan, U_star)
      println("Warning, NAN result in U at step: $(step)")
      break
    elseif any(isinf, U_star)
      println("Warning, INF result in U at step: $(step)")
      break
    elseif any(isnan, rho_star)
      println("Warning, NAN result in rho_star at step: $(step)")
      break
    elseif any(isinf, rho_star)
      println("Warning, INF result in rho_star at step: $(step)")
      break
    end

    if step % saveat == 0
      push!(Us, hdg_1 * U_star)
      push!(rhos, hdg_2 * rho_star)
      println("Loading simulation results: $((step / steps) * 100)%")
      println("Relative mass is : $((sum(interior(Val(2), rho_star, s)) / m₀) * 100)%")
      println("-----")
    end
  end

  return Us, rhos
end

Re = 10
κ = 280 * 300 # R (Dry gas constant) * T = 300K, from P = ρRVT
p = (κ, 1 / Re) # κ, μ
rho_star_0 = inv_hdg_2 * ones(nquads(s))
U_star_0 = deepcopy(u_star_0)

Us, rhos = run_compressible_ns(U_star_0, rho_star_0, 1e-2, 1e-5, p; saveat=10);

time = length(Us)
ω = inv_hdg_0 * dual_d1 * Us[time];

fig = plot_zeroform(s, ω)
save("imgs/LMNS_Cav/FinalVortices.png", fig)

fig = plot_twoform(s, rhos[time])
save("imgs/LMNS_Cav/FinalDensity.png", fig)

function create_mp4(filename, Us, rhos; frames = length(Us), framerate = 15)

  fig = Figure(size=(1000, 1000))
  ax1 = Axis(fig[1, 1], title = "Density Field (p)")
  ax2 = Axis(fig[2, 1], title = "Velocity Magnitude")

  dps = interior(Val(2), dual_points(s), s)
  xs = map(a -> a[1], dps)
  ys = map(a -> a[2], dps)

  step = Observable(1)

  rho = @lift(interior(Val(2), rhos[$step], s))
  vel_mag = @lift(begin
    u_step = wedge_product_dd(Val(0), Val(1), s, 1 ./ rhos[$step], Us[$step])
    X_step = zeros(nquads(s)); Y_step = zeros(nquads(s))
    sharp_dd!(X_step, Y_step, s, u_step)
    interior(Val(2), sqrt.(X_step.^2 .+ Y_step.^2), s)
  end)

  scatter1 = heatmap!(ax1, xs, ys, rho, colormap=Reverse(:oslo))
  scatter2 = heatmap!(ax2, xs, ys, vel_mag, colormap=:viridis)

  Colorbar(fig[1, 2], scatter1, tickformat = ticks -> [@sprintf("%.1f", t) for t in ticks])
  Colorbar(fig[2, 2], scatter2)

  colsize!(fig.layout, 1, Aspect(1, 1.0))
  resize_to_layout!(fig)

  CairoMakie.record(fig, filename, 1:frames; framerate = framerate) do i
    step[] = i
  end
end

create_mp4("imgs/LMNS_Cav/simulation.mp4", Us, rhos; framerate = 15)
