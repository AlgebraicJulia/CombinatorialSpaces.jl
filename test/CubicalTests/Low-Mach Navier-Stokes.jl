using Test
using CairoMakie
using LinearAlgebra

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

lx_ = ly_ = 2π;
nx_ = ny_ = 81
s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_; halo_x = 5, halo_y = 5)

function form_taylor_vortices(s::AbstractCubicalComplex2D, G::Real, a::Real, centers)
  u = zeros(nv(s))

  function taylor_vortex(pnt::Point3d, cntr::Point3d)
    r = norm(pnt .- cntr)
    (G / a) * (2 - (r / a)^2) * exp(0.5 * (1 - (r / a)^2))
  end

  for center in centers
    u += map(p -> taylor_vortex(p, center), points(s))
  end

  return u
end

left_cnt = Point3d(lx_ / 2 - 0.4, ly_ / 2, 0.0); right_cnt = Point3d(lx_ / 2 + 0.4, ly_ / 2, 0.0)
ω = form_taylor_vortices(s, 1.0, 0.3, [left_cnt, right_cnt])

save("imgs/LMNS/Vortices.png", plot_zeroform(s, ω))

# DEC Operators
Δ0 = laplacian(Val(0), s);
dΔ1 = dual_laplacian(Val(1), s);

d0 = exterior_derivative(Val(0), s);
d1 = exterior_derivative(Val(1), s);

dual_d0 = dual_derivative(Val(0), s);
dual_d1 = dual_derivative(Val(1), s);

# TODO: Use this if we don't want periodic boundaries
d_beta = 0.5 * abs.(dual_d1) * spdiagm(dual_d0 * ones(nquads(s)));

hdg_1 = hodge_star(Val(1), s);
hdg_2 = hodge_star(Val(2), s);

inv_hdg_0 = inv_hodge_star(Val(0), s);
inv_hdg_1 = inv_hodge_star(Val(1), s);
inv_hdg_2 = inv_hodge_star(Val(2), s);

δ1 = codifferential(Val(1), s);
dual_δ1 = dual_codifferential(Val(1), s);

# Solve for initial conditions
ψ = Δ0 \ ω;
ψ .= ψ .- minimum(ψ);

u_star_0 = d0 * ψ;
ω_test = δ1 * u_star_0;

save("imgs/LMNS/RoundTrip_Vortices.png", plot_zeroform(s, ω_test))

function momentum_continuity(U_star, rho_star, p)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star
  κ, μ = p

  # Compute velocity u from momentum U and density rho
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)
  
  # Interpolate u to primal edges
  X = zeros(nquads(s)); Y = zeros(nquads(s))
  v = zeros(ne(s))

  sharp_dd!(X, Y, s, u)
  flat_dp!(v, s, X, Y) # TODO: Can we get rid of the need for v?

  # U ∧ δu
  div_term = wedge_product_dd(Val(1), Val(0), s, U, dual_δ1 * u)

  # L(u, U)
  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) + 
    hdg_1 * wedge_product(Val(1), Val(0), s, v, inv_hdg_0 * dual_d1 * U)

  # 1/2 * ρ * d||u||^2
  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 * wedge_product_dd(Val(1), Val(0), s, diff_norm_u, rho)

  # dP, P = κρ
  diff_p = dual_d0 * (κ * rho)

  # μΔu
  lap_term = μ * dΔ1 * u

  println("Extrema of terms: ")
  println("  Div: $(minimum(div_term)) to $(maximum(div_term))")
  println("  L: $(minimum(L_term)) to $(maximum(L_term))")
  println("  Energy: $(minimum(energy)) to $(maximum(energy))")
  println("  Diff P: $(minimum(diff_p)) to $(maximum(diff_p))")
  println("  Laplacian: $(minimum(lap_term)) to $(maximum(lap_term))")

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term)
end

  # return -invhdg1 * (-wdg_10dd(U, codif_1 * u) - # U ∧ δu
  #        dd0_hdg2 * wdg_11(v, invhdg1 * U) - # L(u, U)
  #        hdg1 * wdg_10(v, invhdg0_dd1 * U) +
  #        0.5 * wdg_10dd(dd0_hdg2 * wdg_11(v, invhdg1 * u), ρ) - # 1/2 * ρ * d||u||^2
  #        cu_dual_d0 * (κ * ρ) + # dP, P = κρ
  #        μ * lap_term * u) # μΔu

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

    set_periodic!(U_star, Val(1), s, ALL)
    set_periodic!(rho_star, Val(2), s, ALL)

    U_star_half .= U_star .+ 0.5 * Δt * momentum_continuity(U_star, rho_star, p)

    rho_star_full .= rho_star + Δt * d1 * U_star_half # Mass changes by momentum flux
    rho_star_half .= 0.5 .* (rho_star .+ rho_star_full)

    U_star .= U_star .+ Δt * momentum_continuity(U_star_half, rho_star_half, p)
    rho_star .= rho_star_full

    if any(isnan.(U_star))
      println("Warning, NAN result in U at step: $(step)")
      break
    elseif any(isinf.(U_star))
      println("Warning, INF result in U at step: $(step)")
      break
    elseif any(isnan.(rho_star))
      println("Warning, NAN result in rho_star at step: $(step)")
      break
    elseif any(isinf.(rho_star))
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

Re = Inf # 1000.0
κ = 280 * 300 # R (Dry gas constant) * T = 300K, from P = ρRVT
p = (κ, 1 / Re) # κ, μ
rho_star_0 = inv_hdg_2 * ones(nquads(s))
U_star_0 = deepcopy(u_star_0)

Us, rhos = run_compressible_ns(U_star_0, rho_star_0, 1e-3, 1e-4, p; saveat=25);

time = 5; # length(Us)
ω = inv_hdg_0 * dual_d1 * Us[time];

fig = plot_zeroform(s, ω)
save("imgs/LMNS/FinalVortices.png", fig)

fig = plot_twoform(s, rhos[time])
save("imgs/LMNS/FinalDensity.png", fig)

function create_mp4(filename, Us, rhos; frames = length(Us), framerate = 15)

    fig = Figure()
    ax1 = Axis(fig[1, 1], title = "Density Field (p)")
    ax2 = Axis(fig[2, 1], title = "Vorticity Field")

    dps = interior(Val(2), dual_points(s), s)
    xs = map(a -> a[1], dps)
    ys = map(a -> a[2], dps)

    step = Observable(1)

    rho = @lift(interior(Val(2), rhos[$step], s))
    rot = @lift(interior(Val(0), inv_hdg_0 * dual_d1 * Us[$step], s))

    scatter1 = scatter!(ax1, xs, ys, color = rho, colormap=Reverse(:oslo))
    scatter2 = mesh!(ax2, s, color = rot, colormap=:viridis, colorrange=(-5, 12))

    Colorbar(fig[1, 2], scatter1)
    Colorbar(fig[2, 2], scatter2)

    CairoMakie.record(fig, filename, 1:frames; framerate = framerate) do i
        step[] = i
    end
end

create_mp4("imgs/LMNS/simulation.mp4", Us, rhos; framerate = 15)
