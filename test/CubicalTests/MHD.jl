using CairoMakie
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
##### MHD CONSTANTS      #####
##############################

μ₀ = 4π * 1e-7 # Vacuum permeability
V_s = 1.0       # Sound speed

##############################
##### INITIAL CONDITIONS #####
##############################

# Alfven wave propagating in x along a uniform background magnetic field B₀ x̂.
# The Alfven speed is v_A = B₀ / sqrt(μ₀ ρ₀).
# Perturbations in u_y and B_y are sinusoidal and anti-correlated.

dps = dual_points(s)
simname = "Alfven_Wave"

ρ₀ = 1.0                         # Background density
B₀ = 1.0                         # Background B_x field strength
v_A = B₀ / sqrt(μ₀ * ρ₀)        # Alfven speed
k_wave = 2π / lx_                # Wavenumber (one full wavelength across domain)
δu = 1e-4 * v_A                  # Small velocity perturbation amplitude
δB = δu * sqrt(μ₀ * ρ₀)         # Corresponding B_y perturbation (δB/B₀ = δu/v_A)

# Density: uniform (primal 2-form)
rho_0 = ρ₀ * ones(nquads(s))
rho_star_0 = inv_hdg_2 * rho_0

# Momentum: U = ρu as a primal 1-form
# u_x = 0, u_y = δu sin(kx) → U_y = ρ₀ δu sin(kx) dy
U_star_0 = zeros(ne(s))
for e in (nxedges(s)+1):ne(s)
  x, y = edge_to_coord(s, e)
  x_pos = (x - 1) * dx(s)
  U_star_0[e] = ρ₀ * δu * sin(k_wave * x_pos) * dy(s)
end

# Magnetic field B as a primal 1-form
# B_x = B₀ (constant), B_y = -δB sin(kx)
B_star_0 = zeros(ne(s))
# x-aligned edges: B₀ * dx
for e in 1:nxedges(s)
  B_star_0[e] = B₀ * dx(s)
end
# y-aligned edges: -δB sin(kx) * dy
for e in (nxedges(s)+1):ne(s)
  x, y = edge_to_coord(s, e)
  x_pos = (x - 1) * dx(s)
  B_star_0[e] = -δB * sin(k_wave * x_pos) * dy(s)
end

##################################
##### END INITIAL CONDITIONS #####
##################################

save("imgs/MHD/InitialDensity.png", plot_twoform(s, hdg_2 * rho_star_0))

##############################
##### EQUATIONS OF MOTION ####
##############################

# Momentum equation (eq 23 from derivation):
# ∂U/∂t = -L_{u♯}U + ½ρm ∧ di_{u♯}u - U ∧ δ(u) - V²s ∧ dρm + (1/μ₀)i_{B♯}dB + μΔu
function momentum_mhd(U_star, rho_star, B_star, μ)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star

  # Compute velocity u from momentum U and density rho
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  v = interpolate_dp(Val(1), s, u)

  # U ∧ δu
  div_term = wedge_product_dd(Val(1), Val(0), s, U, dual_δ1 * u)

  # L(u, U) — Lie derivative via Cartan's formula
  L_term = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * U) +
    hdg_1 * wedge_product(Val(1), Val(0), s, v, inv_hdg_0 * dual_d1 * U)

  # ½ ρm ∧ d||u||²
  diff_norm_u = dual_d0 * hdg_2 * wedge_product(Val(1), Val(1), s, v, inv_hdg_1 * u)
  energy = 0.5 * wedge_product_dd(Val(1), Val(0), s, diff_norm_u, rho)

  # V²s ∧ dρm — sound speed pressure gradient
  diff_p = V_s^2 * dual_d0 * rho

  # (1/μ₀) i_{B♯}dB — Lorentz force
  # In 2D: i_{B♯}dB = curl(B) · ⋆B, where curl(B) is ⋆dB (dual 0-form)
  curl_B = hdg_2 * d1 * B_star                                             # dual 0-form (scalar curl)
  lorentz = (1 / μ₀) * wedge_product_dd(Val(0), Val(1), s, curl_B, hdg_1 * B_star)  # dual 1-form

  # μΔu — viscous diffusion
  lap_term = μ * dΔ1 * u

  return -inv_hdg_1 * (-div_term - L_term + energy - diff_p + lap_term + lorentz)
end

# Magnetic induction equation (eq 24 from derivation):
# ∂⋆B/∂t = d(i_{u♯}(⋆B))
# Since d(⋆B) = 0 (div-free), this is the full Lie advection of ⋆B.
# We evolve B_star (primal 1-form) via: ∂B_star/∂t = inv_hdg_1 * d(i_{u♯}(⋆B))
# where i_{u♯}(⋆B) = -⋆(u ∧ B) is a dual 0-form.
function magnetic_induction(U_star, rho_star, B_star, η)
  U = hdg_1 * U_star
  rho = hdg_2 * rho_star

  # Velocity from momentum/density
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)

  # Interpolate u to primal edges
  v = interpolate_dp(Val(1), s, u)

  # i_{u♯}(⋆B) = -⋆(u ∧ B) as a dual 0-form
  contraction = -hdg_2 * wedge_product(Val(1), Val(1), s, v, B_star)

  # d(i_{u♯}(⋆B)) as a dual 1-form
  induction = dual_d0 * contraction

  # ηΔB — resistive diffusion (η = 0 for ideal MHD)
  diffusion = η * dΔ1 * (hdg_1 * B_star)

  return inv_hdg_1 * (induction + diffusion)
end

##############################
##### SMOOTHING          #####
##############################

# Smoothing for dual 0-forms (which live on quads/faces).
# Each quad is averaged with its face-adjacent neighbors (sharing an edge),
# weighted by the inverse distance between quad centers.
function smoothing_dual0(s::UniformCubicalComplex2D, c_smooth)
  n = nquads(s)
  c = c_smooth / 2
  inv_dx = 1 / dx(s)
  inv_dy = 1 / dy(s)
  nqx = nxquads(s)
  nqy = nyquads(s)

  # Pre-allocate COO arrays (at most 5 entries per quad: self + 4 neighbors)
  max_nnz = 5 * n
  I = Vector{Int}(undef, max_nnz)
  J = Vector{Int}(undef, max_nnz)
  V = Vector{Float64}(undef, max_nnz)
  idx = 0

  for q in quads(s)
    x, y = quad_to_coord(s, q)

    # Collect neighbor weights
    tot_w = 0.0
    neighbors = 0

    has_left  = x > 1
    has_right = x < nqx
    has_down  = y > 1
    has_up    = y < nqy

    tot_w += (has_left + has_right) * inv_dx + (has_down + has_up) * inv_dy

    if tot_w > 0
      scale = c / tot_w
      if has_left
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x - 1, y); V[idx] = scale * inv_dx
      end
      if has_right
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x + 1, y); V[idx] = scale * inv_dx
      end
      if has_down
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x, y - 1); V[idx] = scale * inv_dy
      end
      if has_up
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x, y + 1); V[idx] = scale * inv_dy
      end
    end

    # Diagonal
    idx += 1; I[idx] = q; J[idx] = q; V[idx] = 1 - c
  end

  return sparse(view(I, 1:idx), view(J, 1:idx), view(V, 1:idx), n, n)
end

c_smooth = 0.2
forward_smooth = smoothing_dual0(s, c_smooth)
backward_smooth = smoothing_dual0(s, -c_smooth)
smoothing_mat = backward_smooth * forward_smooth

##############################
##### TIME INTEGRATION   #####
##############################

function run_compressible_mhd(U_star_0, rho_star_0, B_star_0, tₑ, Δt; saveat=500)

  U_star = deepcopy(U_star_0)
  rho_star = deepcopy(rho_star_0)
  B_star = deepcopy(B_star_0)

  U_star_half = zeros(Float64, ne(s))
  B_star_half = zeros(Float64, ne(s))

  rho_star_half = zeros(Float64, nquads(s))
  rho_star_full = zeros(Float64, nquads(s))

  steps = ceil(Int64, tₑ / Δt)

  Us = [Array(hdg_1 * U_star_0)]
  rhos = [Array(hdg_2 * rho_star_0)]
  Bs = [Array(B_star_0)]

  m₀ = sum(interior(Val(2), rho_star_0, s))

  for step in 1:steps

    set_periodic!(U_star, Val(1), s, ALL)
    set_periodic!(rho_star, Val(2), s, ALL)
    set_periodic!(B_star, Val(1), s, ALL)

    # Half step for momentum and magnetic field
    U_star_half .= U_star .+ 0.5 * Δt * momentum_mhd(U_star, rho_star, B_star, μ)
    B_star_half .= B_star .+ 0.5 * Δt * magnetic_induction(U_star, rho_star, B_star, η)

    set_periodic!(U_star_half, Val(1), s, ALL)
    set_periodic!(B_star_half, Val(1), s, ALL)

    # Full step for density (mass flux)
    rho_star_full .= smoothing_mat * (rho_star + Δt * d1 * U_star_half)
    set_periodic!(rho_star_full, Val(2), s, ALL)

    rho_star_half .= 0.5 .* (rho_star .+ rho_star_full)

    # Full step for momentum and magnetic field
    U_star .= U_star .+ Δt * momentum_mhd(U_star_half, rho_star_half, B_star_half, μ)
    B_star .= B_star .+ Δt * magnetic_induction(U_star_half, rho_star_half, B_star_half, η)
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
    elseif any(isnan.(B_star))
      println("Warning, NAN result in B at step: $(step)")
      break
    elseif any(isinf.(B_star))
      println("Warning, INF result in B at step: $(step)")
      break
    end

    if step % saveat == 0
      push!(Us, hdg_1 * U_star)
      push!(rhos, hdg_2 * rho_star)
      push!(Bs, B_star)
      println("Step $(step)/$(steps) ($(@sprintf("%.1f", (step / steps) * 100))%)")
    end
  end

  return Us, rhos, Bs
end

##############################
##### RUN SIMULATION     #####
##############################

μ = 1e-3  # Momentum diffusivity (viscosity)
η = 0.0   # Magnetic resistivity (0 for ideal MHD)

tₑ = 2e-2
Δt = 1e-5
Us, rhos, Bs = run_compressible_mhd(U_star_0, rho_star_0, B_star_0, tₑ, Δt; saveat=1);

##############################
##### VISUALIZATION      #####
##############################

timestep = length(Us)

# Vorticity
ω_end = inv_hdg_0 * dual_d1 * Us[timestep];
save("imgs/MHD/Vorticity.png", plot_zeroform(s, ω_end))

# Density
save("imgs/MHD/FinalDensity.png", plot_twoform(s, rhos[timestep]))

# Velocity Magnitude
u_end = wedge_product_dd(Val(0), Val(1), s, 1 ./ rhos[timestep], Us[timestep]);
X_end = zeros(nquads(s)); Y_end = zeros(nquads(s));
sharp_dd!(X_end, Y_end, s, u_end);
vel_mag_end = sqrt.(X_end.^2 .+ Y_end.^2);
save("imgs/MHD/VelocityMagnitude.png", plot_twoform(s, vel_mag_end))

# Magnetic Field Magnitude
BX_end = zeros(nquads(s)); BY_end = zeros(nquads(s));
sharp_dd!(BX_end, BY_end, s, hdg_1 * Bs[timestep]);
B_mag_end = sqrt.(BX_end.^2 .+ BY_end.^2);
save("imgs/MHD/MagneticFieldMagnitude.png", plot_twoform(s, B_mag_end))

# Magnetic Field Curl (current density J_z = curl(B)/μ₀)
curl_B_end = hdg_2 * d1 * Bs[timestep];
save("imgs/MHD/CurrentDensity.png", plot_twoform(s, curl_B_end ./ μ₀))

# Momentum Divergence
U_div = inv_hdg_0 * dual_d1 * Us[timestep];
save("imgs/MHD/MomentumDivergence.png", plot_zeroform(s, U_div))

##############################
##### ANIMATION          #####
##############################

function create_mp4(filename, Us, rhos, Bs; frames=length(Us), framerate=15)

  nqx = nxquads(s) - 2 * hx(s)
  nqy = nyquads(s) - 2 * hy(s)

  dps = interior(Val(2), dual_points(s), s)
  unique_x = sort(unique(map(a -> a[1], dps)))
  unique_y = sort(unique(map(a -> a[2], dps)))

  fig = Figure(size=(1200, 800))
  ax1 = Axis(fig[1, 1], title="Density")
  ax2 = Axis(fig[1, 3], title="Magnetic Field Magnitude")
  ax3 = Axis(fig[2, 1], title="Velocity Magnitude")
  ax4 = Axis(fig[2, 3], title="Current Density (Jz)")

  step = Observable(1)

  rho = @lift(reshape(interior(Val(2), rhos[$step], s), nqx, nqy))
  B_mag = @lift(begin
    BX_step = zeros(nquads(s)); BY_step = zeros(nquads(s))
    sharp_dd!(BX_step, BY_step, s, hdg_1 * Bs[$step])
    reshape(interior(Val(2), sqrt.(BX_step.^2 .+ BY_step.^2), s), nqx, nqy)
  end)
  vel_mag = @lift(begin
    u_step = wedge_product_dd(Val(0), Val(1), s, 1 ./ rhos[$step], Us[$step])
    X_step = zeros(nquads(s)); Y_step = zeros(nquads(s))
    sharp_dd!(X_step, Y_step, s, u_step)
    reshape(interior(Val(2), sqrt.(X_step.^2 .+ Y_step.^2), s), nqx, nqy)
  end)
  Jz = @lift(reshape(interior(Val(2), (hdg_2 * d1 * Bs[$step]) ./ μ₀, s), nqx, nqy))

  h1 = heatmap!(ax1, unique_x, unique_y, rho, colormap=Reverse(:oslo))
  Colorbar(fig[1, 2], h1)

  h2 = heatmap!(ax2, unique_x, unique_y, B_mag, colormap=:inferno)
  Colorbar(fig[1, 4], h2)

  h3 = heatmap!(ax3, unique_x, unique_y, vel_mag, colormap=:viridis)
  Colorbar(fig[2, 2], h3)

  h4 = heatmap!(ax4, unique_x, unique_y, Jz, colormap=Reverse(:acton))
  Colorbar(fig[2, 4], h4)

  colsize!(fig.layout, 1, Aspect(1, 1.0))
  colsize!(fig.layout, 3, Aspect(1, 1.0))
  resize_to_layout!(fig)

  CairoMakie.record(fig, filename, 1:10:frames; framerate=framerate) do i
    step[] = i
  end
end

create_mp4("imgs/MHD/$(simname)_mu=$(μ)_eta=$(η).mp4", Us, rhos, Bs; framerate=15)
