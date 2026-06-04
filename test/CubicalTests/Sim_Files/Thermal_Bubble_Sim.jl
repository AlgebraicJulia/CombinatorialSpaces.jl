##########################
### Initial Conditions ###
##########################

println("Running Thermal Bubble Simulation...")

# # WARM BUBBLE
# simname = "Perturbed_Warm_Bubble_Gradient"
# thetaₕ = 300 * CUDA.ones(Float64, nv(sd))
# Thetaₕ, ρₕ = generate_hydrostatic_vars(sd)
# Pₕ = exact_pressure(Thetaₕ)

# U₀ = CUDA.zeros(Float64, ne(sd))
# theta_dist = MvNormal([lx / 2, ly / 4], 0.25)
# theta′ = 10 * CuArray{Float64}([pdf(theta_dist, [p[1], p[2]]) for p in sd[:point]])
# plot_zeroform(s, Array(theta′))

# ρ′ = (Pₕ .^ (1 - R / Cₚ) * P₀^(R / Cₚ)) ./ (R .* (thetaₕ .+ theta′)) .- ρₕ
# plot_zeroform(s, Array(ρ′))
# Theta′ = ρ′ .* thetaₕ .+ ρₕ .* theta′ .+ ρ′ .* theta′
# plot_zeroform(s, Array(Theta′))

const lx_ = const ly_ = 6.0;
# const nx_ = const ny_ = 129
const nx_ = const ny_ = 65
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_)

const ps = points(s)
const dps = dual_points(s)

const inv_hdg_2 = inv_hodge_star(Val(2), s)

# Introduce hydrostatic potential temperature background

const theta_h = 300.0 * ones(Float64, nquads(s))
const rho_h = map(dps) do (x, y)
  theta = theta_h[1] # Constant background potential temperature
  return hydrostatic_density(theta, y)
end
const Theta_h = theta_h .* rho_h

const U_star_0 = zeros(ne(s))

# Generate a Gaussian perturbation in potential temperature
const theta_dist = MvNormal([lx_ / 2, ly_ / 4], 0.25)
const theta_p = 0.5 * [pdf(theta_dist, [p[1], p[2]]) for p in dps]
const theta_p = theta_p .- minimum(theta_p)
const theta_p = theta_p ./ maximum(theta_p) .* 10

const theta_0 = theta_h .+ theta_p

const rho_0 = Theta_h ./ theta_0

const Theta_0 = rho_0 .* theta_0
const rho_star_0 = inv_hdg_2 * rho_0
const Theta_star_0 = inv_hdg_2 * Theta_0

# const rho_star_0 = inv_hdg_2 * rho_h
# const Theta_star_0 = inv_hdg_2 * Theta_h

# Gravity parameters
# This is on the dual grid, so we multiply by the dual edge length to get the correct force per unit mass
const g_dual = to_device(map(edges(s)) do e
  if is_X_aligned(e, s)
    return gₐ * dual_edge_len(s, e)
  else
    return 0.0
  end
end)

# For dry air
const Pr = 0.7
const Re = 10_000.0
const mu = 1 / Re # Momentum diffusivity
const kappa = mu / Pr # Thermal diffusivity

const p = (mu = mu, kappa = kappa)

const te = 40.0
const dt = 1e-4 # dx / 360

const use_gravity = true
const full_periodic = false
const periodic_left_right = false
const periodic_top_bottom = false

###########################
### Boundary Conditions ###
###########################

const boundary_mask_d = to_device(zeros(Float64, ne(s))); boundary_mask_d[boundary_edges(s)] .= 1.0

@inline function enforce_bc_v!(v)
  v .= v .* (1.0 .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
  return v
end

@inline function enforce_bc_U!(U)
  U .= U .* (1.0 .- boundary_mask_d) # Enforce no-flux boundary condition on momentum
  return U
end

@inline function enforce_bc_V!(V)
  V .= V .* (1.0 .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
  return V
end

#########################
### Saving Parameters ###
#########################

print_re = @sprintf("%.0f", Re)
print_te = @sprintf("%.2f", te)
simspec = "Re=$(print_re)_te=$(print_te)"
savepath = "/blue/fairbanksj/grauta/simulations/LMNS_ThermalBubble/$(simspec)"
mkpath(savepath)

const saveat = 1000
const checkpoint_at = saveat * 40

const T_smooth_constant = 0.2
const Theta_smoothing = to_device(smoothing_dual0(s, -T_smooth_constant) * smoothing_dual0(s, T_smooth_constant))

const r_smooth_constant = 0.05
const rho_smoothing = to_device(smoothing_dual0(s, -r_smooth_constant) * smoothing_dual0(s, r_smooth_constant))

# Plot the density field at the start of the simulation
fig = plot_twoform(s, hodge_star(Val(2), s) * rho_star_0)
save(joinpath(savepath, "initial_density.png"), fig)

fig = plot_xy_oneform(s, inv_hodge_star(Val(1), s) * U_star_0)
save(joinpath(savepath, "initial_velocity.png"), fig)

fig = plot_twoform(s, hodge_star(Val(2), s) * Theta_star_0)
save(joinpath(savepath, "initial_potential_temperature.png"), fig)

fig = plot_xy_oneform(s, Array(g_dual))
save(joinpath(savepath, "initial_gravity_sharp.png"), fig)
