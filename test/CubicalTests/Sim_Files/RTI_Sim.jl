##########################
### Initial Conditions ###
##########################

println("Running RTI Simulation...")

# Domain: 2 × 16 m, 65 × 129 vertices (dx = dy = 0.0625 m)
# halo_x = 10 to support periodic left/right boundaries via SSPRK33 halo scheme
const lx_ = 2.0
const ly_ = 16.0
const nx_ = 65
const ny_ = 513
const halo_x = 10
const halo_y = 0
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_; halo_x = halo_x, halo_y = halo_y)

const ps  = points(s)
const dps = dual_points(s)

const inv_hdg_2 = inv_hodge_star(Val(2), s)

# Hydrostatic background: constant potential temperature theta_h = 300 K
const theta_h = 300.0

# RTI perturbation parameters
# D_rti < 0: cold (dense) fluid sits on top of warm (light) fluid → gravitationally unstable
const C_rti           = 10.0   # Interface sharpness (large C → sharper interface)
const D_rti           = -20.0  # Temperature amplitude (K)
const k_rti           = 1.0 / lx_  # 2 wavelengths across the domain
const Uy_rti          = 0.5    # Amplitude of initial velocity perturbation (m/s)
const x_shift_rti      = 0.5    # Shift the sinusoidal velocity perturbation to center upward flow

# Perturbed potential temperature at quad centres
const theta_prime = map(dps) do p
    D_rti * tanh(C_rti * (p[2] - ly_ / 2))
end

const theta_total = theta_h .+ theta_prime

# Full density from pressure equilibrium with the background hydrostatic profile:
#   rho = P_h(y)^(1 - R/Cₚ) * P₀^(R/Cₚ) / (R * theta_total)
const R_Cp = R_gas / Cₚ
const rho_total = map(1:nquads(s)) do q
    y   = dps[q][2]
    P_h = hydrostatic_pressure(theta_h, y)
    P_h^(1 - R_Cp) * P₀^(R_Cp) / (R_gas * theta_total[q])
end

const Theta_total  = rho_total .* theta_total
const rho_star_0   = inv_hdg_2 * rho_total
const Theta_star_0 = inv_hdg_2 * Theta_total

# Initial velocity: Gaussian-modulated sinusoidal Y-perturbation at the interface.
# The background X-velocity is zero; only Y-aligned (vertical) primal edges are seeded.
#   u_y(x, y) = 2 * pdf(U_dist, [x, y]) * sin(2π k x)
# Momentum is u_y * rho * dy(s) integrated over each edge.
# Attenuation should only happen in the vertical direction, so the Gaussian is only a function of y.
# const U_dist = Distributions.Normal(ly_ / 2, 0.5)
const U_star_0 = let
    U = zeros(Float64, ne(s))
    for e in 1:nxedges(s)
        v1    = src(s, e)
        v2    = tgt(s, e)
        x_mid = 0.5 * (ps[v1][1] + ps[v2][1])
        y_mid = 0.5 * (ps[v1][2] + ps[v2][2])
        rho_e = hydrostatic_density(theta_h, y_mid)
        u_y   = Uy_rti * sin(2π * k_rti * (x_mid - x_shift_rti)) # * pdf(U_dist, y_mid)
        U[e]  = rho_e * u_y * dy(s)
    end
    U
end

# Gravity: non-zero on X-aligned (horizontal) primal edges whose dual edges are vertical.
# The wedge_product_dd body-force term uses this 1-form together with the density 0-form.
const g_dual = to_device(map(edges(s)) do e
    is_X_aligned(e, s) ? gₐ * dual_edge_len(s, e) : 0.0
end)

# Physical parameters (dry air, non-dimensional Reynolds and Prandtl numbers)
const Pr     = 0.7
const Re     = 10_000.0
const mu     = 1 / Re
const kappa  = mu / Pr
const p      = (mu = mu, kappa = kappa)

const te = 5.0
const dt = 5e-5

const use_gravity        = true
const full_periodic      = false
const periodic_left_right = true
const periodic_top_bottom = false

###########################
### Boundary Conditions ###
###########################

# Top and bottom walls are solid (no-slip / no-flux).
# Left and right boundaries are periodic via halos — do NOT zero those edges here.
const boundary_mask_d = to_device(zeros(Float64, ne(s)))
boundary_mask_d[top_edges(s)]    .= 1.0
boundary_mask_d[bottom_edges(s)] .= 1.0

@inline function enforce_bc_v!(v)
    v .= v .* (1.0 .- boundary_mask_d)
    return v
end

@inline function enforce_bc_U!(U)
    U .= U .* (1.0 .- boundary_mask_d)
    return U
end

@inline function enforce_bc_V!(V)
    V .= V .* (1.0 .- boundary_mask_d)
    return V
end

#########################
### Saving Parameters ###
#########################

print_re = @sprintf("%.0f", Re)
print_te = @sprintf("%.2f", te)
simspec   = "RTI_Re=$(print_re)_te=$(print_te)"
save_path = "/blue/fairbanksj/grauta/simulations/LMNS_RTI/$(simspec)"
mkpath(save_path)

const saveat        = 250
const checkpoint_at = saveat * 40

const T_smooth_constant = 0.2
const r_smooth_constant = 0.05

# Initial diagnostic plots
fig = plot_twoform(s, hodge_star(Val(2), s) * rho_star_0)
save(joinpath(save_path, "initial_density.png"), fig)

fig = plot_xy_oneform(s, inv_hodge_star(Val(1), s) * U_star_0)
save(joinpath(save_path, "initial_velocity.png"), fig)

fig = plot_twoform(s, hodge_star(Val(2), s) * Theta_star_0)
save(joinpath(save_path, "initial_potential_temperature.png"), fig)

fig = plot_xy_oneform(s, Array(g_dual))
save(joinpath(save_path, "initial_gravity_sharp.png"), fig)
