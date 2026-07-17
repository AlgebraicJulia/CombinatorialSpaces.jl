##########################
### Initial Conditions ###
##########################

const ps  = points(s)
const dps = dual_points(s)

# Hydrostatic background: constant potential temperature theta_h = 300 K
const theta_h = 300.0

# RTI perturbation parameters
# D_rti < 0: cold (dense) fluid sits on top of warm (light) fluid → gravitationally unstable
const C_rti           = 10.0   # Interface sharpness (large C → sharper interface)
const D_rti           = -20.0  # Temperature amplitude (K)
const k_rti           = 1.0 / LX  # 2 wavelengths across the domain
const Uz_rti          = 1e-4    # Amplitude of initial velocity perturbation (m/s)
const x_shift_rti      = 0.5    # Shift the sinusoidal velocity perturbation to center upward flow
const y_shift_rti      = 0.5    # Shift the sinusoidal velocity perturbation to center upward flow

const CENTERLINE = LZ / 2

# Perturbed potential temperature at quad centres
const theta_prime = map(dps) do p
    D_rti * tanh(C_rti * (p[3] - CENTERLINE))
end

const theta_total = theta_h .+ theta_prime

# Full density from pressure equilibrium with the background hydrostatic profile:
#   rho = P_h(y)^(1 - R/Cₚ) * P₀^(R/Cₚ) / (R * theta_total)
const R_Cp = R_gas / Cₚ
const rho_total = map(1:nboids(s)) do q
    z   = dps[q][3]
    P_h = hydrostatic_pressure(theta_h, z)
    P_h^(1 - R_Cp) * P₀^(R_Cp) / (R_gas * theta_total[q])
end

const Theta_total  = rho_total .* theta_total
const rho_star_0   = inv_hdg_3(rho_total)
const Theta_star_0 = inv_hdg_3(Theta_total)

# Initial velocity: Gaussian-modulated sinusoidal Y-perturbation at the interface.
# The background X-velocity is zero; only Y-aligned (vertical) primal edges are seeded.
#   u_y(x, y) = 2 * pdf(U_dist, [x, y]) * sin(2π k x)
# Momentum is u_y * rho * dy(s) integrated over each edge.
# Attenuation should only happen in the vertical direction, so the Gaussian is only a function of y.
# const U_dist = Distributions.Normal(ly_ / 2, 0.5)
const U_star_0 = zeros(Float64, nquads(s))
for q in 1:nxyquads(s)
    x, y, z, align = quad_to_coord(s, q)
    v1, v2, v3, v4 = quad_vertices(s, x, y, z, align)
    x_mid = 0.5 * (ps[v1][1] + ps[v3][1])
    y_mid = 0.5 * (ps[v1][2] + ps[v3][2])
    z = ps[v1][3]
    if CENTERLINE - dz(s) <= z <= CENTERLINE + dz(s)
        # rho_e = hydrostatic_density(theta_h, y_mid)
        u_z = Uz_rti * sin(2π * k_rti * (x_mid - x_shift_rti)) * sin(2π * k_rti * (y_mid - y_shift_rti)) # * pdf(U_dist, y_mid)
        # U_star_0[e]  = rho_e * u_y * dy(s)
        U_star_0[q] = u_z * dz(s)
    end
end

# For no slip
bdry_down_xedges = to_device(down_tangent_edges(s))
bdry_up_xedges = to_device(up_tangent_edges(s))
@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat
    v[bdry_down_xedges] .= 0
    v[bdry_up_xedges] .= 0
    return v
end

# @inline function enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat
#     # v .= v .* (FT(1.0) .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
#     return v
# end
  
  @inline function enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat
    U .= U .* (FT(1.0) .- boundary_mask_d) # Enforce no-flux boundary condition on momentum
    return U
end
  
@inline function enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat
    # V .= V .* (FT(1.0) .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
    return V
end