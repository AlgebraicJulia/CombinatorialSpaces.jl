using Random

const Nrat = 0.5
const Hrat = 0.0024 # 1e-4
const minRi = 0.1

# const P₀ = FT(1e5)
# const Cₚ = FT(1006)
# const R_gas = FT(287)
# const gₐ = -FT(9.81)
# const ρᵣ = FT(1) # Reference density
# const θᵣ = FT(300) # Reference potential temperature, temperature at P₀
# const Pᵣ = ρᵣ * R_gas * θᵣ

const CENTERLINE = LZ / 2

# Assuming theta is constant, pass in a float
# function hydrostatic_pressure(theta::FT, h::FT) where FT <: AbstractFloat
#     return (1 / (Cₚ * theta) * (P₀)^(R_gas / Cₚ) * gₐ * h + Pᵣ^(R_gas / Cₚ))^(Cₚ / R_gas)
# end
  
# function hydrostatic_density(theta::FT, h::FT) where FT <: AbstractFloat
#     Pₕ = hydrostatic_pressure(theta, h)
#     return (Pₕ / (R_gas * theta)) * (P₀ / Pₕ)^(R_gas / Cₚ)
# end

const ps  = points(s)
const dps = dual_points(s)

const theta_h = 283.01

const G = 9.81
const theta = map(dps) do p
    z = -(p[3] - CENTERLINE)
    theta_h * (1 - Hrat * z - minRi/G * (1.0 - Nrat^2) * tanh(z))
end

const U = zeros(ne(s))

# TODO: Turn on momentum again
for z in 1:nzb(s), y in 1:nyb(s), x in 1:nx(s)
    q = coord_to_quad(s, x, y, z, X_ALIGN)
    boids, valid = quad_boids(s, x, y, z, X_ALIGN)

    h = (valid[1] ? dps[boids[1]][3] : dps[boids[2]][3]) - CENTERLINE
        
    U[q] = tanh(-h) * dual_edge_len(s, x, y, z, X_ALIGN)
    # U[q] = dual_edge_len(s, x, y, z, X_ALIGN)
end

const R_Cp = R_gas / Cₚ
const rho = map(1:nboids(s)) do b
    z   = dps[b][3] - CENTERLINE
    P_h = hydrostatic_pressure(theta_h, z)
    P_h^(1 - R_Cp) * P₀^(R_Cp) / (R_gas * theta[b])
end

Random.seed!(cart_rank)
rho .+= rand(nboids(s)) * 1e-4

# Per the Scinocca_ paper for minRi = 0.1 and Nrat = 0.5
# Length scale is the half shear height of 0.5m
const k = 0.475 / 0.5

# TODO: Is this the right way of setting up the sinusoidal perturbation?
const dom_wave = map(dps) do p
    z = p[3]
    if CENTERLINE - dz(s) <= z <= CENTERLINE + dz(s)
        x = p[1]
        sin(k * x)
    else
        0
    end
end

theta .+= rand(nboids(s)) .* 1e-4 # .+ dom_wave .* 1e-2 # 1e-3

const Theta = rho .* theta

const rho_star_0 = inv_hdg_3(rho)
const Theta_star_0 = inv_hdg_3(Theta)
const U_star_0 = inv_hdg_2(U)

# TODO: Turn back on this boundary velocity
# This'll need the closure operator for dd1
# For now we just set the boundary velocities to a value that matches the profile
bdry_down_xedges = to_device(down_tangent_edges(s, Val(X_ALIGN)))
bdry_up_xedges = to_device(up_tangent_edges(s, Val(X_ALIGN)))
@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat
    v[bdry_down_xedges] .= dx(s)
    v[bdry_up_xedges] .= -dx(s)
    # v[bdry_up_xedges] .= dx(s)
    return v
end

# @inline function enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat
#     V .= V .* (FT(1.0) .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
#     return V
# end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat
    U .= U .* (FT(1.0) .- boundary_mask_d) # Enforce no-flux boundary condition on momentum
    return U
end
  
