##########################
### Initial Conditions ###
##########################

const ps = points(s)
const dps = dual_points(s)

# Introduce hydrostatic potential temperature background
const theta_h = 300.0 * ones(Float64, nboids(s))
const rho_h = map(dps) do (x, y, z)
  theta = theta_h[1] # Constant background potential temperature
  return hydrostatic_density(theta, z)
end
const Theta_h = theta_h .* rho_h

# Generate a Gaussian perturbation in potential temperature
const theta_dist = MvNormal([LX / 2, LY / 2, LZ / 4], 0.25)
const theta_p_raw = 0.5 * [pdf(theta_dist, [p[1], p[2], p[3]]) for p in dps]

const theta_global_min = MPI.Allreduce(minimum(theta_p_raw), MPI.MIN, cart_comm)
const theta_p_shifted  = theta_p_raw .- theta_global_min

const theta_global_max = MPI.Allreduce(maximum(theta_p_shifted), MPI.MAX, cart_comm)
const theta_p          = theta_p_shifted ./ theta_global_max .* 10

const theta_0 = theta_h .+ theta_p

const rho_0 = Theta_h ./ theta_0

const Theta_0 = rho_0 .* theta_0
const rho_star_0 = to_device(inv_hdg_3(rho_0))
const Theta_star_0 = to_device(inv_hdg_3(Theta_0))
const U_star_0 = to_device(zeros(nquads(s)))

const boundary_mask_edge_cpu = zeros(Float64, ne(s))
const bdry_edges = boundary_tangent_edges(s)
boundary_mask_edge_cpu[bdry_edges] .= FT(1.0)
const boundary_mask_edge = Adapt.adapt(BACKEND, boundary_mask_edge_cpu)

@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat
    v .= v .* (FT(1.0) .- boundary_mask_edge)
    return v
end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat
  U .= U .* (FT(1.0) .- boundary_mask_d) # Enforce no-flux boundary condition on momentum
  return U
end

@inline function enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat
  V .= V .* (FT(1.0) .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
  return V
end