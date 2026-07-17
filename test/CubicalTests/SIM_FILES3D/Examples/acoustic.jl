Theta_dist    = MvNormal([LX/2, LY/2, LZ/2], [0.25, 0.25, 0.25])
Theta_perturb = zeros(FT, nboids(s))
for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
    b  = coord_to_boid(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s))
    dp = real_dual_point(s, rx, ry, rz)
    Theta_perturb[b] = pdf(Theta_dist, [dp[1], dp[2], dp[3]]) * 0.1
end

# U_star is a primal 2-form: zero initial velocity → nquads(s) zeros
U_star_0     = zeros(FT, nquads(s))
rho_star_0   = inv_hdg_3(ones(FT, nboids(s)))
Theta_star_0 = inv_hdg_3(fill(FT(300), nboids(s)) .+ Theta_perturb)

@inline enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat = U
@inline enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat = v
@inline enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat = V
