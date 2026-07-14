const inv_hdg_2 = x -> inv_hodge_star(Val(2), cache, x)

# TODO: Should streamline this process to avoid future errors
Theta_dist = MvNormal([LX/2, LY/2], [0.1, 0.1])
Theta_perturb = zeros(FT, nquads(s))
for ry in 1:nyqr(s), rx in 1:nxqr(s)
    q = coord_to_quad(s, rx + halo_west(s), ry + halo_south(s))
    dp = real_dual_point(s, rx, ry)
    Theta_perturb[q] = pdf(Theta_dist, [dp[1], dp[2]]) * 0.5
end

U_star_0 = to_device(zeros(FT, ne(s)))
rho_star_0 = to_device(inv_hdg_2(ones(FT, nquads(s))))
Theta_star_0 = to_device(inv_hdg_2(fill(FT(300), nquads(s)) .+ Theta_perturb))

@inline enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat = U
@inline enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat = v
@inline enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat = V
