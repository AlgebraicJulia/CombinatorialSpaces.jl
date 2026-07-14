const alpha = FT(50)

function kh_density(y::FT) where {FT}
    return FT(1) + FT(0.5) * tanh(alpha * (y - FT(0.25))) - FT(0.5) * tanh(alpha * (y - FT(0.75)))
end

const ps = points(s)
const dps = dual_points(s)

rho_star_0 = inv_hdg_2(map(dps) do (x, y)
    return kh_density(FT(y))
end)

U_star_0 = zeros(FT, ne(s))
for ex in 1:nxedges(s)
    v1 = src(s, ex)
    v2 = tgt(s, ex)
    xc = FT(0.5) * (ps[v1][1] + ps[v2][1])
    yc = FT(ps[v1][2])
    U_star_0[ex] = FT(0.01) * sin(FT(2π) * xc) * edge_len(s, X_ALIGN) * kh_density(yc)
end
for ey in (nxedges(s) + 1):ne(s)
    v1 = src(s, ey)
    v2 = tgt(s, ey)
    yc = FT(0.5) * (ps[v1][2] + ps[v2][2])
    flow = (FT(0.25) <= yc <= FT(0.75)) ? FT(0.5) : FT(-0.5)
    U_star_0[ey] = flow * edge_len(s, Y_ALIGN) * kh_density(yc)
end

Theta_star_0 = inv_hdg_2(fill(FT(300), nquads(s)))

@inline enforce_bc_v!(v::AbstractVector{FT}) where FT <: AbstractFloat = v
@inline enforce_bc_V!(V::AbstractVector{FT}) where FT <: AbstractFloat = V
@inline enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat = U
