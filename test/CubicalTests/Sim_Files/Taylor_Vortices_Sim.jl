##########################
### Initial Conditions ###
##########################

println("Running Taylor Vortices Simulation...")

function form_taylor_vortices_initial_conditions(s::AbstractCubicalComplex2D)

  left_cnt = Point3d(lx_ / 2 - 0.4, ly_ / 2, 0.0)
  right_cnt = Point3d(lx_ / 2 + 0.4, ly_ / 2, 0.0)

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

  ω = form_taylor_vortices(s, 1.0, 0.3, [left_cnt, right_cnt])

  ψ = laplacian(Val(0), s) \ ω;
  ψ .= ψ .- minimum(ψ);

  u_star_0 = exterior_derivative(Val(0), s) * ψ;

  rho_star_0 = inv_hodge_star(Val(2), s) * ones(nquads(s));

  return ComponentArray(
    U_star = u_star_0,
    rho_star = rho_star_0,
  )
end

const u0 = form_taylor_vortices_initial_conditions(s)

###########################
### Boundary Conditions ###
###########################

@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT
  return v
end

@inline function enforce_bc_V!(V::AbstractVector{FT}) where FT
  return V
end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT
  return U
end