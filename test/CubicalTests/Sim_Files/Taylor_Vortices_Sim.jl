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

const lx_ = ly_ = 2π;
const nx_ = ny_ = 81
const halo_x = halo_y = 10

const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_, halo_x = halo_x, halo_y = halo_y)

const u0 = form_taylor_vortices_initial_conditions(s)

const Re = 10_000
const p = (mu = 1 / Re,) # μ

const te = 1.0;
const dt = 1e-4;

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

const full_periodic = true
const periodic_left_right = true
const periodic_top_bottom = true

#########################
### Saving Parameters ###
#########################

print_re = @sprintf("%.0f", Re)
print_te = @sprintf("%.2f", te)
const simspec = "Re=$(print_re)_te=$(print_te)"
const save_path = "/blue/fairbanksj/grauta/simulations/LMNS_TaylorVortices/$(simspec)"
mkpath(save_path)

# Save every 0.01s and checkpoint every 50 save events.
const saveat = floor(Int64, 0.01 / dt)
const checkpoint_every_saveat = 50
const checkpoint_at = saveat * checkpoint_every_saveat