##########################
### Initial Conditions ###
##########################

# Collect points between 0 and 0.25 in y-direction for the shear layer

# Conditions follow from San and Kara (2015)
const ps = points(s)
const dps = dual_points(s)
const kh_inv_hdg_2 = inv_hodge_star(Val(2), s)

density(y) = 1.0 + 0.5 * tanh(alpha * (y - 0.25)) - 0.5 * tanh(alpha * (y - 0.75))

const alpha = 50 # 20
const rho_star_0 = kh_inv_hdg_2 * map(dps) do (x, y)
  density(y)
end

const U_star_0 = zeros(ne(s));

# The momentum flow is computed on the primal
# So for the x-momentum (y-aligned edges), we set a shear flow in the x-direction
const U_x = map(1:nxedges(s)) do ey
  v1 = src(s, ey); v2 = tgt(s, ey);
  x = 0.5 * (ps[v1][1] + ps[v2][1])
  return 0.01 * sin(2π * x) * edge_len(s, X_ALIGN) * density(ps[v1][2]) # Multiplied by density to get momentum
end;

const U_y = map(nxedges(s)+1:ne(s)) do ex
  v1 = src(s, ex); v2 = tgt(s, ex);
  y = 0.5 * (ps[v1][2] + ps[v2][2])
  if 0.25 <= y <= 0.75
    flow = 0.5 * edge_len(s, Y_ALIGN)
  else
    flow = -0.5 * edge_len(s, Y_ALIGN)
  end

  # Multiplied by density to get momentum
  return flow * density(y)
end;

U_star_0[1:nxedges(s)] .= U_x;
U_star_0[nxedges(s)+1:ne(s)] .= U_y;

const Theta_star_0 = kh_inv_hdg_2 * (300.0 * ones(nquads(s)))

u0 = ComponentVector(U_star = U_star_0, rho_star = rho_star_0, Theta_star = Theta_star_0)

###########################
### Boundary Conditions ###
###########################

@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT
  return v
end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT
  return U
end

@inline function enforce_bc_V!(V::AbstractVector{FT}) where FT
  return V
end

###################
### Diagnostics ###
###################

# Plot the density field at the start of the simulation
fig = plot_twoform(s, hodge_star(Val(2), s) * rho_star_0)
save(joinpath(savepath, "initial_density.png"), fig)

fig = plot_xy_oneform(s, inv_hodge_star(Val(1), s) * U_star_0)
save(joinpath(savepath, "initial_velocity.png"), fig)

fig = plot_twoform(s, hodge_star(Val(2), s) * Theta_star_0)
save(joinpath(savepath, "initial_potential_temperature.png"), fig)
