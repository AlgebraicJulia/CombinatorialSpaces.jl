##########################
### Initial Conditions ###
##########################

println("Running Kelvin-Helmholtz Simulation...")

const lx_ = ly_ = 1.0;
const nx_ = ny_ = 256;
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_, halo_x = 10, halo_y = 10)

# Initial conditions for Kelvin-Helmholtz instability

# Collect points between 0 and 0.25 in y-direction for the shear layer

# Conditions follow from San and Kara (2015)
const ps = points(s)
const dps = dual_points(s)
const kh_inv_hdg_2 = inv_hodge_star(Val(2), s)

# const rho_star_0 = kh_inv_hdg_2 * map(dps) do (x, y)
#   if 0.25 <= y <= 0.75
#     return 2.0
#   else
#     return 1.0
#   end
# end

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

const Theta_star_0 = inv_hodge_star(Val(2), s) * (300.0 * ones(nquads(s)))

const Re = 10_000
const Pr = 0.71
const p = (mu = 1 / Re, kappa = 1 / (Re * Pr)) # μ and κ

const te = 5.0;
const dt = 5e-6; # Choose small time step for stability at high Re

const use_gravity = false
const full_periodic = true
const periodic_left_right = true
const periodic_top_bottom = true

const T_smooth_constant = 0.2
const r_smooth_constant = 0.05

###########################
### Boundary Conditions ###
###########################

@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT
  return v
end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT
  return U
end

#########################
### Saving Parameters ###
#########################

print_re = @sprintf("%.0f", Re)
print_te = @sprintf("%.2f", te)
simspec = "Re=$(print_re)_te=$(print_te)"
save_path = "/blue/fairbanksj/grauta/simulations/LMNS_KelvinHelmholtz/$(simspec)"
mkpath(save_path)

const saveat = 1000
const checkpoint_at = 20_000

# Plot the density field at the start of the simulation
fig = plot_twoform(s, hodge_star(Val(2), s) * rho_star_0)
save(joinpath(save_path, "initial_density.png"), fig)

fig = plot_xy_oneform(s, inv_hodge_star(Val(1), s) * U_star_0)
save(joinpath(save_path, "initial_velocity.png"), fig)

fig = plot_twoform(s, hodge_star(Val(2), s) * Theta_star_0)
save(joinpath(save_path, "initial_potential_temperature.png"), fig)
