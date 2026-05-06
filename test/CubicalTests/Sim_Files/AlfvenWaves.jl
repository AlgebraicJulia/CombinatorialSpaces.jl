##########################
### Simulation Case    ###
##########################

println("Running Alfven Wave Simulation...")

const lx_ = ly_ = 1.0
const nx_ = ny_ = 151
const s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_; halo_x = 10, halo_y = 10)

const simname = "Alfven_Wave"
const save_path = joinpath(@__DIR__, "..", "imgs", "MHD")
mkpath(save_path)
for f in readdir(save_path)
  rm(joinpath(save_path, f))
end

#####################
### MHD Constants ###
#####################

const μ₀ = 4π * 1e-7  # Vacuum permeability
const V_s = 1.0       # Sound speed
const rho_base = 1.0
const rho_perturbation = 0.1
const B_0 = 1.0    # Background magnetic field (x-direction)
const δB  = 0.01   # Sinusoidal perturbation amplitude (y-component, varying in x)

##########################
### Initial Conditions ###
##########################

function initialize_gaussian_density(s::UniformCubicalComplex2D, lx::Float64, ly::Float64)
  ρ₀ = rho_base
  δρ = rho_perturbation
  σ  = 0.1
  x_c = lx / 2
  y_c = ly / 2

  dist = MvNormal([x_c, y_c], σ^2 * I)
  _inv_hdg_2 = inv_hodge_star(Val(2), s)

  _dps = dual_points(s)

  rho_0 = zeros(nquads(s))
  for q in 1:nquads(s)
    dpx, dpy = _dps[q]
    
    rho_0[q] = ρ₀ + δρ * pdf(dist, [dpx, dpy]) / pdf(dist, [x_c, y_c])
  end
  rho_star_0 = _inv_hdg_2 * rho_0

  U_star_0 = zeros(ne(s))

  B_star_0 = zeros(ne(s))
  # # Background B in x-direction (x-aligned edges)
  # for ex in 1:nxedges(s)
  #   B_star_0[ex] = B_0 * edge_len(s, X_ALIGN)
  # end
  # # Sinusoidal perturbation in y-component of B, varying in x
  # for ey in nxedges(s)+1:ne(s)
  #   v1 = src(s, ey); v2 = tgt(s, ey)
  #   x_mid = 0.5 * (_ps[v1][1] + _ps[v2][1])
  #   B_star_0[ey] = δB * sin(2π * x_mid) * edge_len(s, Y_ALIGN)
  # end

  return U_star_0, rho_star_0, B_star_0
end

const U_star_0, rho_star_0, B_star_0 = initialize_gaussian_density(s, lx_, ly_)

#########################
### Physical Parameters ###
#########################

const μ = 0.0
const η = 0.0
const p = (
  μ = μ,
  η = η,
  V_s = V_s,
  μ₀ = μ₀,
)

#########################
### Saving Parameters ###
#########################

const te = 2.5
const dt = 1e-3 # 5e-5
const saveat = 50
const checkpoint_at = 2500

const full_periodic = true
const periodic_left_right = true
const periodic_top_bottom = true

#########################
### Smoothing Parameters ###
#########################

const c_smooth = 0.2

###########################
### Boundary Conditions ###
###########################

# Fully periodic — no flux conditions needed at boundaries.
@inline enforce_bc_v!(v::AbstractVector{FT}) where FT = v
@inline enforce_bc_V!(V::AbstractVector{FT}) where FT = V
