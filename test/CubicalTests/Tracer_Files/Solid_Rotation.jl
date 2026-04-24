#################################
### Prescribed Velocity Field ###
#################################

# Solid-body rotation: vx = -Ω*(y - cy), vy = Ω*(x - cx)
# Stream function: ψ = (Ω/2) * ((x - cx)² + (y - cy)²)
# Velocity as dual 1-form: U = ★(dψ) = hdg_1 * (d0 * ψ)

const Ω  = 1.0                 # angular velocity; period T = 2π/Ω
const cx = lx_ / 2
const cy = ly_ / 2

const pts    = points(s)
const ψ_0   = [(Ω / 2) * ((p[1] - cx)^2 + (p[2] - cy)^2) for p in pts]

const u_star = d0 * ψ_0          # primal 1-form (stream function gradient)

##########################
### Initial Conditions ###
##########################

const dps         = dual_points(s)
const tracer_dist = MvNormal([lx_ / 4, ly_ / 2], [0.1, 0.1])
const phi_0       = [pdf(tracer_dist, [p[1], p[2]]) for p in dps]
const phi_star_0  = inv_hdg_2 * phi_0   # star representation of tracer

const kappa   = 1e-3                       # tracer diffusivity
const te      = 2π / Ω                     # one full rotation period
const dt      = 1e-3
const saveat  = max(1, floor(Int64, 0.05 / dt))
