##########################
### Initial Conditions ###
##########################

testcase = "RE10000" # "RE10", "RE100", "RE1000", "RE10000"

println("Running Cavity Flow Simulation...")
println("Test case: $testcase")

if testcase == "RE10"
  const dt = 1e-5
  const te = 1.0
  const Re = 10.0
  const _nx = _ny = 129
elseif testcase == "RE100"
  const dt = 1e-5
  const te = 5.0
  const Re = 100.0
  const _nx = _ny = 129
elseif testcase == "RE1000"
  const dt = 1e-5
  const te = 25.0
  const Re = 1_000.0
  const _nx = _ny = 129
elseif testcase == "RE10000"
  const dt = 9e-6
  const te = 40.0
  const Re = 10_000.0
  const _nx = _ny = 257
end

const p = Dict(:mu => 1 / Re) # μ

const _lx = _ly = 1.0;
const s = UniformCubicalComplex2D(_nx, _ny, _lx, _ly)

const rho_star_0 = inv_hodge_star(Val(2), s) * ones(Float64, nquads(s))
const U_star_0 = zeros(Float64, ne(s))

###########################
### Boundary Conditions ###
###########################

const boundary_mask_d = to_device(zeros(Float64, ne(s))); boundary_mask_d[boundary_edges(s)] .= 1.0
const top_mask_d = to_device(zeros(Float64, ne(s))); top_mask_d[top_edges(s)] .= 1.0

@inline function enforce_bc_v!(v::AbstractVector{FT}) where FT
  v .= v .* (1.0 .- boundary_mask_d) # Enforce no-slip boundary condition on velocity
  v .= v .* (1.0 .- top_mask_d) .+ edge_len(s, X_ALIGN) .* top_mask_d # Lid velocity
  return v
end

@inline function enforce_bc_U!(U::AbstractVector{FT}) where FT
  U .= U .* (1.0 .- boundary_mask_d) # Enforce no-flux boundary condition on momentum
  return U
end

#########################
### Saving Parameters ###
#########################

print_re = @sprintf("%.0f", Re)
print_te = @sprintf("%.2f", te)
const simspec = "Re=$(print_re)_te=$(print_te)"
const save_path = "/blue/fairbanksj/grauta/simulations/LMNS_CavityFlow/$(simspec)"
mkpath(save_path)

# Run for set time, checkpoint every 5s, save every 0.01s to get 500 frames for the mp4
const saveat = floor(Int64, 0.01 / dt)
const checkpoint_at = floor(Int64, 5.0 / dt)
