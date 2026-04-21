using CairoMakie
using LinearAlgebra
using SparseArrays
using Test

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

##############################
##### GRID SETUP         #####
##############################

lx_ = ly_ = 2π
nx_ = ny_ = 501
s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_)

##############################
##### SMOOTHING          #####
##############################

# Smoothing for dual 0-forms (which live on quads/faces).
# Each quad is averaged with its face-adjacent neighbors (sharing an edge),
# weighted by the inverse distance between quad centers.
function smoothing_dual0(s::UniformCubicalComplex2D, c_smooth)
  n = nquads(s)
  c = c_smooth / 2
  inv_dx = 1 / dx(s)
  inv_dy = 1 / dy(s)
  nqx = nxquads(s)
  nqy = nyquads(s)

  # Pre-allocate COO arrays (at most 5 entries per quad: self + 4 neighbors)
  max_nnz = 5 * n
  I = Vector{Int}(undef, max_nnz)
  J = Vector{Int}(undef, max_nnz)
  V = Vector{Float64}(undef, max_nnz)
  idx = 0

  for q in quads(s)
    x, y = quad_to_coord(s, q)

    # Collect neighbor weights
    has_left  = x > 1
    has_right = x < nqx
    has_down  = y > 1
    has_up    = y < nqy

    tot_w = (has_left + has_right) * inv_dx + (has_down + has_up) * inv_dy

    if tot_w > 0
      scale = c / tot_w
      if has_left
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x - 1, y); V[idx] = scale * inv_dx
      end
      if has_right
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x + 1, y); V[idx] = scale * inv_dx
      end
      if has_down
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x, y - 1); V[idx] = scale * inv_dy
      end
      if has_up
        idx += 1; I[idx] = q; J[idx] = coord_to_quad(s, x, y + 1); V[idx] = scale * inv_dy
      end
    end

    # Diagonal
    idx += 1; I[idx] = q; J[idx] = q; V[idx] = 1 - c
  end

  return sparse(view(I, 1:idx), view(J, 1:idx), view(V, 1:idx), n, n)
end

##############################
##### INITIAL CONDITION  #####
##############################

# Sinusoidal initial condition on dual 0-forms (one per quad)
dps = dual_points(s)
freq = 25
f0 = [sin(freq * 2π * p[1] / lx_) * sin(freq * 2π * p[2] / ly_) for p in dps]

hdg_2 = hodge_star(Val(2), s)

##############################
##### APPLY SMOOTHING    #####
##############################

c_smooth = 1.0
forward_smooth = smoothing_dual0(s, c_smooth)
backward_smooth = smoothing_dual0(s, -c_smooth)
smoothing_mat = backward_smooth * forward_smooth

# Check that each row sums to 1 (preserves constant functions)
@test all(abs.(sum(forward_smooth, dims=2) .- 1) .< 1e-10)
@test all(abs.(sum(backward_smooth, dims=2) .- 1) .< 1e-10)
@test all(abs.(sum(smoothing_mat, dims=2) .- 1) .< 1e-10) # Check that each row sums to 1 (preserves constant functions)

mkpath("imgs/Smoothing")
save("imgs/Smoothing/Initial.png", plot_twoform(s, f0))

n_iterations = 50
f_smoothed = copy(f0)
for i in 1:n_iterations

  f_smoothed .= smoothing_mat * f_smoothed

  if i % 10 == 0
    save("imgs/Smoothing/Smoothed_$(i)_iters.png", plot_twoform(s, f_smoothed))
    save("imgs/Smoothing/Smoothed_$(i)_diff.png", plot_twoform(s, f_smoothed - f0))

    println("Max initial:  ", maximum(abs, f0))
    println("Max smoothed: ", maximum(abs, f_smoothed))
    println("Max diff:    ", maximum(abs, (f_smoothed - f0)))
  end
end

println("Saved plots to imgs/Smoothing/")
