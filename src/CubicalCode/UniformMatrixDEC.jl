using SparseArrays
using LinearAlgebra

function exterior_derivative(::Val{0}, s::UniformCubicalComplex2D)

  tot = 2 * ne(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  for e in edges(s)
    idx = 2 * e - 1
    x, y, align = edge_to_coord(s, e)
    v0, v1 = tgt(s, x, y, align), src(s, x, y, align)

    I[idx] = e
    I[idx + 1] = e

    J[idx] = v0
    J[idx + 1] = v1

    V[idx] = 1
    V[idx + 1] = -1
  end

  return sparse(I, J, V)
end

function exterior_derivative(::Val{1}, s::UniformCubicalComplex2D)

  tot = 4 * nquads(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  orients = (1,1,-1,-1)
  for q in quads(s)
    idx = 4 * q - 3
    x, y = quad_to_coord(s, q)
    for (i, e) in enumerate(quad_edges(s, x, y))
      j = idx + i - 1
      I[j] = q
      J[j] = e
      V[j] = orients[i]
    end
  end

  return sparse(I, J, V)
end

dual_derivative(::Val{0}, s::UniformCubicalComplex2D) = transpose(exterior_derivative(Val(1), s))
dual_derivative(::Val{1}, s::UniformCubicalComplex2D) = -transpose(exterior_derivative(Val(0), s))

function no_flux_dual_derivative(::Val{0}, s::UniformCubicalComplex2D)
  dd0 = dual_derivative(Val(0), s)
  dd0[boundary_edges(s), :] .= 0.0 # Enforce no-flux boundary condition on density
  return dd0
end

dual_derivative_beta(::Val{1}, s::UniformCubicalComplex2D) = 0.5 * abs.(dual_derivative(Val(1), s)) * spdiagm(dual_derivative(Val(0), s) * ones(nquads(s)))

hodge_star(::Val{0}, s::UniformCubicalComplex2D) = spdiagm(map(dq -> dual_quad_area(s, dq), vertices(s)))
function hodge_star(::Val{1}, s::UniformCubicalComplex2D)
  e_lens = map(e -> edge_len(s, e), edges(s))
  de_lens = map(de -> dual_edge_len(s, de), edges(s))
  return spdiagm(de_lens ./ e_lens)
end
hodge_star(::Val{2}, s::UniformCubicalComplex2D) = spdiagm(fill(1 / quad_area(s), nquads(s)))

inv_hodge_star(::Val{1}, s::UniformCubicalComplex2D) = spdiagm(-1 ./ diag(hodge_star(Val(1), s)))
inv_hodge_star(::Val{k}, s::UniformCubicalComplex2D) where k = spdiagm(1 ./ diag(hodge_star(Val(k), s)))

codifferential(::Val{1}, s::UniformCubicalComplex2D) = inv_hodge_star(Val(0), s) * dual_derivative(Val(1), s) * hodge_star(Val(1), s)
codifferential(::Val{2}, s::UniformCubicalComplex2D) = inv_hodge_star(Val(1), s) * dual_derivative(Val(0), s) * hodge_star(Val(2), s)

dual_codifferential(::Val{1}, s::UniformCubicalComplex2D) = hodge_star(Val(2), s) * exterior_derivative(Val(1), s) * inv_hodge_star(Val(1), s)
dual_codifferential(::Val{2}, s::UniformCubicalComplex2D) = hodge_star(Val(1), s) * exterior_derivative(Val(0), s) * inv_hodge_star(Val(0), s)

laplacian(::Val{0}, s::UniformCubicalComplex2D) = codifferential(Val(1), s) * exterior_derivative(Val(0), s)
laplacian(::Val{1}, s::UniformCubicalComplex2D) = exterior_derivative(Val(0), s) * codifferential(Val(1), s) + codifferential(Val(2), s) * exterior_derivative(Val(1), s)
laplacian(::Val{2}, s::UniformCubicalComplex2D) = exterior_derivative(Val(1), s) * codifferential(Val(2), s)

dual_laplacian(::Val{0}, s::UniformCubicalComplex2D) = dual_codifferential(Val(1), s) * dual_derivative(Val(0), s)
dual_laplacian(::Val{1}, s::UniformCubicalComplex2D) = dual_codifferential(Val(2), s) * dual_derivative(Val(1), s) + dual_derivative(Val(0), s) * dual_codifferential(Val(1), s)
dual_laplacian(::Val{2}, s::UniformCubicalComplex2D) = dual_derivative(Val(1), s) * dual_codifferential(Val(2), s)

# Create a matrix that maps values on dual points to primal points by taking the average of the adjacent dual points for each primal point
# TODO: Write a test for this function to make sure it's doing what we expect, especially at the boundaries
function interpolate_dp(::Val{0}, s::UniformCubicalComplex2D)
  I, J = Int64[], Int64[]
  V = Float64[]

  for v in vertices(s)
    x, y = vert_to_coord(s, v)

    adjacent_quads = vert_quads(s, x, y)
    valid_quads = filter(q -> 1 <= q <= nquads(s), adjacent_quads)
    n = length(valid_quads)
    for q in valid_quads
      push!(I, v)
      push!(J, q)
      push!(V, 1 / n)
    end
  end

  return sparse(I, J, V)
end

# @enum GridSide EASTWEST NORTHSOUTH ALL

# # TODO: Improve this function to handle non-constant halo values
# # This function sets the halo values of a field defined on a grid with halo to the given value, by realigning the field
# function set_halo!(f::AbstractVector, ::Val{0}, s::UniformCubicalComplex2D, value::AbstractFloat, side::GridSide)
#   nx_, ny_ = nx(s), ny(s)
#   hx_, hy_ = hx(s), hy(s)

#   if side == EASTWEST || side == ALL
#     for i in 1:hx_, j in 1:ny_
#       f[coord_to_vert(s,i,j)] = value
#     end
#     for i in nx_-hx_+1:nx_, j in 1:ny_
#       f[coord_to_vert(s,i,j)] = value
#     end
#   end
#   if side == NORTHSOUTH || side == ALL
#     for i in 1:nx_, j in 1:hy_
#       f[coord_to_vert(s,i,j)] = value
#     end
#     for i in 1:nx_, j in ny_-hy_+1:ny_
#       f[coord_to_vert(s,i,j)] = value
#     end
#   end
#   return f
# end

# This functions gets the interior values of a field defined on a grid with halo, by realigning the field and then taking the appropriate view
function interior(::Val{0}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp = reshape(f, (nx(s), ny(s)))
  real_x_range = (hx(s) + 1):(nx(s) - hx(s))
  real_y_range = (hy(s) + 1):(ny(s) - hy(s))
  return reshape(tmp[real_x_range, real_y_range], nvr(s))
end

function interior(::Val{1}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp_x = reshape(f[1:nxedges(s)], (nxe(s), ny(s)))
  tmp_y = reshape(f[nxedges(s)+1:end], (nx(s), nye(s)))

  interior_x = reshape(tmp_x[(hx(s) + 1):(nxe(s) - hx(s)), (hy(s) + 1):(ny(s) - hy(s))], nxe_r(s) * nyr(s))
  interior_y = reshape(tmp_y[(hx(s) + 1):(nx(s) - hx(s)), (hy(s) + 1):(nye(s) - hy(s))], nxr(s) * nye_r(s))

  return vcat(interior_x, interior_y)
end

function interior(::Val{2}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp = reshape(f, (nxquads(s), nyquads(s)))
  real_x_range = (hx(s) + 1):(nxquads(s) - hx(s))
  real_y_range = (hy(s) + 1):(nyquads(s) - hy(s))
  return reshape(tmp[real_x_range, real_y_range], nquadsr(s))
end

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
