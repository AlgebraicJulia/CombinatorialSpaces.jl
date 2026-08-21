using SparseArrays
using LinearAlgebra

function exterior_derivative(::Val{0}, s::UniformCubicalComplex2D)

  tot = 2 * ne(s)
  I, J = zeros(Int32, tot), zeros(Int32, tot)
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
  I, J = zeros(Int32, tot), zeros(Int32, tot)
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

function no_flux_dual_derivative(::Val{0}, s::UniformCubicalComplex2D{FT}) where FT <: AbstractFloat
  dd0 = dual_derivative(Val(0), s)
  dd0[boundary_edges(s), :] .= FT(0.0) # Enforce no-flux boundary condition on density
  return dd0
end

d_beta(::Val{1}, s::UniformCubicalComplex2D{FT}) where FT <: AbstractFloat = FT(0.5) * abs.(dual_derivative(Val(1), s)) * spdiagm(dual_derivative(Val(0), s) * ones(nquads(s)))

hodge_star(::Val{0}, s::UniformCubicalComplex2D) = spdiagm(map(dq -> dual_quad_area(s, dq), vertices(s)))
function hodge_star(::Val{1}, s::UniformCubicalComplex2D)
  e_lens = map(e -> edge_len(s, e), edges(s))
  de_lens = map(de -> dual_edge_len(s, de), edges(s))
  return spdiagm(de_lens ./ e_lens)
end
hodge_star(::Val{2}, s::UniformCubicalComplex2D) = spdiagm(fill(inv(quad_area(s)), nquads(s)))

inv_hodge_star(::Val{1}, s::UniformCubicalComplex2D) = spdiagm(-inv.(diag(hodge_star(Val(1), s))))
inv_hodge_star(::Val{k}, s::UniformCubicalComplex2D) where k = spdiagm(inv.(diag(hodge_star(Val(k), s))))

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
  I, J = Int32[], Int32[]
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

function interior(::Val{0}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp = reshape(f, (nx(s), ny(s)))
  real_x_range = (halo_west(s) + 1):(nx(s) - halo_east(s))
  real_y_range = (halo_south(s) + 1):(ny(s) - halo_north(s))
  return reshape(tmp[real_x_range, real_y_range], nvr(s))
end

function interior(::Val{1}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp_x = reshape(f[1:nxedges(s)], (nxe(s), ny(s)))
  tmp_y = reshape(f[(nxedges(s) + 1):end], (nx(s), nye(s)))

  interior_x = reshape(tmp_x[(halo_west(s) + 1):(nxe(s) - halo_east(s)),
                              (halo_south(s) + 1):(ny(s) - halo_north(s))], nxe_r(s) * nyr(s))
  interior_y = reshape(tmp_y[(halo_west(s) + 1):(nx(s) - halo_east(s)),
                              (halo_south(s) + 1):(nye(s) - halo_north(s))], nxr(s) * nye_r(s))

  return vcat(interior_x, interior_y)
end

function interior(::Val{2}, f::AbstractVector, s::UniformCubicalComplex2D)
  tmp = reshape(f, (nxq(s), nyq(s)))
  real_x_range = (halo_west(s) + 1):(halo_west(s) + nxqr(s))
  real_y_range = (halo_south(s) + 1):(halo_south(s) + nyqr(s))
  return reshape(tmp[real_x_range, real_y_range], nquadsr(s))
end
