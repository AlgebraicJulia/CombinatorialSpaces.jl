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

function wedge_product(::Val{i}, ::Val{j}, s::UniformCubicalComplex2D, f, g) where {i, j}
  k = i + j
  if k == 0
    res = zeros(nv(s))
  elseif k == 1
    res = zeros(ne(s))
  elseif k == 2
    res = zeros(nquads(s))
  else
    error("Invalid wedge product: forms of degree $i and $j cannot be wedged together in 2D")
  end

  return wedge_product!(res, Val(i), Val(j), s, f, g)
end

# TODO: Convert these to kernel functions for high-performance
function wedge_product!(res, ::Val{0}, ::Val{0}, s::UniformCubicalComplex2D, f, g)
  for v in vertices(s)
    res[v] = f[v] * g[v]
  end
  return res
end

function wedge_product!(res, ::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  for e in edges(s)
    x, y, align = edge_to_coord(s, e)
    v1 = src(s, x, y, align)
    v0 = tgt(s, x, y, align)
    res[e] = 0.5 * (f[v0] + f[v1]) * a[e]
  end
  return res
end

wedge_product!(res, ::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) = wedge_product!(res, Val(0), Val(1), s, f, a)

function wedge_product!(res, ::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a, b)
  for q in quads(s)
    x, y = quad_to_coord(s, q)
    es = quad_edges(s, x, y) # Order is (x-, y-, x+, y+)
    res[q] = 0.25 * (a[es[1]] + a[es[3]]) * (b[es[2]] + b[es[4]]) - 0.25 * (a[es[2]] + a[es[4]]) * (b[es[1]] + b[es[3]])
  end
  return res
end

function wedge_product_dd(::Val{i}, ::Val{j}, s::UniformCubicalComplex2D, f, g) where {i, j}
  k = i + j
  if k == 0
    res = zeros(nv(s))
  elseif k == 1
    res = zeros(ne(s))
  elseif k == 2
    res = zeros(nquads(s))
  else
    error("Invalid wedge product: forms of degree $i and $j cannot be wedged together in 2D")
  end

  return wedge_product_dd!(res, Val(i), Val(j), s, f, g)
end

# TODO: Make sure this is handling boundary cases correctly
function wedge_product_dd!(res, ::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  for e in edges(s)
    x, y, align = edge_to_coord(s, e)
    dv1, dv2 = edge_quads(s, x, y, align)

    ae = a[e]

    if is_left_edge(s, x, y, align)
      tmp = f[dv2] * ae
    elseif is_right_edge(s, x, y, align)
      tmp = f[dv1] * ae
    elseif is_top_edge(s, x, y, align)
      tmp = f[dv1] * ae
    elseif is_bottom_edge(s, x, y, align)
      tmp = f[dv2] * ae
    else
      tmp = 0.5 * (f[dv1] + f[dv2]) * ae
    end

    res[e] = tmp
  end
  return res
end

wedge_product_dd!(res, ::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) = wedge_product_dd!(res, Val(0), Val(1), s, f, a)

function sharp_dd!(X, Y, s::UniformCubicalComplex2D, a)
  for dv in quads(s)
    x, y = quad_to_coord(s, dv)
    e1, e2, e3, e4 = quad_edges(s, x, y) # Order is (x-, y-, x+, y+)

    le1 = dual_edge_len(s, e1)
    le2 = dual_edge_len(s, e2)
    le3 = dual_edge_len(s, e3)
    le4 = dual_edge_len(s, e4)

    # Remember that for an X-aligned primal edge, we have a Y-aligned dual edge
    X[dv] = -(a[e2]/le2 + a[e4]/le4) / 2
    Y[dv] = (a[e1]/le1 + a[e3]/le3) / 2
  end
  return X, Y
end

# Given a vector field on the dual points, return a one-form on the primal edges
# Edge cases arise when primal edge is on the boundary of the grid, in which case we just take the value of the vector field at the single adjacent dual point
function flat_dp!(res, s::UniformCubicalComplex2D, X, Y)
  for e in edges(s)
    x, y, align = edge_to_coord(s, e)
    if align == X_ALIGN
      if y == 1
        res[e] = X[coord_to_quad(s,x,y)] * edge_len(s, X_ALIGN)
      elseif y == ny(s)
        res[e] = X[coord_to_quad(s,x,y-1)] * edge_len(s, X_ALIGN)
      else
        res[e] = 0.5 * (X[coord_to_quad(s,x,y)] + X[coord_to_quad(s,x,y-1)]) * edge_len(s, X_ALIGN)
      end
    else
      if x == 1
        res[e] = Y[coord_to_quad(s,x,y)] * edge_len(s, Y_ALIGN)
      elseif x == nx(s)
        res[e] = Y[coord_to_quad(s,x-1,y)] * edge_len(s, Y_ALIGN)
      else
        res[e] = 0.5 * (Y[coord_to_quad(s,x,y)] + Y[coord_to_quad(s,x-1,y)]) * edge_len(s, Y_ALIGN)
      end
    end
  end
  return res
end

# Create a matrix that maps values on dual points to primal points by taking the average of the adjacent dual points for each primal point
# TODO: Write a test for this function to make sure it's doing what we expect, especially at the boundaries
function interpolate_dp(::Val{0}, s::UniformCubicalComplex2D)
  I, J = Int64[], Int64[]
  V = Float64[]

  for v in vertices(s)
    x, y = vert_to_coord(s, v)
    q1, q2, q3, q4 = vert_quads(s, x, y)

    ### TODO: Fill me in!
  end

  return sparse(I, J, V)
end

function interpolate_dp(::Val{1}, s::UniformCubicalComplex2D, u::AbstractVector)
  X = zeros(nquads(s)); Y = zeros(nquads(s))
  v = zeros(ne(s))

  sharp_dd!(X, Y, s, u)
  flat_dp!(v, s, X, Y)
  return v
end


@enum GridSide EASTWEST NORTHSOUTH ALL

# TODO: Improve this function to handle non-constant halo values
# This function sets the halo values of a field defined on a grid with halo to the given value, by realigning the field
function set_halo!(f::AbstractVector, ::Val{0}, s::UniformCubicalComplex2D, value::AbstractFloat, side::GridSide)
  nx_, ny_ = nx(s), ny(s)
  hx_, hy_ = hx(s), hy(s)

  if side == EASTWEST || side == ALL
    for i in 1:hx_, j in 1:ny_
      f[coord_to_vert(s,i,j)] = value
    end
    for i in nx_-hx_+1:nx_, j in 1:ny_
      f[coord_to_vert(s,i,j)] = value
    end
  end
  if side == NORTHSOUTH || side == ALL
    for i in 1:nx_, j in 1:hy_
      f[coord_to_vert(s,i,j)] = value
    end
    for i in 1:nx_, j in ny_-hy_+1:ny_
      f[coord_to_vert(s,i,j)] = value
    end
  end
  return f
end

# This function sets the halo values depending on the interior values, by realigning the field and then setting the halo values to match the corresponding interior values on the opposite side of the grid
function set_periodic!(f::AbstractVector, ::Val{0}, s::UniformCubicalComplex2D, side::GridSide)
  nx_, ny_ = nx(s), ny(s)
  hx_, hy_ = hx(s), hy(s)

  if side == EASTWEST || side == ALL
    for i in 1:hx_, j in 1:ny_ # Set the left halo to match the right interior
      f[coord_to_vert(s,i,j)] = f[coord_to_vert(s,nx_-2hx_+i,j)]
    end
    for i in nx_-hx_+1:nx_, j in 1:ny_ # Set the right halo to match the left interior
      f[coord_to_vert(s,i,j)] = f[coord_to_vert(s,2hx_+i-nx_,j)]
    end
  end
  if side == NORTHSOUTH || side == ALL
    for i in 1:nx_, j in 1:hy_ # Set the bottom halo to match the top interior
      f[coord_to_vert(s,i,j)] = f[coord_to_vert(s,i,ny_-2hy_+j)]
    end
    for i in 1:nx_, j in ny_-hy_+1:ny_ # Set the top halo to match the bottom interior
      f[coord_to_vert(s,i,j)] = f[coord_to_vert(s,i,2hy_+j-ny_)]
    end
  end
  return f
end

function set_periodic!(f::AbstractVector, ::Val{1}, s::UniformCubicalComplex2D, side::GridSide)
  nx_, ny_ = nxe(s), nye(s)
  nxv_, nyv_ = nx(s), ny(s)
  hx_, hy_ = hx(s), hy(s)

  if side == EASTWEST || side == ALL
    for i in 1:hx_, j in 1:nyv_ # Set the left x-edges to match the right interior
      f[coord_to_edge(s,i,j,X_ALIGN)] = f[coord_to_edge(s,nx_-2hx_+i,j,X_ALIGN)]
    end
    for i in nx_-hx_+1:nx_, j in 1:nyv_ # Set the right x-edges to match the left interior
      f[coord_to_edge(s,i,j,X_ALIGN)] = f[coord_to_edge(s,2hx_+i-nx_,j,X_ALIGN)]
    end
    for i in 1:hx_, j in 1:ny_ # Set the left y-edges to match the right interior
      f[coord_to_edge(s,i,j,Y_ALIGN)] = f[coord_to_edge(s,nxv_-2hx_+i,j,Y_ALIGN)]
    end
    for i in nxv_-hx_+1:nxv_, j in 1:ny_ # Set the right y-edges to match the left interior
      f[coord_to_edge(s,i,j,Y_ALIGN)] = f[coord_to_edge(s,2hx_+i-nxv_,j,Y_ALIGN)]
    end
  end
  if side == NORTHSOUTH || side == ALL
    for i in 1:nx_, j in 1:hy_ # Set the bottom x-edges to match the top interior
      f[coord_to_edge(s,i,j,X_ALIGN)] = f[coord_to_edge(s,i,nyv_-2hy_+j,X_ALIGN)]
    end
    for i in 1:nx_, j in nyv_-hy_+1:nyv_ # Set the top x-edges to match the bottom interior
      f[coord_to_edge(s,i,j,X_ALIGN)] = f[coord_to_edge(s,i,2hy_+j-nyv_,X_ALIGN)]
    end
    for i in 1:nxv_, j in 1:hy_ # Set bottom y-edges to match the top interior
      f[coord_to_edge(s,i,j,Y_ALIGN)] = f[coord_to_edge(s,i,ny_-2hy_+j,Y_ALIGN)]
    end
    for i in 1:nxv_, j in ny_-hy_+1:ny_ # Set top y-edges to match the bottom interior
      f[coord_to_edge(s,i,j,Y_ALIGN)] = f[coord_to_edge(s,i,2hy_+j-ny_,Y_ALIGN)]
    end
  end
  return f
end

function set_periodic!(f::AbstractVector, ::Val{2}, s::UniformCubicalComplex2D, side::GridSide)
  nx_, ny_ = nxquads(s), nyquads(s)
  hx_, hy_ = hx(s), hy(s)

  if side == EASTWEST || side == ALL
    for i in 1:hx_, j in 1:ny_ # Set the left halo to match the right interior
      f[coord_to_quad(s,i,j)] = f[coord_to_quad(s,nx_-2hx_+i,j)]
    end
    for i in nx_-hx_+1:nx_, j in 1:ny_ # Set the right halo to match the left interior
      f[coord_to_quad(s,i,j)] = f[coord_to_quad(s,2hx_+i-nx_,j)]
    end
  end
  if side == NORTHSOUTH || side == ALL
    for i in 1:nx_, j in 1:hy_ # Set the bottom halo to match the top interior
      f[coord_to_quad(s,i,j)] = f[coord_to_quad(s,i,ny_-2hy_+j)]
    end
    for i in 1:nx_, j in ny_-hy_+1:ny_ # Set the top halo to match the bottom interior
      f[coord_to_quad(s,i,j)] = f[coord_to_quad(s,i,2hy_+j-ny_)]
    end
  end
  return f
end

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
