using KernelAbstractions

# Kernel for exterior derivative of 0-forms (primal 0-forms to primal 1-forms)
@kernel function kernel_exterior_derivative_zero!(res, s, f)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  if align == X_ALIGN
    @inbounds res[coord_to_edge(s, x, y, align)] = f[tgt_x(s, x, y)] - f[src_x(s, x, y)]
  elseif align == Y_ALIGN
    @inbounds res[coord_to_edge(s, x, y, align)] = f[tgt_y(s, x, y)] - f[src_y(s, x, y)]
  end
end

# Kernel for exterior derivative of 1-forms (primal 1-forms to primal 2-forms)
@kernel function kernel_exterior_derivative_one!(res, s, f, padding)
  idx = @index(Global)
  x, y = idx

  b_xe, t_xe, l_ye, r_ye = quad_edges(x, y)
  @inbounds res[idx] = f[b_xe] - f[t_xe] - f[l_ye] + f[r_ye]
end

# Main interface functions

function exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex2D, f)
  kernel = kernel_exterior_derivative_zero!(get_backend(res))

  kernel(res, s, f; ndrange = size(res))
end

function exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex2D, f)
  kernel = kernel_exterior_derivative_one!(get_backend(res))

  kernel(res, s, f; ndrange = size(res))
end

# Kernel for wedge 01, 11, and dual 01 products

@kernel function kernel_wedge_product_01(res, s, f, a)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  # Get the values of the 0-form at the vertices of the edge
  v1 = f[src(s, x, y, align)]
  v2 = f[tgt(s, x, y, align)]

  @inbounds res[idx] = (v1 + v2) * a[idx] / 2
end

@kernel function kernel_wedge_product_11(res, s, a, b)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  es = quad_edges(s, x, y)

  @inbounds res[idx] = 0.25 * (a[es[1]] + a[es[3]]) * (b[es[2]] + b[es[4]]) -
    0.25 * (a[es[2]] + a[es[4]]) * (b[es[1]] + b[es[3]])
end

# TODO: Remove control flow to make this more efficient on the GPU
@kernel function kernel_wedge_product_dual_01(res, s, f, a)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  dv1, dv2 = edge_quads(s, x, y, align)

  ae = a[idx]

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

  @inbounds res[idx] = tmp
end

function wedge_product(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_wedge_product_01(backend)

  kernel(res, s, f, a; ndrange = size(res))
  return res
end

wedge_product(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) = wedge_product(Val(0), Val(1), s, f, a)

function wedge_product(::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, f1, f2)
  backend = get_backend(f1)
  res = KernelAbstractions.zeros(backend, Float64, nquads(s))
  kernel = kernel_wedge_product_11(backend)

  kernel(res, s, f1, f2; ndrange = size(res))
  return res
end

function wedge_product_dd(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_wedge_product_dual_01(backend)

  kernel(res, s, f, a; ndrange = size(res))
  return res
end

wedge_product_dd(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) = wedge_product_dd(Val(0), Val(1), s, f, a)


# Convert a dual 1-form to a vector field on the dual points
@kernel function kernel_sharp_dd(X, Y, s, a)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)
  e1, e2, e3, e4 = quad_edges(s, x, y) # Order is (x-, y-, x+, y+)

  le1 = dual_edge_len(s, e1)
  le2 = dual_edge_len(s, e2)
  le3 = dual_edge_len(s, e3)
  le4 = dual_edge_len(s, e4)

  # Remember that for an X-aligned primal edge, we have a Y-aligned dual edge
  X[idx] = -(a[e2]/le2 + a[e4]/le4) / 2
  Y[idx] = (a[e1]/le1 + a[e3]/le3) / 2
end

function sharp_dd(s::UniformCubicalComplex2D, a)
  backend = get_backend(a)
  X = KernelAbstractions.zeros(backend, Float64, nquads(s))
  Y = KernelAbstractions.zeros(backend, Float64, nquads(s))
  kernel = kernel_sharp_dd(backend)

  kernel(X, Y, s, a; ndrange = size(X))
  return X, Y
end

# Flat, convert dual vector field to primal 1-form
@kernel function kernel_flat_dp(res, s, X, Y)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)
  if align == X_ALIGN
    if y == 1
      res[idx] = X[coord_to_quad(s,x,y)] * edge_len(s, X_ALIGN)
    elseif y == ny(s)
      res[idx] = X[coord_to_quad(s,x,y-1)] * edge_len(s, X_ALIGN)
    else
      res[idx] = 0.5 * (X[coord_to_quad(s,x,y)] + X[coord_to_quad(s,x,y-1)]) * edge_len(s, X_ALIGN)
    end
  else
    if x == 1
      res[idx] = Y[coord_to_quad(s,x,y)] * edge_len(s, Y_ALIGN)
    elseif x == nx(s)
      res[idx] = Y[coord_to_quad(s,x-1,y)] * edge_len(s, Y_ALIGN)
    else
      res[idx] = 0.5 * (Y[coord_to_quad(s,x,y)] + Y[coord_to_quad(s,x-1,y)]) * edge_len(s, Y_ALIGN)
    end
  end
end

function flat_dp(s::UniformCubicalComplex2D, X, Y)
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_flat_dp(backend)

  kernel(res, s, X, Y; ndrange = size(res))
  return res
end

# Flat, convert dual vector field to dual 1-form
@kernel function kernel_flat_dd(res, s, X, Y)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  if align == X_ALIGN
    if y == 1
      res[idx] = Y[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif y == ny(s)
      res[idx] = Y[coord_to_quad(s, x, y - 1)] * dual_edge_len(s, idx)
    else
      res[idx] = 0.5 * (Y[coord_to_quad(s, x, y)] + Y[coord_to_quad(s, x, y - 1)]) * dual_edge_len(s, idx)
    end
  else
    if x == 1
      res[idx] = -X[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif x == nx(s)
      res[idx] = -X[coord_to_quad(s, x - 1, y)] * dual_edge_len(s, idx)
    else
      res[idx] = -0.5 * (X[coord_to_quad(s, x, y)] + X[coord_to_quad(s, x - 1, y)]) * dual_edge_len(s, idx)
    end
  end
end

function flat_dd(s::UniformCubicalComplex2D, X, Y)
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_flat_dd(backend)

  kernel(res, s, X, Y; ndrange = size(res))
  return res
end

# TODO: Why is there scalar indexing in this function? Can we remove it to make it more efficient on the GPU?
function interpolate_dp(::Val{1}, s::UniformCubicalComplex2D, a)
  backend = get_backend(a)
  X, Y = sharp_dd(s, a)
  synchronize(backend)
  return flat_dp(s, X, Y)
end

# ═══════════════════════════════════════════════════════════════════════════
#  set_periodic! – KernelAbstractions versions
# ═══════════════════════════════════════════════════════════════════════════
#
# Each kernel iterates over the halo cells on one pair of opposing sides
# (east/west or north/south).  A single linear index maps to an (i, j)
# pair within the halo strip; each thread copies from the matching interior
# cell on the opposite side of the domain.

@enum GridSide EASTWEST NORTHSOUTH ALL

# ── Val{0}  (0-forms on vertices) ────────────────────────────────────────

@kernel function kernel_set_periodic_0_ew!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nx_ = nx(s); ny_ = ny(s)

  j = div(idx - 1, hx_ + 1) + 1
  i = idx - (j - 1) * (hx_ + 1)

  # Right halo ← left interior
  @inbounds f[coord_to_vert(s, nx_ - hx_ + i - 1, j)] = f[coord_to_vert(s, hx_ + i, j)]

  # Left halo ← right interior
  @inbounds f[coord_to_vert(s, i, j)] = f[coord_to_vert(s, nx_ - 2hx_ + i - 1, j)]
end

@kernel function kernel_set_periodic_0_ns!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nx_ = nx(s); ny_ = ny(s)
  j = div(idx - 1, nx_) + 1
  i = idx - (j - 1) * nx_

  # Top halo ← bottom interior
  # Correct, map first vertex to last vertex continue rightwards to fill halo
  @inbounds f[coord_to_vert(s, i, ny_ - hy_ + j - 1)] = f[coord_to_vert(s, i, hy_ + j)]

  # Bottom halo ← top interior
  @inbounds f[coord_to_vert(s, i, j)] = f[coord_to_vert(s, i, ny_ - 2hy_ + j - 1)]
end

function set_periodic!(f::AbstractVector, ::Val{0}, s::UniformCubicalComplex2D, side::GridSide)
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_0_ew!(backend)(f, s; ndrange = (hx(s) + 1) * ny(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_0_ns!(backend)(f, s; ndrange = nx(s) * (hy(s) + 1))
  end
  return f
end

# ── Val{1}  (1-forms on edges) ───────────────────────────────────────────
#
# Edges come in two families (X-aligned and Y-aligned) with different grid
# dimensions, so the EW and NS kernels each handle both families in turn.

# East/West halo for x-edges:  strip is hx × ny(s)
@kernel function kernel_set_periodic_1_ew_x!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nxe_ = nxe(s); nyv_ = ny(s)
  j = div(idx - 1, hx_) + 1
  i = idx - (j - 1) * hx_
  @inbounds f[coord_to_edge(s, nxe_ - hx_ + i, j, X_ALIGN)] = f[coord_to_edge(s, hx_ + i, j, X_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, X_ALIGN)] = f[coord_to_edge(s, nxe_ - 2hx_ + i, j, X_ALIGN)]
end

# East/West halo for y-edges:  strip is hx × nye(s)
@kernel function kernel_set_periodic_1_ew_y!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nye_ = nye(s); nxv_ = nx(s)
  j = div(idx - 1, hx_ + 1) + 1
  i = idx - (j - 1) * (hx_ + 1)

  @inbounds f[coord_to_edge(s, nxv_ - hx_ + i - 1, j, Y_ALIGN)] = f[coord_to_edge(s, hx_ + i, j, Y_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, Y_ALIGN)] = f[coord_to_edge(s, nxv_ - 2hx_ + i - 1, j, Y_ALIGN)]
end

# North/South halo for x-edges:  strip is nxe(s) × hy
@kernel function kernel_set_periodic_1_ns_x!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nxe_ = nxe(s); nyv_ = ny(s)
  j = div(idx - 1, nxe_) + 1
  i = idx - (j - 1) * nxe_
  @inbounds f[coord_to_edge(s, i, nyv_ - hy_ + j - 1, X_ALIGN)] = f[coord_to_edge(s, i, hy_ + j, X_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, X_ALIGN)] = f[coord_to_edge(s, i, nyv_ - 2hy_ + j - 1, X_ALIGN)]
end

# North/South halo for y-edges:  strip is nx(s) × hy
@kernel function kernel_set_periodic_1_ns_y!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nxv_ = nx(s); nye_ = nye(s)
  j = div(idx - 1, nxv_) + 1
  i = idx - (j - 1) * nxv_
  @inbounds f[coord_to_edge(s, i, j, Y_ALIGN)] = f[coord_to_edge(s, i, nye_ - 2hy_ + j, Y_ALIGN)]
  @inbounds f[coord_to_edge(s, i, nye_ - hy_ + j, Y_ALIGN)] = f[coord_to_edge(s, i, hy_ + j, Y_ALIGN)]
end

function set_periodic!(f::AbstractVector, ::Val{1}, s::UniformCubicalComplex2D, side::GridSide)
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_1_ew_x!(backend)(f, s; ndrange = hx(s) * ny(s))
    kernel_set_periodic_1_ew_y!(backend)(f, s; ndrange = (hx(s) + 1) * nye(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_1_ns_x!(backend)(f, s; ndrange = nxe(s) * (hy(s) + 1))
    kernel_set_periodic_1_ns_y!(backend)(f, s; ndrange = nx(s) * hy(s))
  end
  return f
end

# ── Val{2}  (2-forms on quads) ───────────────────────────────────────────

@kernel function kernel_set_periodic_2_ew!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nqx_ = nxquads(s); nqy_ = nyquads(s)
  j = div(idx - 1, hx_) + 1
  i = idx - (j - 1) * hx_
  @inbounds f[coord_to_quad(s, i, j)] = f[coord_to_quad(s, nqx_ - 2hx_ + i, j)]
  @inbounds f[coord_to_quad(s, nqx_ - hx_ + i, j)] = f[coord_to_quad(s, hx_ + i, j)]
end

@kernel function kernel_set_periodic_2_ns!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nqx_ = nxquads(s); nqy_ = nyquads(s)
  j = div(idx - 1, nqx_) + 1
  i = idx - (j - 1) * nqx_
  @inbounds f[coord_to_quad(s, i, j)] = f[coord_to_quad(s, i, nqy_ - 2hy_ + j)]
  @inbounds f[coord_to_quad(s, i, nqy_ - hy_ + j)] = f[coord_to_quad(s, i, hy_ + j)]
end

function set_periodic!(f::AbstractVector, ::Val{2}, s::UniformCubicalComplex2D, side::GridSide)
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_2_ew!(backend)(f, s; ndrange = hx(s) * nyquads(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_2_ns!(backend)(f, s; ndrange = nxquads(s) * hy(s))
  end
  return f
end
