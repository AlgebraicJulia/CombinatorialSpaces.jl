using KernelAbstractions
using Adapt

# Exterior derivatives

@kernel function kernel_exterior_derivative_zero!(res, s, @Const(f))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  @inbounds res[idx] = f[tgt(s, x, y, align)] - f[src(s, x, y, align)]
end

# matrix orientation (+1, +1, -1, -1) = bottom + right - top - left.
@kernel function kernel_exterior_derivative_one!(res, s, @Const(f))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  e1, e2, e3, e4 = quad_edges(s, x, y)
  @inbounds res[idx] = f[e1] + f[e2] - f[e3] - f[e4]
end

function exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex2D, f)
  backend = get_backend(res)
  kernel_exterior_derivative_zero!(backend)(res, s, f; ndrange = size(res))
  return res
end

function exterior_derivative(::Val{0}, s::UniformCubicalComplex2D, f::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(f), FT, ne(s))
  return exterior_derivative!(res, Val(0), s, f)
end

function exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex2D, f)
  backend = get_backend(res)
  kernel_exterior_derivative_one!(backend)(res, s, f; ndrange = size(res))
  return res
end

function exterior_derivative(::Val{1}, s::UniformCubicalComplex2D, f::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(f), FT, nquads(s))
  return exterior_derivative!(res, Val(1), s, f)
end

# Wedge Products

@kernel function kernel_wedge_product_01(res, s, @Const(f), @Const(a))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  FT = eltype(a)
  v1 = f[src(s, x, y, align)]
  v2 = f[tgt(s, x, y, align)]

  @inbounds res[idx] = (v1 + v2) * a[idx] * FT(0.5)
end

@kernel function kernel_wedge_product_11(res, s, @Const(a), @Const(b))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  es = quad_edges(s, x, y)

  FT = eltype(a)
  @inbounds res[idx] = FT(0.25) * (a[es[1]] + a[es[3]]) * (b[es[2]] + b[es[4]]) -
    FT(0.25) * (a[es[2]] + a[es[4]]) * (b[es[1]] + b[es[3]])
end

@kernel function kernel_wedge_product_dual_01(res, s, @Const(f), @Const(a))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  dv1, dv2 = edge_quads(s, x, y, align)

  FT = eltype(a)
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
    tmp = FT(0.5) * (f[dv1] + f[dv2]) * ae
  end

  @inbounds res[idx] = tmp
end

@kernel function kernel_wedge_product_dd_11(res, s, @Const(a), @Const(b),)
  v = @index(Global)
  x, y = vert_to_coord(s, v)

  FT = eltype(a)
  z = zero(FT)

  has_xs = x < nx(s)
  has_ys = y < ny(s)
  has_xt = x > 1
  has_yt = y > 1

  axs = has_xs ? a[coord_to_edge(s, x,     y,     X_ALIGN)] : z
  ays = has_ys ? a[coord_to_edge(s, x,     y,     Y_ALIGN)] : z
  axt = has_xt ? a[coord_to_edge(s, x - 1, y,     X_ALIGN)] : z
  ayt = has_yt ? a[coord_to_edge(s, x,     y - 1, Y_ALIGN)] : z

  bxs = has_xs ? b[coord_to_edge(s, x,     y,     X_ALIGN)] : z
  bys = has_ys ? b[coord_to_edge(s, x,     y,     Y_ALIGN)] : z
  bxt = has_xt ? b[coord_to_edge(s, x - 1, y,     X_ALIGN)] : z
  byt = has_yt ? b[coord_to_edge(s, x,     y - 1, Y_ALIGN)] : z

  nx_valid = Int(has_xs) + Int(has_xt)
  ny_valid = Int(has_ys) + Int(has_yt)
  scale = inv(FT(nx_valid * ny_valid))

  @inbounds res[v] = scale * (
    (axs + axt) * (bys + byt) -
    (ays + ayt) * (bxs + bxt)
  )
end

# Implements Equation 70 of Kraus & Maj (2017).
@kernel function kernel_wedge_product_pd_11(res, s, @Const(a), @Const(b))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)
  e1, e2, e3, e4 = quad_edges(s, x, y)

  FT = eltype(a)
  x1, y1, al1 = edge_to_coord(s, e1)
  x2, y2, al2 = edge_to_coord(s, e2)
  x3, y3, al3 = edge_to_coord(s, e3)
  x4, y4, al4 = edge_to_coord(s, e4)

  w1 = is_boundary_edge(s, x1, y1, al1) ? FT(0.5) : one(FT)
  w2 = is_boundary_edge(s, x2, y2, al2) ? FT(0.5) : one(FT)
  w3 = is_boundary_edge(s, x3, y3, al3) ? FT(0.5) : one(FT)
  w4 = is_boundary_edge(s, x4, y4, al4) ? FT(0.5) : one(FT)

  @inbounds res[idx] = FT(0.5) * (w1 * a[e1] * b[e1] +
                  w3 * a[e3] * b[e3] -
                  w2 * a[e2] * b[e2] -
                  w4 * a[e4] * b[e4])
end

function wedge_product!(res, ::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  backend = get_backend(res)
  kernel_wedge_product_01(backend)(res, s, f, a; ndrange = size(res))
  return res
end

function wedge_product(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(f), FT, ne(s))
  return wedge_product!(res, Val(0), Val(1), s, f, a)
end

wedge_product!(res, ::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) =
  wedge_product!(res, Val(0), Val(1), s, f, a)

wedge_product(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, f::AbstractVector{FT}) where FT <: AbstractFloat =
  wedge_product(Val(0), Val(1), s, f, a)

function wedge_product!(res, ::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a, b)
  backend = get_backend(res)
  kernel_wedge_product_11(backend)(res, s, a, b; ndrange = size(res))
  return res
end

function wedge_product(::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(a), FT, nquads(s))
  return wedge_product!(res, Val(1), Val(1), s, a, b)
end

function wedge_product_dd!(res, ::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f, a)
  backend = get_backend(res)
  kernel_wedge_product_dual_01(backend)(res, s, f, a; ndrange = size(res))
  return res
end

function wedge_product_dd(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(f), FT, ne(s))
  return wedge_product_dd!(res, Val(0), Val(1), s, f, a)
end

wedge_product_dd!(res, ::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) =
  wedge_product_dd!(res, Val(0), Val(1), s, f, a)

wedge_product_dd(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, f::AbstractVector{FT}) where FT <: AbstractFloat =
  wedge_product_dd(Val(0), Val(1), s, f, a)

function wedge_product_dd!(res, ::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a, b)
  backend = get_backend(res)
  kernel_wedge_product_dd_11(backend)(res, s, a, b; ndrange = size(res))
  return res
end

function wedge_product_dd(::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(a), FT, nv(s))
  return wedge_product_dd!(res, Val(1), Val(1), s, a, b)
end

function wedge_product_pd!(res, ::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a, b)
  backend = get_backend(res)
  kernel_wedge_product_pd_11(backend)(res, s, a, b; ndrange = size(res))
  return res
end

function wedge_product_pd(::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(a), FT, nquads(s))
  return wedge_product_pd!(res, Val(1), Val(1), s, a, b)
end

# Convert a dual 1-form to a vector field on the dual points
@kernel function kernel_sharp_dd(X, Y, s, @Const(a))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)
  e1, e2, e3, e4 = quad_edges(s, x, y) # Order is (x-, y-, x+, y+)

  le1 = dual_edge_len(s, e1)
  le2 = dual_edge_len(s, e2)
  le3 = dual_edge_len(s, e3)
  le4 = dual_edge_len(s, e4)

  FT = eltype(a)
  # Remember that for an X-aligned primal edge, we have a Y-aligned dual edge
  X[idx] = -(a[e2]/le2 + a[e4]/le4) * FT(0.5)
  Y[idx] = (a[e1]/le1 + a[e3]/le3) * FT(0.5)
end

function sharp_dd!(X, Y, s::UniformCubicalComplex2D, a)
  backend = get_backend(X)
  kernel_sharp_dd(backend)(X, Y, s, a; ndrange = size(X))
  return X, Y
end

function sharp_dd(s::UniformCubicalComplex2D, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(a)
  X = KernelAbstractions.zeros(backend, FT, nquads(s))
  Y = KernelAbstractions.zeros(backend, FT, nquads(s))
  return sharp_dd!(X, Y, s, a)
end

# Convert dual vector field to primal 1-form
@kernel function kernel_flat_dp(res, s, @Const(X), @Const(Y))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)
  FT = eltype(X)
  if align == X_ALIGN
    if y == 1
    res[idx] = X[coord_to_quad(s,x,y)] * edge_len(s, X_ALIGN)
    elseif y == ny(s)
    res[idx] = X[coord_to_quad(s,x,y-1)] * edge_len(s, X_ALIGN)
    else
    res[idx] = FT(0.5) * (X[coord_to_quad(s,x,y)] + X[coord_to_quad(s,x,y-1)]) * edge_len(s, X_ALIGN)
    end
  else
    if x == 1
    res[idx] = Y[coord_to_quad(s,x,y)] * edge_len(s, Y_ALIGN)
    elseif x == nx(s)
    res[idx] = Y[coord_to_quad(s,x-1,y)] * edge_len(s, Y_ALIGN)
    else
    res[idx] = FT(0.5) * (Y[coord_to_quad(s,x,y)] + Y[coord_to_quad(s,x-1,y)]) * edge_len(s, Y_ALIGN)
    end
  end
end

function flat_dp!(res, s::UniformCubicalComplex2D, X, Y)
  backend = get_backend(res)
  kernel_flat_dp(backend)(res, s, X, Y; ndrange = size(res))
  return res
end

function flat_dp(s::UniformCubicalComplex2D, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(X), FT, ne(s))
  return flat_dp!(res, s, X, Y)
end

# Convert dual vector field to dual 1-form
@kernel function kernel_flat_dd(res, s, @Const(X), @Const(Y))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)
  FT = eltype(X)

  if align == X_ALIGN
    if y == 1
    res[idx] = Y[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif y == ny(s)
    res[idx] = Y[coord_to_quad(s, x, y - 1)] * dual_edge_len(s, idx)
    else
    res[idx] = FT(0.5) * (Y[coord_to_quad(s, x, y)] + Y[coord_to_quad(s, x, y - 1)]) * dual_edge_len(s, idx)
    end
  else
    if x == 1
    res[idx] = -X[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif x == nx(s)
    res[idx] = -X[coord_to_quad(s, x - 1, y)] * dual_edge_len(s, idx)
    else
    res[idx] = -FT(0.5) * (X[coord_to_quad(s, x, y)] + X[coord_to_quad(s, x - 1, y)]) * dual_edge_len(s, idx)
    end
  end
end

function flat_dd!(res, s::UniformCubicalComplex2D, X, Y)
  backend = get_backend(res)
  kernel_flat_dd(backend)(res, s, X, Y; ndrange = size(res))
  return res
end

function flat_dd(s::UniformCubicalComplex2D, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(X), FT, ne(s))
  return flat_dd!(res, s, X, Y)
end

function interpolate_dp!(res, ::Val{1}, s::UniformCubicalComplex2D, a)
  backend = get_backend(res)
  X = KernelAbstractions.zeros(backend, eltype(res), nquads(s))
  Y = KernelAbstractions.zeros(backend, eltype(res), nquads(s))
  sharp_dd!(X, Y, s, a)
  flat_dp!(res, s, X, Y)
  return res
end

function interpolate_dp(::Val{1}, s::UniformCubicalComplex2D, a::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(a), FT, ne(s))
  return interpolate_dp!(res, Val(1), s, a)
end
