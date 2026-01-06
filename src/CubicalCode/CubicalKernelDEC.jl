function exterior_derivative!(res, ::Val{0}, f)
  backend = get_backend(f)

  for edge_set in res
    @assert backend == get_backend(edge_set)
  end

  for (i, edge_set) in enumerate(res)
    kernel = kernel_exterior_derivative_zero(backend)
    kernel(edge_set, i, f, ndrange = size(edge_set))
  end
  return res
end

# TODO: Can move the horizontal/vertical handling into the higher level function
@kernel function kernel_exterior_derivative_zero(res, z::Int, @Const(f))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = f[tgt(z, x, y)...] - f[src(z, x, y)...]
end

function exterior_derivative!(res, ::Val{1}, f)
  backend = get_backend(res)

  for edge_set in f
    @assert backend == get_backend(edge_set)
  end

  kernel = kernel_exterior_derivative_one(backend, 32, size(res))
  kernel(res, f, ndrange = size(res))
  return res
end

@kernel function kernel_exterior_derivative_one(res, @Const(f))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  hes = xedges(f)
  ves = yedges(f)

  b_xe, t_xe, l_ye, r_ye = quad_edges(x, y)

  @inbounds res[idx] = hes[b_xe...] - hes[t_xe...] - ves[l_ye...] + ves[r_ye...]
end

function dual_derivative!(res, ::Val{0}, s::HasCubicalComplex, f; padding = 0)
  backend = get_backend(f)

  for edge_set in res
    @assert backend == get_backend(edge_set)
  end

  for (i, edge_set) in enumerate(res)
    kernel = kernel_dual_derivative_zero(backend, 32, size(edge_set))
    kernel(edge_set, s, i, f, padding, ndrange = size(edge_set))
  end
  return res
end

@kernel function kernel_dual_derivative_zero(res, s::EmbeddedCubicalComplex2D, z::Int, f, padding)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  tl_q, br_q = edge_quads(z, x, y)

  res[idx] = get_zerodf(s, f, br_q..., padding = padding) - get_zerodf(s, f, tl_q..., padding = padding)
end

function dual_derivative!(res, ::Val{1}, s::HasCubicalComplex, f; padding = 0)
  backend = get_backend(res)

  for edge_set in f
    @assert backend == get_backend(edge_set)
  end

  kernel = kernel_dual_derivative_one(backend, 32, size(res))
  kernel(res, s, f, padding, ndrange = size(res))
  return res
end

@kernel function kernel_dual_derivative_one(res, s::EmbeddedCubicalComplex2D, @Const(f), padding::Real)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  # bottom, right, top, left, ccw orientation
  # ---->
  # | . |
  # V   V
  # ---->

  b_dxe, t_dxe, l_dye, r_dye = vertex_edges(x, y)

  @inbounds res[idx] = get_onedf(s, f, 2, b_dxe...; padding = padding) - get_onedf(s, f, 1, r_dye...; padding = padding) - 
                       get_onedf(s, f, 2, t_dxe...; padding = padding) + get_onedf(s, f, 1, l_dye...; padding = padding)
end

function hodge_star!(res, ::Val{0}, s::HasCubicalComplex, f; inv::Bool = false)
  backend = get_backend(f)

  @assert backend == get_backend(res)

  kernel = kernel_hodge_star_zero(backend, 32, size(res))
  kernel(res, s, f, inv, ndrange = size(res))
  return res
end

@kernel function kernel_hodge_star_zero(res, s::HasCubicalComplex, @Const(f), is_inv::Bool)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  dqa = d_quad_area(s, x, y)
  α = is_inv ? inv(dqa) : dqa

  @inbounds res[idx] = α * f[idx]
end

function hodge_star!(res, ::Val{1}, s::HasCubicalComplex, f; inv::Bool = false)
  backend = get_backend(xedges(res))

  for (i, (res_set, f_set)) in enumerate(zip(res, f))
    kernel = kernel_hodge_star_one(backend, 32, size(res_set))
    kernel(res_set, s, i, f_set, inv, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_hodge_star_one(res, s::HasCubicalComplex, z::Int, @Const(f), is_inv::Bool)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  del = d_edge_len(s, z, x, y)
  el = edge_len(s, z, x, y)

  α = is_inv ? (el / del) : (del / el)

  @inbounds res[idx] = α * f[idx]
end

function hodge_star!(res, ::Val{2}, s::HasCubicalComplex, f; inv::Bool = false)
  backend = get_backend(res)

  kernel = kernel_hodge_star_two(backend, 32, size(res))
  kernel(res, s, f, inv, ndrange = size(res))
  return res
end

@kernel function kernel_hodge_star_two(res, s::HasCubicalComplex, @Const(f), is_inv::Bool)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  qa = quad_area(s, x, y)

  α = is_inv ? 1 / qa : qa

  @inbounds res[idx] = α * f[idx]
end

function wedge_product!(res, ::Val{(0,1)}, s::HasCubicalComplex, f, α)
  backend = get_backend(f)

  for (i, (res_set, α_set)) in enumerate(zip(res, α))
    kernel = kernel_wedge_product_zero_one(backend, 32, size(res_set))
    kernel(res_set, i, f, α_set, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_wedge_product_zero_one(res, z::Int, @Const(f), @Const(α))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = 0.5 * (f[src(z, x, y)...] + f[tgt(z, x, y)...]) * α[idx]
end

function wedge_product!(res, ::Val{(1,1)}, s::HasCubicalComplex, α, β)
  backend = get_backend(xedges(α))

  kernel = kernel_wedge_product_one_one(backend, 32, size(res))
  kernel(res, α, β, ndrange = size(res))
  return res
end

@kernel function kernel_wedge_product_one_one(res, @Const(α), @Const(β))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  xα = xedges(α); yα = yedges(α)
  xβ = xedges(β); yβ = yedges(β)


  # a ∧ b = (a[x]b[y] - a[y]b[x])dx ∧ dy
  e1, e2, e3, e4 = quad_edges(x, y)
  @inbounds res[idx] = 0.25 * ((xα[e1...] + xα[e2...]) * (yβ[e3...] + yβ[e4...]) - 
    (yα[e3...] + yα[e4...]) * (xβ[e1...] + xβ[e2...]))
end

# function wedge_product(::Val{(0,2)}, s::HasCubicalComplex, f, alpha)
#   res = similar(alpha)

#   for q in quadrilaterals(s)
#     v1, v2, v3, v4 = quad_vertices(s, q)
#     res[q] = 0.25 * (getindex(f, v1) + getindex(f, v2) + getindex(f, v3) + getindex(f, v4)) * alpha[q]
#   end

#   return res
# end

function wedge_product_dd!(res, ::Val{(0,1)}, s::HasCubicalComplex, f, α)
  backend = get_backend(f)

  for (i, (res_set, α_set)) in enumerate(zip(res, α))
    kernel = kernel_wedge_product_dd_zero_one(backend, 32, size(res_set))
    kernel(res_set, i, f, α_set, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_wedge_product_dd_zero_one(res, z::Int, @Const(f), @Const(α))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  src_w, tgt_w = d_edge_ratio(s, z, x, y)

  f_dsrc = get_zerodf(s, f, d_src(z, x, y)...) 
  f_dtgt = get_zerodf(s, f, d_tgt(z, x, y)...) 

  @inbounds res[idx] = (src_w * f_dsrc + tgt_w * f_dtgt) * α[idx]
end

function sharp_dd!(X, Y, s::HasCubicalComplex, f)
  backend = get_backend(X)

  @assert size(X) == size(Y)

  kernel = kernel_sharp_dd(backend, 32, size(X))
  kernel(X, Y, s, f, ndrange = size(X))
  return X, Y
end

# Take dual 1-form and output dual vector field
@kernel function kernel_sharp_dd(X, Y, s::EmbeddedCubicalComplex2D, @Const(f))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  deb, det, del, der = quad_edges(x, y)

  @inbounds X[idx] = 0.5 * (d_xedges(f)[del...] / d_xedge_len(s, del...) + d_xedges(f)[der...] / d_xedge_len(s, der...))
  @inbounds Y[idx] = -0.5 * (d_yedges(f)[deb...] / d_yedge_len(s, deb...) + d_yedges(f)[det...] / d_yedge_len(s, det...))
end

function flat_dp!(res, s::HasCubicalComplex, X, Y)
  backend = get_backend(X)

  @assert size(X) == size(Y)

  for (i, (res_set, vec_set)) in enumerate(zip(res, (X, Y)))
    kernel = kernel_flat_dp(backend, 32, size(res_set))
    kernel(res_set, s, i, vec_set, ndrange = size(res_set))
  end

  return res
end

# Take dual 1-form and output dual vector field
@kernel function kernel_flat_dp(f, s::EmbeddedCubicalComplex2D, z::Int, @Const(V))
  idx = @index(Global, Cartesian)
  x, y = idx.I

  tlq, brq = edge_quads(z, x, y)
  l = edge_len(s, z, x, y)

  avg_denom = 0 # Divide for linear interpolation
  tot = 0 # Local total to be stored globally

  if valid_quad(s, tlq...) # Top or left quad
    α = is_xedge(z) ? quad_width(s, tlq...) : quad_height(s, tlq...)
    avg_denom += α

    tot += α * V[tlq...]
  end
  
  if valid_quad(s, brq...) # Bottom or right quad
    β = is_xedge(z) ? quad_width(s, brq...) : quad_height(s, brq...)
    avg_denom += β

    tot += β * V[brq...]
  end

  @inbounds f[idx] = tot * l / (avg_denom)
end

# Take dual vector field and output primal 1-form
# TODO: Might need to weight by dimensions of quad
function flat_dp(s::HasCubicalComplex, X, Y)
  alpha = zeros(eltype(X), nde(s))

  for q in quadrilaterals(s)
    h = quad_height(s, q)
    w = quad_width(s, q)

    eb, et, el, er = quad_edges(s, q)
    alpha[el] += 0.5 * h * Y[q]
    alpha[er] += 0.5 * h * Y[q]

    alpha[eb] += 0.5 * w * X[q]
    alpha[et] += 0.5 * w * X[q]
  end

  return alpha
end