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
  idx_2 = @index(Global, Cartesian)
  idx_3 = CartesianIndex(z, idx_2.I...)

  @inbounds res[idx_2] = f[tgt(idx_3)] - f[src(idx_3)]
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

@kernel function kernel_exterior_derivative_one(res, f)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  hes = xedges(f)
  ves = yedges(f)

  @inbounds res[idx] = hes[x, y] - hes[x, y + 1] - ves[x, y] + ves[x + 1, y]
end

function dual_derivative!(res, ::Val{0}, f; padding = 0)
  backend = get_backend(f)

  @assert backend == get_backend(f)

  for (i, edge_set) in enumerate(res)
    kernel = kernel_dual_derivative_zero(backend, 32, size(edge_set))
    kernel(edge_set, s, i == 1, f, padding, ndrange = size(edge_set))
  end
  return res
end

@kernel function kernel_dual_derivative_zero(res, s::HasCubicalComplex, is_h::Bool, f, padding)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds if is_h # Horizontal edges
    res[idx] = get_zerodf(s, x, y - 1, f, padding = padding) - get_zerodf(s, x, y, f, padding = padding)
  else # Vertical edges
    res[idx] = get_zerodf(s, x, y, f, padding = padding) - get_zerodf(s, x - 1, y, f, padding = padding)
  end
end

function dual_derivative!(res, ::Val{1}, s::HasCubicalComplex, f; padding = 0)
  backend = get_backend(res)

  @assert backend == get_backend(xedges(f))
  @assert backend == get_backend(yedges(f))

  kernel = kernel_dual_derivative_one(backend, 32, size(res))
  kernel(res, s, f, padding, ndrange = size(res))
  return res
end

@kernel function kernel_dual_derivative_one(res, s::HasCubicalComplex, f, padding)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  # bottom, right, top, left, ccw orientation
  # ---->
  # | . |
  # V   V
  # ---->
  @inbounds res[idx] = get_onedf(s, 2, x, y - 1, f; padding = padding) - get_onedf(s, 1, x, y, f; padding = padding) - 
                       get_onedf(s, 2, x, y, f; padding = padding) + get_onedf(s, 1, x - 1, y, f; padding = padding)
end

function hodge_star!(res, ::Val{0}, s::HasCubicalComplex, f; inv::Bool = false)
  backend = get_backend(f)

  @assert backend == get_backend(res)

  args = (backend, 32, size(res))
  kernel = inv ? kernel_inv_hodge_star_zero(args...) : kernel_hodge_star_zero(args...)
  kernel(res, s, f, ndrange = size(res))
  return res
end

@kernel function kernel_hodge_star_zero(res, s::HasCubicalComplex, f)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = dual_quad_area(s, x, y) * f[idx]
end

@kernel function kernel_inv_hodge_star_zero(res, s::HasCubicalComplex, f)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = f[idx] / dual_quad_area(s, x, y)
end

function hodge_star!(res, ::Val{1}, s::HasCubicalComplex, f; inv::Bool = false)
  backend = get_backend(xedges(res))

  for (i, (res_set, f_set)) in enumerate(zip(res, f))
    args = (backend, 32, size(res_set))
    kernel = inv ? kernel_inv_hodge_star_one(args...) : kernel_hodge_star_one(args...)
    kernel(res_set, s, i == 1, f_set, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_hodge_star_one(res, s::HasCubicalComplex, is_h::Bool, f)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = dual_edge_length(s, x, y, is_h) * f[idx] / edge_length(s, x, y, is_h)
end

@kernel function kernel_inv_hodge_star_one(res, s::HasCubicalComplex, is_h::Bool, f)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = edge_length(s, x, y, is_h) * f[idx] / dual_edge_length(s, x, y, is_h)
end

function wedge_product!(res, ::Val{(0,1)}, s::HasCubicalComplex, f, α)
  backend = get_backend(f)

  for (i, (res_set, α_set)) in enumerate(zip(res, α))
    kernel = kernel_wedge_product_zero_one(backend, 32, size(res_set))
    kernel(res_set, i == 1, f, α_set, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_wedge_product_zero_one(res, is_h::Bool, f, α)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = 0.5 * (f[src(x, y)] + f[tgt(x, y, is_h)]) * α[idx]
end

function wedge_product!(res, ::Val{(1,1)}, s::HasCubicalComplex, α, β)
  backend = get_backend(xedges(f))

  kernel = kernel_wedge_product_one_one(backend, 32, size(res))
  kernel(res_set, f, α_set, ndrange = size(res_set))
  return res
end

@kernel function kernel_wedge_product_one_one(res, α, β)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  # a ∧ b = (a[x]b[y] - a[y]b[x])dx ∧ dy
  e1, e2, e3, e4 = quad_edges(x, y)
  @inbounds res[q] = 0.25 * ((α[e1] + α[e2]) * (β[e3] + β[e4]) - (α[e3] + α[e4]) * (β[e1] + β[e2]))
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
    kernel(res_set, i == 1, f, α_set, ndrange = size(res_set))
  end
  return res
end

@kernel function kernel_wedge_product_zero_one(res, is_h::Bool, f, α)
  idx = @index(Global, Cartesian)
  x, y = idx.I

  @inbounds res[idx] = 0.5 * (f[src(x, y)] + f[tgt(x, y, is_h)]) * α[idx]
end
