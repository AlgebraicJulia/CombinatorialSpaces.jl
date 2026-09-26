using KernelAbstractions
using Adapt

### Exterior Derivatives ###

@kernel function kernel_exterior_derivative_zero_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = edge_to_coord(s, idx)

  @inbounds res[idx] = f[tgt(s, x, y, z, align)] - f[src(s, x, y, z, align)]
end

@kernel function kernel_exterior_derivative_one_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = quad_to_coord(s, idx)

  e1, e2, e3, e4 = quad_edges(s, x, y, z, align)
  @inbounds res[idx] = f[e1] + f[e2] - f[e3] - f[e4]
end

@kernel function kernel_exterior_derivative_two_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z = boid_to_coord(s, idx)
  q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)
  @inbounds res[idx] = f[q2] - f[q1] + f[q4] - f[q3] + f[q6] - f[q5]
end

function exterior_derivative!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_exterior_derivative_zero_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function exterior_derivative!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_exterior_derivative_one_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function exterior_derivative!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_exterior_derivative_two_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function exterior_derivative(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
  return exterior_derivative!(res, op, s, f)
end

function exterior_derivative(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
  return exterior_derivative!(res, op, s, f)
end

function exterior_derivative(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
  return exterior_derivative!(res, op, s, f)
end

### Hodge Star ###

@kernel function kernel_hodge_star_zero_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z = vert_to_coord(s, idx)

  ratio = dual_boid_volume(s, x, y, z)
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_hodge_star_one_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = edge_to_coord(s, idx)

  ratio = dual_quad_area(s, x, y, z, align) / edge_len(s, align)
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_hodge_star_two_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = quad_to_coord(s, idx)

  ratio = dual_edge_len(s, x, y, z, align) / quad_area(s, align)
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_hodge_star_three_3d!(res, s, @Const(f))
  idx = @index(Global)

  ratio = inv(boid_volume(s))
  @inbounds res[idx] = f[idx] * ratio
end

function hodge_star!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_hodge_star_zero_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function hodge_star!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_hodge_star_one_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function hodge_star!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_hodge_star_two_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function hodge_star!(res::AbstractVector{FT}, ::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_hodge_star_three_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
  return hodge_star!(res, op, s, f)
end

function hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
  return hodge_star!(res, op, s, f)
end

function hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
  return hodge_star!(res, op, s, f)
end

function hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
  return hodge_star!(res, op, s, f)
end

### Inverse Hodge Star ###

@kernel function kernel_inv_hodge_star_zero_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z = vert_to_coord(s, idx)

  ratio = inv(dual_boid_volume(s, x, y, z))
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_inv_hodge_star_one_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = edge_to_coord(s, idx)

  ratio = edge_len(s, align) / dual_quad_area(s, x, y, z, align)
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_inv_hodge_star_two_3d!(res, s, @Const(f))
  idx = @index(Global)
  x, y, z, align = quad_to_coord(s, idx)

  ratio = quad_area(s, align) / dual_edge_len(s, x, y, z, align)
  @inbounds res[idx] = f[idx] * ratio
end

@kernel function kernel_inv_hodge_star_three_3d!(res, s, @Const(f))
  idx = @index(Global)

  ratio = boid_volume(s)
  @inbounds res[idx] = f[idx] * ratio
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_inv_hodge_star_zero_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_inv_hodge_star_one_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_inv_hodge_star_two_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_inv_hodge_star_three_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function inv_hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
  return inv_hodge_star!(res, op, s, f)
end

function inv_hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
  return inv_hodge_star!(res, op, s, f)
end

function inv_hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
  return inv_hodge_star!(res, op, s, f)
end

function inv_hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
  return inv_hodge_star!(res, op, s, f)
end

### Dual Derivative ###

@kernel function kernel_dual_derivative_zero_3d!(res, s, @Const(f))
  idx = @index(Global)
  FT = eltype(f)
  x, y, z, align = quad_to_coord(s, idx)
  (b_indices, b_valid) = quad_boids(s, x, y, z, align)

  val1 = b_valid[1] ? f[b_indices[1]] : zero(FT)
  val2 = b_valid[2] ? f[b_indices[2]] : zero(FT)

  @inbounds res[idx] = val2 - val1
end

@kernel function kernel_dual_derivative_one_3d!(res, s, @Const(f))
  idx = @index(Global)
  FT = eltype(f)
  x, y, z, align = edge_to_coord(s, idx)
  (q_indices, q_valid) = edge_quads(s, x, y, z, align)

  val1 = q_valid[1] ? f[q_indices[1]] : zero(FT)
  val2 = q_valid[2] ? f[q_indices[2]] : zero(FT)
  val3 = q_valid[3] ? f[q_indices[3]] : zero(FT)
  val4 = q_valid[4] ? f[q_indices[4]] : zero(FT)

  @inbounds res[idx] = -val1 + val2 + val3 - val4
end

@kernel function kernel_dual_derivative_two_3d!(res, s, @Const(f))
  idx = @index(Global)
  FT = eltype(f)
  x, y, z = vert_to_coord(s, idx)
  (e_indices, e_valid) = vertex_edges(s, x, y, z)

  val1 = e_valid[1] ? f[e_indices[1]] : zero(FT)
  val2 = e_valid[2] ? f[e_indices[2]] : zero(FT)
  val3 = e_valid[3] ? f[e_indices[3]] : zero(FT)
  val4 = e_valid[4] ? f[e_indices[4]] : zero(FT)
  val5 = e_valid[5] ? f[e_indices[5]] : zero(FT)
  val6 = e_valid[6] ? f[e_indices[6]] : zero(FT)

  @inbounds res[idx] = val1 - val2 + val3 - val4 + val5 - val6
end

function dual_derivative!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_dual_derivative_zero_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function dual_derivative!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_dual_derivative_one_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function dual_derivative!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_dual_derivative_two_3d!(backend)
  kernel(res, s, f, ndrange=size(res))
  return res
end

function dual_derivative(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
  return dual_derivative!(res, op, s, f)
end

function dual_derivative(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
  return dual_derivative!(res, op, s, f)
end

function dual_derivative(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
  return dual_derivative!(res, op, s, f)
end

### Wedge Product ###

# Canonical orientations are dxdy, dzdx, dydz
@kernel function kernel_wedge_product_11_3d!(res, s, @Const(a), @Const(b))
  idx = @index(Global)
  x, y, z, align = quad_to_coord(s, idx)
  e1, e2, e3, e4 = quad_edges(s, x, y, z, align)

  # X_ALIGN first pair Y, second pair Z
  # Y_ALIGN first pair Z, second pair X
  # Z_ALIGN first pair X, second pair Y
  @inbounds begin
    FT = eltype(a)
    a1 = FT(0.5) * (a[e1] + a[e3]); a2 = FT(0.5) * (a[e2] + a[e4])
    b1 = FT(0.5) * (b[e1] + b[e3]); b2 = FT(0.5) * (b[e2] + b[e4])
    res[idx] = a1 * b2 - a2 * b1
  end
end

@kernel function kernel_wedge_product_12_3d!(res, s, @Const(a), @Const(b))
  idx = @index(Global)
  x, y, z = boid_to_coord(s, idx)
  q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)
  e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12 = boid_edges(s, x, y, z)

  @inbounds begin
    FT = eltype(a)
    ax = FT(0.25) * (a[e1] + a[e2] + a[e3] + a[e4])
    ay = FT(0.25) * (a[e5] + a[e6] + a[e7] + a[e8])
    az = FT(0.25) * (a[e9] + a[e10] + a[e11] + a[e12])

    # bx, by, bz are the dydz, dzdx, dxdy components respectively, so each
    # wedges with its matching edge component to give +dxdydz.
    bz = FT(0.5) * (b[q1] + b[q2])
    by = FT(0.5) * (b[q3] + b[q4])
    bx = FT(0.5) * (b[q5] + b[q6])
    res[idx] = ax * bx + ay * by + az * bz
  end
end

function wedge_product!(res::AbstractVector{FT}, ::Val{1}, ::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_wedge_product_11_3d!(backend)
  kernel(res, s, a, b, ndrange=size(res))
  return res
end

function wedge_product!(res::AbstractVector{FT}, ::Val{1}, ::Val{2}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_wedge_product_12_3d!(backend)
  kernel(res, s, a, b, ndrange=size(res))
  return res
end

function wedge_product(op1::Val{1}, op2::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(a)
  res = KernelAbstractions.zeros(backend, eltype(a), nquads(s))
  return wedge_product!(res, op1, op2, s, a, b)
end

function wedge_product(op1::Val{1}, op2::Val{2}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(a)
  res = KernelAbstractions.zeros(backend, eltype(a), nboids(s))
  return wedge_product!(res, op1, op2, s, a, b)
end
wedge_product(::Val{2}, ::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}) where FT <: AbstractFloat = wedge_product(Val(1), Val(2), s, b, a)

### Dual Wedge Product ###

@kernel function kernel_wedge_product_dd_01_3d!(res, s, @Const(f), @Const(a))
  idx = @index(Global)
  FT = eltype(f)
  x, y, z, align = quad_to_coord(s, idx)
  b_indices, b_valid = quad_boids(s, x, y, z, align)

  f_val = if b_valid[1] && b_valid[2]
    FT(0.5) * (f[b_indices[1]] + f[b_indices[2]])
  elseif b_valid[1]
    f[b_indices[1]]
  elseif b_valid[2]
    f[b_indices[2]]
  else
    zero(FT)
  end

  @inbounds res[idx] = f_val * a[idx]
end

function wedge_product_dd!(res::AbstractVector{FT}, ::Val{0}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_wedge_product_dd_01_3d!(backend)
  kernel(res, s, f, a, ndrange=size(res))
  return res
end

function wedge_product_dd(op1::Val{0}, op2::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
  return wedge_product_dd!(res, op1, op2, s, f, a)
end

### Sharp and Flat Operators ###

@kernel function kernel_sharp_dd_3d!(X, Y, Z, s, @Const(f))
  idx = @index(Global)
  x, y, z = boid_to_coord(s, idx)
  FT = eltype(f)

  # boid_quads returns faces in order: Z-low, Z-high, Y-low, Y-high, X-low, X-high
  q_z1, q_z2, q_y1, q_y2, q_x1, q_x2 = boid_quads(s, x, y, z)

  # --- X Component ---
  val_x1 = f[q_x1] # West
  val_x2 = f[q_x2] # East

  local_X = zero(FT)
  local_X += ifelse(x == 1, FT(1.0), FT(0.5)) * val_x1
  local_X += ifelse(x == nxb(s), FT(1.0), FT(0.5)) * val_x2

  # --- Y Component ---
  val_y1 = f[q_y1] # South
  val_y2 = f[q_y2] # North

  local_Y = zero(FT)
  local_Y += ifelse(y == 1, FT(1.0), FT(0.5)) * val_y1
  local_Y += ifelse(y == nyb(s), FT(1.0), FT(0.5)) * val_y2

  # --- Z Component ---
  val_z1 = f[q_z1] # Down
  val_z2 = f[q_z2] # Up

  local_Z = zero(FT)
  local_Z += ifelse(z == 1, FT(1.0), FT(0.5)) * val_z1
  local_Z += ifelse(z == nzb(s), FT(1.0), FT(0.5)) * val_z2

  @inbounds begin
    X[idx] = local_X / dx(s)
    Y[idx] = local_Y / dy(s)
    Z[idx] = local_Z / dz(s)
  end
end

@kernel function kernel_flat_dp_3d!(res, s, @Const(X), @Const(Y), @Const(Z))
  idx = @index(Global)
  x, y, z, align = edge_to_coord(s, idx)
  FT = eltype(res)

  b_indices, b_valid = edge_boids(s, x, y, z, align)

  total_val = zero(FT)
  valid_boids = 0

  V = if align == X_ALIGN
    X
  elseif align == Y_ALIGN
    Y
  else # Z_ALIGN
    Z
  end

  # Sum the relevant vector component from all valid adjacent boids
  if b_valid[1]; total_val += V[b_indices[1]]; valid_boids += 1; end
  if b_valid[2]; total_val += V[b_indices[2]]; valid_boids += 1; end
  if b_valid[3]; total_val += V[b_indices[3]]; valid_boids += 1; end
  if b_valid[4]; total_val += V[b_indices[4]]; valid_boids += 1; end

  # Calculate the average and multiply by the edge length
  avg_val = valid_boids > 0 ? total_val / valid_boids : zero(FT)

  @inbounds res[idx] = avg_val * edge_len(s, align)
end

function sharp_dd!(X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}, s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(X)
  kernel = kernel_sharp_dd_3d!(backend)
  kernel(X, Y, Z, s, f, ndrange=size(X))
  return X, Y, Z
end

function sharp_dd(s::UniformCubicalComplex3D, f::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)

  # The resulting vector field lives on the dual vertices, which correspond
  # to the primal boids.
  X = KernelAbstractions.zeros(backend, FT, nboids(s))
  Y = KernelAbstractions.zeros(backend, FT, nboids(s))
  Z = KernelAbstractions.zeros(backend, FT, nboids(s))

  return sharp_dd!(X, Y, Z, s, f)
end

function flat_dp!(res::AbstractVector{FT}, s::UniformCubicalComplex3D, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  kernel = kernel_flat_dp_3d!(backend)
  kernel(res, s, X, Y, Z, ndrange=size(res))
  return res
end

function flat_dp(s::UniformCubicalComplex3D, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(X)

  # The result is a primal 1-form, which lives on the primal edges.
  res = KernelAbstractions.zeros(backend, FT, ne(s))

  return flat_dp!(res, s, X, Y, Z)
end

function interpolate_dp!(res::AbstractVector{FT}, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}, ::Val{1},
    s::UniformCubicalComplex3D, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(res)
  sharp_dd!(X, Y, Z, s, a)
  return flat_dp!(res, s, X, Y, Z)
end

function interpolate_dp(::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}) where FT <: AbstractFloat
  X, Y, Z = sharp_dd(s, a)
  return flat_dp(s, X, Y, Z)
end

# These operators exist for UniformCubicalComplex2D but have no 3D counterpart
# yet. The stubs below intercept the natural 3D call and
# give a clear error instead of a raw MethodError.

_not_implemented_3d(name) = error("$name is not yet implemented for UniformCubicalComplex3D")

codifferential(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("codifferential")

dual_codifferential(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("dual_codifferential")

laplacian(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("laplacian")

dual_laplacian(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("dual_laplacian")

d_beta(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("d_beta")

wedge_product_pd(::Val, ::Val, s::UniformCubicalComplex3D, a::AbstractVector, b::AbstractVector) = _not_implemented_3d("wedge_product_pd")
wedge_product_pd!(res, ::Val, ::Val, s::UniformCubicalComplex3D, a::AbstractVector, b::AbstractVector) = _not_implemented_3d("wedge_product_pd!")

flat_dd(s::UniformCubicalComplex3D, X::AbstractVector, Y::AbstractVector, Z::AbstractVector) = _not_implemented_3d("flat_dd")
flat_dd!(res, s::UniformCubicalComplex3D, X::AbstractVector, Y::AbstractVector, Z::AbstractVector) = _not_implemented_3d("flat_dd!")

no_flux_dual_derivative(::Val, s::UniformCubicalComplex3D) = _not_implemented_3d("no_flux_dual_derivative")
