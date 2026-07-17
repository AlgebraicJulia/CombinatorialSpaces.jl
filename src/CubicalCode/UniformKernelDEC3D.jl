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

function exterior_derivative!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_zero_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_one_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_two_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function exterior_derivative(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function exterior_derivative(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Hodge Star ###

@kernel function kernel_hodge_star_zero_3d!(res, s, @Const(f))
    idx = @index(Global) 
    v_idx = idx
    x, y, z = vert_to_coord(s, v_idx)

    ratio = dual_boid_volume(s, x, y, z)
    @inbounds res[idx] = f[v_idx] * ratio
end

@kernel function kernel_hodge_star_one_3d!(res, s, @Const(f))
    idx = @index(Global)
    e_idx = idx
    x, y, z, align = edge_to_coord(s, e_idx)

    ratio = dual_quad_area(s, x, y, z, align) / edge_len(s, align)
    @inbounds res[idx] = f[e_idx] * ratio
end

@kernel function kernel_hodge_star_two_3d!(res, s, @Const(f))
    idx = @index(Global) 
    q_idx = idx 
    x, y, z, align = quad_to_coord(s, q_idx)

    ratio = dual_edge_len(s, x, y, z, align) / quad_area(s, align)
    @inbounds res[idx] = f[q_idx] * ratio
end

@kernel function kernel_hodge_star_three_3d!(res, s, @Const(f))
    idx = @index(Global) 
    b_idx = idx 
    x, y, z = boid_to_coord(s, b_idx)
    
    ratio = 1.0 / boid_volume(s)
    @inbounds res[idx] = f[b_idx] * ratio
end

function hodge_star!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_hodge_star_zero_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_hodge_star_one_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_hodge_star_two_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res::AbstractVector{FT}, ::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_hodge_star_three_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Inverse Hodge Star ###

@kernel function kernel_inv_hodge_star_zero_3d!(res, s, @Const(f))
    idx = @index(Global)
    b_idx = idx
    x, y, z = vert_to_coord(s, idx)

    ratio = 1.0 / dual_boid_volume(s, x, y, z)
    @inbounds res[idx] = f[b_idx] * ratio
end

@kernel function kernel_inv_hodge_star_one_3d!(res, s, @Const(f))
    idx = @index(Global)
    q_idx = idx
    x, y, z, align = edge_to_coord(s, idx)

    ratio = edge_len(s, align) / dual_quad_area(s, x, y, z, align)
    @inbounds res[idx] = f[q_idx] * ratio
end

@kernel function kernel_inv_hodge_star_two_3d!(res, s, @Const(f))
    idx = @index(Global)
    e_idx = idx
    x, y, z, align = quad_to_coord(s, idx)
    
    ratio = quad_area(s, align) / dual_edge_len(s, x, y, z, align)
    @inbounds res[idx] = f[e_idx] * ratio
end

@kernel function kernel_inv_hodge_star_three_3d!(res, s, @Const(f))
    idx = @index(Global)
    v_idx = idx
    
    ratio = boid_volume(s)
    @inbounds res[idx] = f[v_idx] * ratio
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_zero_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_one_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_two_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res::AbstractVector{FT}, ::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_three_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
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

@kernel function kernel_free_slip_dd1_cached_3d!(res, s, @Const(dd1_emask), @Const(f))
    idx = @index(Global)
    x, y, z, align = edge_to_coord(s, idx)
    (q_indices, q_valid) = edge_quads(s, x, y, z, align)
    begin
        mask = dd1_emask[idx]
        z = zero(eltype(f))

        val1 = Bool(mask & Int8(1)) ? f[q_indices[1]] : z
        val2 = Bool((mask >> Int8(1)) & Int8(1)) ? f[q_indices[2]] : z
        val3 = Bool((mask >> Int8(2)) & Int8(1)) ? f[q_indices[3]] : z
        val4 = Bool((mask >> Int8(3)) & Int8(1)) ? f[q_indices[4]] : z

        @inbounds res[idx] = -val1 + val2 + val3 - val4
    end
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

function dual_derivative!(res::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_dual_derivative_zero_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function dual_derivative!(res::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_dual_derivative_one_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function free_slip_dd1!(res::AbstractVector{FT}, s::UniformCubicalComplex3D, free_dd1_emask::AbstractVector{Int8}, f::AbstractVector{FT}; workgroup_size::Int = 256) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_free_slip_dd1_cached_3d!(backend, workgroup_size)
    kernel(res, s, free_dd1_emask, f, ndrange = size(res))
    return res
end

function dual_derivative!(res::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_dual_derivative_two_3d!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function dual_derivative(op::Val{0}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    dual_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function dual_derivative(op::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    dual_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function dual_derivative(op::Val{2}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    dual_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
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
        a1 = 0.5 * (a[e1] + a[e3]); a2 = 0.5 * (a[e2] + a[e4])
        b1 = 0.5 * (b[e1] + b[e3]); b2 = 0.5 * (b[e2] + b[e4])
        res[idx] = a1 * b2 - a2 * b1
    end
end

@kernel function kernel_wedge_product_12_3d!(res, s, @Const(a), @Const(b))
    idx = @index(Global)
    x, y, z = boid_to_coord(s, idx)
    q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)
    e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12 = boid_edges(s, x, y, z)
    
    @inbounds begin
        ax = 0.25 * (a[e1] + a[e2] + a[e3] + a[e4])
        ay = 0.25 * (a[e5] + a[e6] + a[e7] + a[e8])
        az = 0.25 * (a[e9] + a[e10] + a[e11] + a[e12])
    
        bz = 0.5 * (b[q1] + b[q2])
        by = 0.5 * (b[q3] + b[q4])
        bx = 0.5 * (b[q5] + b[q6])
        res[idx] = ax * bx - ay * by + az * bz
    end
end

function wedge_product!(res::AbstractVector{FT}, ::Val{1}, ::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_wedge_product_11_3d!(backend, workgroup_size)
    kernel(res, s, a, b, ndrange=size(res))
end

function wedge_product!(res::AbstractVector{FT}, ::Val{1}, ::Val{2}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_wedge_product_12_3d!(backend, workgroup_size)
    kernel(res, s, a, b, ndrange=size(res))
end

function wedge_product(op1::Val{1}, op2::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, eltype(a), nquads(s))
    wedge_product!(res, op1, op2, s, a, b; workgroup_size=workgroup_size)
    return res
end

function wedge_product(op1::Val{1}, op2::Val{2}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, eltype(a), nboids(s))
    wedge_product!(res, op1, op2, s, a, b; workgroup_size=workgroup_size)
    return res
end
wedge_product(::Val{2}, ::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}, b::AbstractVector{FT}; workgroup_size = 256) where FT <: AbstractFloat = wedge_product(Val(1), Val(2), s, a, b; workgroup_size=workgroup_size)

### Dual Wedge Product ###

@kernel function kernel_wedge_product_dd_01_3d!(res, s, @Const(f), @Const(a))
    idx = @index(Global)
    FT = eltype(f)
    x, y, z, align = quad_to_coord(s, idx)
    b_indices, b_valid = quad_boids(s, x, y, z, align)
    
    f_val = if b_valid[1] && b_valid[2]
        0.5 * (f[b_indices[1]] + f[b_indices[2]])
    elseif b_valid[1]
        f[b_indices[1]]
    elseif b_valid[2]
        f[b_indices[2]]
    else
        zero(FT)
    end
    
    @inbounds res[idx] = f_val * a[idx]
end

function wedge_product_dd!(res::AbstractVector{FT}, ::Val{0}, ::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}, a::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_wedge_product_dd_01_3d!(backend, workgroup_size)
    kernel(res, s, f, a, ndrange=size(res))
end

function wedge_product_dd(op1::Val{0}, op2::Val{1}, s::UniformCubicalComplex3D, f::AbstractVector{FT}, a::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    wedge_product_dd!(res, op1, op2, s, f, a; workgroup_size=workgroup_size)
    return res
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

function sharp_dd!(X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}, s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(X)
    kernel = kernel_sharp_dd_3d!(backend, workgroup_size)
    kernel(X, Y, Z, s, f, ndrange=size(X))
    return X, Y, Z
end

function sharp_dd(s::UniformCubicalComplex3D, f::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(f)
    
    # The resulting vector field lives on the dual vertices, which correspond
    # to the primal boids.
    X = KernelAbstractions.zeros(backend, FT, nboids(s))
    Y = KernelAbstractions.zeros(backend, FT, nboids(s))
    Z = KernelAbstractions.zeros(backend, FT, nboids(s))

    sharp_dd!(X, Y, Z, s, f; workgroup_size=workgroup_size)
    return (X, Y, Z)
end

function flat_dp!(res::AbstractVector{FT}, s::UniformCubicalComplex3D, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(res)
    kernel = kernel_flat_dp_3d!(backend, workgroup_size)
    kernel(res, s, X, Y, Z, ndrange=size(res))
    return res
end

function flat_dp(s::UniformCubicalComplex3D, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}; workgroup_size::Int = 256) where FT <: AbstractFloat
    backend = get_backend(X)
    
    # The result is a primal 1-form, which lives on the primal edges.
    res = KernelAbstractions.zeros(backend, FT, ne(s))

    flat_dp!(res, s, X, Y, Z; workgroup_size=workgroup_size)
    return res
end

function interpolate_dp!(res::AbstractVector{FT}, X::AbstractVector{FT}, Y::AbstractVector{FT}, Z::AbstractVector{FT}, ::Val{1}, 
        s::UniformCubicalComplex3D, a::AbstractVector{FT}) where FT <: AbstractFloat
    backend = get_backend(a)
    sharp_dd!(X, Y, Z, s, a)
    KernelAbstractions.synchronize(backend)
    return flat_dp!(res, s, X, Y, Z)
end
  
function interpolate_dp(::Val{1}, s::UniformCubicalComplex3D, a::AbstractVector{FT}) where FT <: AbstractFloat
    backend = get_backend(a)
    X, Y, Z = sharp_dd(s, a)
    KernelAbstractions.synchronize(backend)
    return flat_dp(s, X, Y, Z)
end

struct SmoothingCache3D{IT <: AbstractVector{Int32},
    FT <: AbstractVector,
    MT <: AbstractVector{Int8}}
    nboids_ :: Int

    # Neighbor boid indices (dummy = 1 for missing boundary neighbors)
    b_west  :: IT;  b_east  :: IT
    b_south :: IT;  b_north :: IT
    b_down  :: IT;  b_up    :: IT

    # Per-boid neighbor weights
    w_west  :: FT;  w_east  :: FT
    w_south :: FT;  w_north :: FT
    w_down  :: FT;  w_up    :: FT

    # Scalar diagonal weights (same for all boids, differ between passes)
    diag_fwd :: Float64   # = 1 - c_smooth/2
    diag_bwd :: Float64   # = 1 + c_smooth/2

    # Bits 0..5 → west, east, south, north, down, up exist
    b_smask :: MT
end

Adapt.@adapt_structure SmoothingCache3D

function SmoothingCache3D(s::UniformCubicalComplex3D{FT}, c_smooth::Real) where FT <: AbstractFloat
    nb_    = nboids(s)
    c      = FT(c_smooth) / 2
    inv_dx = 1.0 / dx(s)
    inv_dy = 1.0 / dy(s)
    inv_dz = 1.0 / dz(s)
    nxb_   = nxb(s);  nyb_ = nyb(s);  nzb_ = nzb(s)

    b_west  = Vector{Int32}(undef, nb_);  b_east  = Vector{Int32}(undef, nb_)
    b_south = Vector{Int32}(undef, nb_);  b_north = Vector{Int32}(undef, nb_)
    b_down  = Vector{Int32}(undef, nb_);  b_up    = Vector{Int32}(undef, nb_)

    w_west  = Vector{FT}(undef, nb_);  w_east  = Vector{FT}(undef, nb_)
    w_south = Vector{FT}(undef, nb_);  w_north = Vector{FT}(undef, nb_)
    w_down  = Vector{FT}(undef, nb_);  w_up    = Vector{FT}(undef, nb_)

    b_smask = Vector{Int8}(undef, nb_)

    for b in 1:nb_
        x, y, z = boid_to_coord(s, b)

        has_west  = x > 1;     has_east  = x < nxb_
        has_south = y > 1;     has_north = y < nyb_
        has_down  = z > 1;     has_up    = z < nzb_

        tot_w = (Int(has_west)  + Int(has_east))  * inv_dx +
                (Int(has_south) + Int(has_north)) * inv_dy +
                (Int(has_down)  + Int(has_up))    * inv_dz

        if tot_w > 0
            scale = c / tot_w
            w_west[b]  = has_west  ? scale * inv_dx : 0.0
            w_east[b]  = has_east  ? scale * inv_dx : 0.0
            w_south[b] = has_south ? scale * inv_dy : 0.0
            w_north[b] = has_north ? scale * inv_dy : 0.0
            w_down[b]  = has_down  ? scale * inv_dz : 0.0
            w_up[b]    = has_up    ? scale * inv_dz : 0.0
        else
            w_west[b] = w_east[b] = w_south[b] =
            w_north[b] = w_down[b] = w_up[b] = 0.0
        end

        b_west[b]  = Int32(has_west  ? coord_to_boid(s, x-1, y, z) : 1)
        b_east[b]  = Int32(has_east  ? coord_to_boid(s, x+1, y, z) : 1)
        b_south[b] = Int32(has_south ? coord_to_boid(s, x, y-1, z) : 1)
        b_north[b] = Int32(has_north ? coord_to_boid(s, x, y+1, z) : 1)
        b_down[b]  = Int32(has_down  ? coord_to_boid(s, x, y, z-1) : 1)
        b_up[b]    = Int32(has_up    ? coord_to_boid(s, x, y, z+1) : 1)

        b_smask[b] = Int8(has_west)              |
                    (Int8(has_east)  << Int8(1)) |
                    (Int8(has_south) << Int8(2)) |
                    (Int8(has_north) << Int8(3)) |
                    (Int8(has_down)  << Int8(4)) |
                    (Int8(has_up)    << Int8(5))
    end

    return SmoothingCache3D(nb_,
        b_west, b_east, b_south, b_north, b_down, b_up,
        w_west, w_east, w_south, w_north, w_down, w_up,
        1.0 - c, 1.0 + c,
        b_smask)
end

@kernel function kernel_smooth_dual0_3d_cached!(res,
        @Const(b_west),  @Const(b_east),
        @Const(b_south), @Const(b_north),
        @Const(b_down),  @Const(b_up),
        @Const(w_west),  @Const(w_east),
        @Const(w_south), @Const(w_north),
        @Const(w_down),  @Const(w_up),
        @Const(b_smask),
        @Const(f),
        diag, nbr_sign)
    b = @index(Global)
    @inbounds begin
        mask = b_smask[b]
        z    = zero(eltype(f))
        nbrs =
            ifelse(Bool( mask        & Int8(1)), w_west[b]  * f[b_west[b]],  z) +
            ifelse(Bool((mask >> 1)  & Int8(1)), w_east[b]  * f[b_east[b]],  z) +
            ifelse(Bool((mask >> 2)  & Int8(1)), w_south[b] * f[b_south[b]], z) +
            ifelse(Bool((mask >> 3)  & Int8(1)), w_north[b] * f[b_north[b]], z) +
            ifelse(Bool((mask >> 4)  & Int8(1)), w_down[b]  * f[b_down[b]],  z) +
            ifelse(Bool((mask >> 5)  & Int8(1)), w_up[b]    * f[b_up[b]],    z)
        res[b] = diag * f[b] + nbr_sign * nbrs
    end
end

function _smooth_dual0_pass_3d!(res, cache::SmoothingCache3D, f, diag, sign)
    backend = get_backend(f)
    kernel_smooth_dual0_3d_cached!(backend)(
        res,
        cache.b_west, cache.b_east, cache.b_south, cache.b_north,
        cache.b_down, cache.b_up,
        cache.w_west, cache.w_east, cache.w_south, cache.w_north,
        cache.w_down, cache.w_up,
        cache.b_smask, f, diag, sign; ndrange = cache.nboids_)
    return res
end

function smooth_dual0_fused!(res::AbstractVector{FT}, tmp::AbstractVector{FT},
        cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    _smooth_dual0_pass_3d!(tmp, cache, f,   cache.diag_fwd,  1.0)
    _smooth_dual0_pass_3d!(res, cache, tmp, cache.diag_bwd, -1.0)
    return res
end

function smooth_dual0_fused!(res::AbstractVector{FT},
        cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    tmp = similar(f)
    return smooth_dual0_fused!(res, tmp, cache, f)
end

function smooth_dual0_fused(cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    res = similar(f)
    return smooth_dual0_fused!(res, cache, f)
end
  
# ═══════════════════════════════════════════════════════════════════════════
#  Cached interface functions
# ═══════════════════════════════════════════════════════════════════════════

# NOTE:
# This first pass mirrors the 2D cache strategy and uses Int32 lookup arrays
# for bandwidth efficiency. As 3D problem sizes grow, we may need to make the
# cache parametric over index type and support Int64 lookup arrays.

struct UniformDECCache3D{
    IT <: AbstractVector{Int32},
    MT <: AbstractVector{Int8},
    VT <: AbstractVector,
    ST
}
    nv_       :: Int
    ne_       :: Int
    nquads_   :: Int
    nboids_   :: Int
    nxedges_  :: Int
    nyedges_  :: Int
    nzedges_  :: Int
    nxyquads_ :: Int
    nxzquads_ :: Int
    nyzquads_ :: Int

    # -------------------------------------------------------------------------
    # Exterior-derivative cache
    # -------------------------------------------------------------------------
    src_v :: IT
    tgt_v :: IT

    q_e1  :: IT
    q_e2  :: IT
    q_e3  :: IT
    q_e4  :: IT

    b_q1  :: IT
    b_q2  :: IT
    b_q3  :: IT
    b_q4  :: IT
    b_q5  :: IT
    b_q6  :: IT

    # -------------------------------------------------------------------------
    # Hodge star cache
    # -------------------------------------------------------------------------
    hs0_scale  :: VT
    hs1_scale  :: VT
    hs2_scale  :: VT
    hs3_val    :: ST

    ihs0_scale :: VT
    ihs1_scale :: VT
    ihs2_scale :: VT
    ihs3_val   :: ST

    # -------------------------------------------------------------------------
    # Dual-derivative cache
    # -------------------------------------------------------------------------

    # dd0 : dual 0-form (boids) -> dual 1-form (quads)
    # quad_boids ordering:
    #   Z_ALIGN: (Down, Up)
    #   Y_ALIGN: (South, North)
    #   X_ALIGN: (West, East)
    dd0_bp    :: IT   # positive / higher-side boid
    dd0_bn    :: IT   # negative / lower-side boid
    dd0_qmask :: MT   # bit0 = positive exists, bit1 = negative exists

    # dd1 : dual 1-form (quads) -> dual 2-form (edges)
    # edge_quads ordering matches existing 3D kernel:
    #   res = -q1 + q2 + q3 - q4
    dd1_q1    :: IT
    dd1_q2    :: IT
    dd1_q3    :: IT
    dd1_q4    :: IT
    dd1_emask :: MT   # bits 0:3 indicate which quad slots exist

    # dd2 : dual 2-form (edges) -> dual 3-form (vertices)
    # vertex_edges ordering:
    #   (z-low, z-high, y-south, y-north, x-west, x-east)
    # signs in existing kernel:
    #   +e1 - e2 + e3 - e4 + e5 - e6
    dd2_e1    :: IT
    dd2_e2    :: IT
    dd2_e3    :: IT
    dd2_e4    :: IT
    dd2_e5    :: IT
    dd2_e6    :: IT
    dd2_vmask :: MT   # bits 0:5 indicate which edge slots exist

    # -------------------------------------------------------------------------
    # Wedge 1∧2 cache
    # -------------------------------------------------------------------------

    # boid_edges ordering:
    #   e1:e4   -> X edges
    #   e5:e8   -> Y edges
    #   e9:e12  -> Z edges
    w12_e1  :: IT
    w12_e2  :: IT
    w12_e3  :: IT
    w12_e4  :: IT
    w12_e5  :: IT
    w12_e6  :: IT
    w12_e7  :: IT
    w12_e8  :: IT
    w12_e9  :: IT
    w12_e10 :: IT
    w12_e11 :: IT
    w12_e12 :: IT

    # boid_quads ordering:
    #   q1, q2 = Down, Up
    #   q3, q4 = South, North
    #   q5, q6 = West, East
    w12_q1  :: IT
    w12_q2  :: IT
    w12_q3  :: IT
    w12_q4  :: IT
    w12_q5  :: IT
    w12_q6  :: IT
end

Adapt.@adapt_structure UniformDECCache3D

function UniformDECCache3D(s::UniformCubicalComplex3D{FT}) where {FT <: AbstractFloat}
    nv_       = nv(s)
    ne_       = ne(s)
    nquads_   = nquads(s)
    nboids_   = nboids(s)
    nxedges_  = nxedges(s)
    nyedges_  = nyedges(s)
    nzedges_  = nzedges(s)
    nxyquads_ = nxyquads(s)
    nxzquads_ = nxzquads(s)
    nyzquads_ = nyzquads(s)

    # -------------------------------------------------------------------------
    # Exterior derivative cache
    # -------------------------------------------------------------------------
    src_v = Vector{Int32}(undef, ne_)
    tgt_v = Vector{Int32}(undef, ne_)
    for e in 1:ne_
        src_v[e] = Int32(src(s, e))
        tgt_v[e] = Int32(tgt(s, e))
    end

    q_e1 = Vector{Int32}(undef, nquads_)
    q_e2 = Vector{Int32}(undef, nquads_)
    q_e3 = Vector{Int32}(undef, nquads_)
    q_e4 = Vector{Int32}(undef, nquads_)
    for q in 1:nquads_
        x, y, z, align = quad_to_coord(s, q)
        e1, e2, e3, e4 = quad_edges(s, x, y, z, align)
        q_e1[q] = Int32(e1)
        q_e2[q] = Int32(e2)
        q_e3[q] = Int32(e3)
        q_e4[q] = Int32(e4)
    end

    b_q1 = Vector{Int32}(undef, nboids_)
    b_q2 = Vector{Int32}(undef, nboids_)
    b_q3 = Vector{Int32}(undef, nboids_)
    b_q4 = Vector{Int32}(undef, nboids_)
    b_q5 = Vector{Int32}(undef, nboids_)
    b_q6 = Vector{Int32}(undef, nboids_)
    for b in 1:nboids_
        x, y, z = boid_to_coord(s, b)
        q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)
        b_q1[b] = Int32(q1)
        b_q2[b] = Int32(q2)
        b_q3[b] = Int32(q3)
        b_q4[b] = Int32(q4)
        b_q5[b] = Int32(q5)
        b_q6[b] = Int32(q6)
    end

    # -------------------------------------------------------------------------
    # Wedge 1∧2 cache
    # -------------------------------------------------------------------------
    w12_e1  = Vector{Int32}(undef, nboids_)
    w12_e2  = Vector{Int32}(undef, nboids_)
    w12_e3  = Vector{Int32}(undef, nboids_)
    w12_e4  = Vector{Int32}(undef, nboids_)
    w12_e5  = Vector{Int32}(undef, nboids_)
    w12_e6  = Vector{Int32}(undef, nboids_)
    w12_e7  = Vector{Int32}(undef, nboids_)
    w12_e8  = Vector{Int32}(undef, nboids_)
    w12_e9  = Vector{Int32}(undef, nboids_)
    w12_e10 = Vector{Int32}(undef, nboids_)
    w12_e11 = Vector{Int32}(undef, nboids_)
    w12_e12 = Vector{Int32}(undef, nboids_)

    w12_q1  = Vector{Int32}(undef, nboids_)
    w12_q2  = Vector{Int32}(undef, nboids_)
    w12_q3  = Vector{Int32}(undef, nboids_)
    w12_q4  = Vector{Int32}(undef, nboids_)
    w12_q5  = Vector{Int32}(undef, nboids_)
    w12_q6  = Vector{Int32}(undef, nboids_)

    for b in 1:nboids_
        x, y, z = boid_to_coord(s, b)

        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12 = boid_edges(s, x, y, z)
        q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)

        w12_e1[b]  = Int32(e1)
        w12_e2[b]  = Int32(e2)
        w12_e3[b]  = Int32(e3)
        w12_e4[b]  = Int32(e4)
        w12_e5[b]  = Int32(e5)
        w12_e6[b]  = Int32(e6)
        w12_e7[b]  = Int32(e7)
        w12_e8[b]  = Int32(e8)
        w12_e9[b]  = Int32(e9)
        w12_e10[b] = Int32(e10)
        w12_e11[b] = Int32(e11)
        w12_e12[b] = Int32(e12)

        w12_q1[b]  = Int32(q1)
        w12_q2[b]  = Int32(q2)
        w12_q3[b]  = Int32(q3)
        w12_q4[b]  = Int32(q4)
        w12_q5[b]  = Int32(q5)
        w12_q6[b]  = Int32(q6)
    end

    # -------------------------------------------------------------------------
    # Hodge star cache
    # -------------------------------------------------------------------------
    hs0_scale  = Vector{FT}(undef, nv_)
    ihs0_scale = Vector{FT}(undef, nv_)
    for v in 1:nv_
        x, y, z = vert_to_coord(s, v)
        dvol = dual_boid_volume(s, x, y, z)
        hs0_scale[v]  = dvol
        ihs0_scale[v] = inv(dvol)
    end

    hs1_scale  = Vector{FT}(undef, ne_)
    ihs1_scale = Vector{FT}(undef, ne_)
    for e in 1:ne_
        x, y, z, align = edge_to_coord(s, e)
        ratio = dual_quad_area(s, x, y, z, align) / edge_len(s, align)
        hs1_scale[e]  = ratio
        ihs1_scale[e] = inv(ratio)
    end

    hs2_scale  = Vector{FT}(undef, nquads_)
    ihs2_scale = Vector{FT}(undef, nquads_)
    for q in 1:nquads_
        x, y, z, align = quad_to_coord(s, q)
        ratio = dual_edge_len(s, x, y, z, align) / quad_area(s, align)
        hs2_scale[q]  = ratio
        ihs2_scale[q] = inv(ratio)
    end

    hs3_val  = inv(boid_volume(s))
    ihs3_val = boid_volume(s)

    # -------------------------------------------------------------------------
    # Dual derivative cache
    # -------------------------------------------------------------------------

    # dd0 : quad -> adjacent boids with 2-bit mask
    dd0_bp    = Vector{Int32}(undef, nquads_)
    dd0_bn    = Vector{Int32}(undef, nquads_)
    dd0_qmask = Vector{Int8}(undef, nquads_)

    DUMMY_BOID = Int32(1)

    for q in 1:nquads_
        x, y, z, align = quad_to_coord(s, q)
        (b_idx, b_valid) = quad_boids(s, x, y, z, align)

        has_neg = b_valid[1]
        has_pos = b_valid[2]

        dd0_bn[q] = has_neg ? Int32(b_idx[1]) : DUMMY_BOID
        dd0_bp[q] = has_pos ? Int32(b_idx[2]) : DUMMY_BOID
        dd0_qmask[q] = Int8(has_pos) | (Int8(has_neg) << 1)
    end

    # dd1 : edge -> adjacent quads with 4-bit mask
    dd1_q1    = Vector{Int32}(undef, ne_)
    dd1_q2    = Vector{Int32}(undef, ne_)
    dd1_q3    = Vector{Int32}(undef, ne_)
    dd1_q4    = Vector{Int32}(undef, ne_)
    dd1_emask = Vector{Int8}(undef, ne_)

    DUMMY_QUAD = Int32(1)

    for e in 1:ne_
        x, y, z, align = edge_to_coord(s, e)
        (q_idx, q_valid) = edge_quads(s, x, y, z, align)

        dd1_q1[e] = q_valid[1] ? Int32(q_idx[1]) : DUMMY_QUAD
        dd1_q2[e] = q_valid[2] ? Int32(q_idx[2]) : DUMMY_QUAD
        dd1_q3[e] = q_valid[3] ? Int32(q_idx[3]) : DUMMY_QUAD
        dd1_q4[e] = q_valid[4] ? Int32(q_idx[4]) : DUMMY_QUAD

        dd1_emask[e] =
            Int8(q_valid[1]) |
            (Int8(q_valid[2]) << 1) |
            (Int8(q_valid[3]) << 2) |
            (Int8(q_valid[4]) << 3)
    end

    # dd2 : vertex -> incident edges with 6-bit mask
    dd2_e1    = Vector{Int32}(undef, nv_)
    dd2_e2    = Vector{Int32}(undef, nv_)
    dd2_e3    = Vector{Int32}(undef, nv_)
    dd2_e4    = Vector{Int32}(undef, nv_)
    dd2_e5    = Vector{Int32}(undef, nv_)
    dd2_e6    = Vector{Int32}(undef, nv_)
    dd2_vmask = Vector{Int8}(undef, nv_)

    DUMMY_EDGE = Int32(1)

    for v in 1:nv_
        x, y, z = vert_to_coord(s, v)
        (e_idx, e_valid) = vertex_edges(s, x, y, z)

        dd2_e1[v] = e_valid[1] ? Int32(e_idx[1]) : DUMMY_EDGE
        dd2_e2[v] = e_valid[2] ? Int32(e_idx[2]) : DUMMY_EDGE
        dd2_e3[v] = e_valid[3] ? Int32(e_idx[3]) : DUMMY_EDGE
        dd2_e4[v] = e_valid[4] ? Int32(e_idx[4]) : DUMMY_EDGE
        dd2_e5[v] = e_valid[5] ? Int32(e_idx[5]) : DUMMY_EDGE
        dd2_e6[v] = e_valid[6] ? Int32(e_idx[6]) : DUMMY_EDGE

        dd2_vmask[v] =
            Int8(e_valid[1]) |
            (Int8(e_valid[2]) << 1) |
            (Int8(e_valid[3]) << 2) |
            (Int8(e_valid[4]) << 3) |
            (Int8(e_valid[5]) << 4) |
            (Int8(e_valid[6]) << 5)
    end

    return UniformDECCache3D(
        nv_, ne_, nquads_, nboids_,
        nxedges_, nyedges_, nzedges_,
        nxyquads_, nxzquads_, nyzquads_,

        src_v, tgt_v,
        q_e1, q_e2, q_e3, q_e4,
        b_q1, b_q2, b_q3, b_q4, b_q5, b_q6,

        hs0_scale, hs1_scale, hs2_scale, hs3_val,
        ihs0_scale, ihs1_scale, ihs2_scale, ihs3_val,

        dd0_bp, dd0_bn, dd0_qmask,
        dd1_q1, dd1_q2, dd1_q3, dd1_q4, dd1_emask,
        dd2_e1, dd2_e2, dd2_e3, dd2_e4, dd2_e5, dd2_e6, dd2_vmask,

        w12_e1, w12_e2, w12_e3, w12_e4, w12_e5, w12_e6, w12_e7, w12_e8, w12_e9, w12_e10, w12_e11, w12_e12,
        w12_q1, w12_q2, w12_q3, w12_q4, w12_q5, w12_q6,
    )
end

# ============================================================================
# Cached exterior derivative kernels
# ============================================================================

@kernel function kernel_d0_cached_3d!(res, @Const(src_v), @Const(tgt_v), @Const(f))
    e = @index(Global)
    @inbounds res[e] = f[tgt_v[e]] - f[src_v[e]]
end

@kernel function kernel_d1_cached_3d!(
    res,
    @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
    @Const(f),
)
    q = @index(Global)
    @inbounds res[q] = f[q_e1[q]] + f[q_e2[q]] - f[q_e3[q]] - f[q_e4[q]]
end

@kernel function kernel_d2_cached_3d!(
    res,
    @Const(b_q1), @Const(b_q2), @Const(b_q3),
    @Const(b_q4), @Const(b_q5), @Const(b_q6),
    @Const(f),
)
    b = @index(Global)
    @inbounds res[b] = f[b_q2[b]] - f[b_q1[b]] +
                       f[b_q4[b]] - f[b_q3[b]] +
                       f[b_q6[b]] - f[b_q5[b]]
end

# ============================================================================
# Cached Hodge kernels
# ============================================================================

@kernel function kernel_hodge_vec_3d!(res, @Const(scale), @Const(f))
    i = @index(Global)
    @inbounds res[i] = scale[i] * f[i]
end

@kernel function kernel_hodge_scalar_3d!(res, val, @Const(f))
    i = @index(Global)
    @inbounds res[i] = val * f[i]
end

# ============================================================================
# Cached exterior derivative interface
# ============================================================================

function exterior_derivative!(
    res::AbstractVector{FT},
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_d0_cached_3d!(backend, workgroup_size)
    kernel(res, cache.src_v, cache.tgt_v, f, ndrange = cache.ne_)
    return res
end

function exterior_derivative!(
    res::AbstractVector{FT},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_d1_cached_3d!(backend, workgroup_size)
    kernel(res, cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4, f, ndrange = cache.nquads_)
    return res
end

function exterior_derivative!(
    res::AbstractVector{FT},
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_d2_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.b_q1, cache.b_q2, cache.b_q3,
        cache.b_q4, cache.b_q5, cache.b_q6,
        f,
        ndrange = cache.nboids_,
    )
    return res
end

function exterior_derivative(
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.ne_)
    return exterior_derivative!(res, Val(0), cache, f; workgroup_size = workgroup_size)
end

function exterior_derivative(
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return exterior_derivative!(res, Val(1), cache, f; workgroup_size = workgroup_size)
end

function exterior_derivative(
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    return exterior_derivative!(res, Val(2), cache, f; workgroup_size = workgroup_size)
end

# ============================================================================
# Cached hodge_star interface
# ============================================================================

function hodge_star!(
    res::AbstractVector{FT},
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.hs0_scale, f, ndrange = cache.nv_)
    return res
end

function hodge_star!(
    res::AbstractVector{FT},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.hs1_scale, f, ndrange = cache.ne_)
    return res
end

function hodge_star!(
    res::AbstractVector{FT},
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.hs2_scale, f, ndrange = cache.nquads_)
    return res
end

function hodge_star!(
    res::AbstractVector{FT},
    ::Val{3},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_scalar_3d!(backend, workgroup_size)
    kernel(res, cache.hs3_val, f, ndrange = cache.nboids_)
    return res
end

function hodge_star(
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nv_)
    return hodge_star!(res, Val(0), cache, f; workgroup_size = workgroup_size)
end

function hodge_star(
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.ne_)
    return hodge_star!(res, Val(1), cache, f; workgroup_size = workgroup_size)
end

function hodge_star(
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return hodge_star!(res, Val(2), cache, f; workgroup_size = workgroup_size)
end

function hodge_star(
    ::Val{3},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    return hodge_star!(res, Val(3), cache, f; workgroup_size = workgroup_size)
end

# ============================================================================
# Cached inv_hodge_star interface
# ============================================================================

function inv_hodge_star!(
    res::AbstractVector{FT},
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.ihs0_scale, f, ndrange = cache.nv_)
    return res
end

function inv_hodge_star!(
    res::AbstractVector{FT},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.ihs1_scale, f, ndrange = cache.ne_)
    return res
end

function inv_hodge_star!(
    res::AbstractVector{FT},
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_vec_3d!(backend, workgroup_size)
    kernel(res, cache.ihs2_scale, f, ndrange = cache.nquads_)
    return res
end

function inv_hodge_star!(
    res::AbstractVector{FT},
    ::Val{3},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_hodge_scalar_3d!(backend, workgroup_size)
    kernel(res, cache.ihs3_val, f, ndrange = cache.nboids_)
    return res
end

function inv_hodge_star(
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nv_)
    return inv_hodge_star!(res, Val(0), cache, f; workgroup_size = workgroup_size)
end

function inv_hodge_star(
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.ne_)
    return inv_hodge_star!(res, Val(1), cache, f; workgroup_size = workgroup_size)
end

function inv_hodge_star(
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return inv_hodge_star!(res, Val(2), cache, f; workgroup_size = workgroup_size)
end

function inv_hodge_star(
    ::Val{3},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    return inv_hodge_star!(res, Val(3), cache, f; workgroup_size = workgroup_size)
end

# ============================================================================
# Cached dual derivative kernels
# ============================================================================

# dd0 : dual 0-form (boids) -> dual 1-form (quads)
# quad_boids ordering:
#   q stores (lower, higher) boids according to alignment
# signs match uncached kernel: higher - lower
@kernel function kernel_dd0_cached_3d!(
    res,
    @Const(dd0_bp), @Const(dd0_bn), @Const(dd0_qmask),
    @Const(f),
)
    q = @index(Global)
    @inbounds begin
        mask = dd0_qmask[q]
        z = zero(eltype(f))

        pos = ifelse(Bool(mask & Int8(1)),              f[dd0_bp[q]], z)
        neg = ifelse(Bool((mask >> Int8(1)) & Int8(1)), f[dd0_bn[q]], z)

        res[q] = pos - neg
    end
end

# dd1 : dual 1-form (quads) -> dual 2-form (edges)
# edge_quads ordering from mesh:
#   q1, q2, q3, q4
# signs match uncached kernel: -q1 + q2 + q3 - q4
@kernel function kernel_dd1_cached_3d!(
    res,
    @Const(dd1_q1), @Const(dd1_q2), @Const(dd1_q3), @Const(dd1_q4),
    @Const(dd1_emask),
    @Const(f),
)
    e = @index(Global)
    @inbounds begin
        mask = dd1_emask[e]
        z = zero(eltype(f))

        v1 = ifelse(Bool(mask & Int8(1)),              f[dd1_q1[e]], z)
        v2 = ifelse(Bool((mask >> Int8(1)) & Int8(1)), f[dd1_q2[e]], z)
        v3 = ifelse(Bool((mask >> Int8(2)) & Int8(1)), f[dd1_q3[e]], z)
        v4 = ifelse(Bool((mask >> Int8(3)) & Int8(1)), f[dd1_q4[e]], z)

        res[e] = -v1 + v2 + v3 - v4
    end
end

# dd2 : dual 2-form (edges) -> dual 3-form (vertices)
# vertex_edges ordering from mesh:
#   (z-low, z-high, y-south, y-north, x-west, x-east)
# signs match uncached kernel:
#   +e1 - e2 + e3 - e4 + e5 - e6
@kernel function kernel_dd2_cached_3d!(
    res,
    @Const(dd2_e1), @Const(dd2_e2), @Const(dd2_e3),
    @Const(dd2_e4), @Const(dd2_e5), @Const(dd2_e6),
    @Const(dd2_vmask),
    @Const(f),
)
    v = @index(Global)
    @inbounds begin
        mask = dd2_vmask[v]
        z = zero(eltype(f))

        a1 = ifelse(Bool(mask & Int8(1)),              f[dd2_e1[v]], z)
        a2 = ifelse(Bool((mask >> Int8(1)) & Int8(1)), f[dd2_e2[v]], z)
        a3 = ifelse(Bool((mask >> Int8(2)) & Int8(1)), f[dd2_e3[v]], z)
        a4 = ifelse(Bool((mask >> Int8(3)) & Int8(1)), f[dd2_e4[v]], z)
        a5 = ifelse(Bool((mask >> Int8(4)) & Int8(1)), f[dd2_e5[v]], z)
        a6 = ifelse(Bool((mask >> Int8(5)) & Int8(1)), f[dd2_e6[v]], z)

        res[v] = a1 - a2 + a3 - a4 + a5 - a6
    end
end

# ============================================================================
# Cached dual_derivative interface
# ============================================================================

function dual_derivative!(
    res::AbstractVector{FT},
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_dd0_cached_3d!(backend, workgroup_size)
    kernel(res, cache.dd0_bp, cache.dd0_bn, cache.dd0_qmask, f, ndrange = cache.nquads_)
    return res
end

function dual_derivative!(
    res::AbstractVector{FT},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_dd1_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.dd1_q1, cache.dd1_q2, cache.dd1_q3, cache.dd1_q4,
        cache.dd1_emask,
        f,
        ndrange = cache.ne_,
    )
    return res
end

function dual_derivative!(
    res::AbstractVector{FT},
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_dd2_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.dd2_e1, cache.dd2_e2, cache.dd2_e3,
        cache.dd2_e4, cache.dd2_e5, cache.dd2_e6,
        cache.dd2_vmask,
        f,
        ndrange = cache.nv_,
    )
    return res
end

function dual_derivative(
    ::Val{0},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return dual_derivative!(res, Val(0), cache, f; workgroup_size = workgroup_size)
end

function dual_derivative(
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.ne_)
    return dual_derivative!(res, Val(1), cache, f; workgroup_size = workgroup_size)
end

function dual_derivative(
    ::Val{2},
    cache::UniformDECCache3D,
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nv_)
    return dual_derivative!(res, Val(2), cache, f; workgroup_size = workgroup_size)
end

# ============================================================================
# Cached wedge product kernels
# ============================================================================

@kernel function kernel_wedge_11_cached_3d!(
    res,
    @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
    @Const(a), @Const(b),
)
    q = @index(Global)
    @inbounds begin
        a1 = (a[q_e1[q]] + a[q_e3[q]]) * 0.5
        a2 = (a[q_e2[q]] + a[q_e4[q]]) * 0.5
        b1 = (b[q_e1[q]] + b[q_e3[q]]) * 0.5
        b2 = (b[q_e2[q]] + b[q_e4[q]]) * 0.5
        res[q] = a1 * b2 - a2 * b1
    end
end

@kernel function kernel_wedge_12_cached_3d!(
    res,
    @Const(w12_e1),  @Const(w12_e2),  @Const(w12_e3),  @Const(w12_e4),
    @Const(w12_e5),  @Const(w12_e6),  @Const(w12_e7),  @Const(w12_e8),
    @Const(w12_e9),  @Const(w12_e10), @Const(w12_e11), @Const(w12_e12),
    @Const(w12_q1),  @Const(w12_q2),  @Const(w12_q3),
    @Const(w12_q4),  @Const(w12_q5),  @Const(w12_q6),
    @Const(a), @Const(b),
)
    idx = @index(Global)
    @inbounds begin
        ax = 0.25 * (a[w12_e1[idx]] + a[w12_e2[idx]] + a[w12_e3[idx]] + a[w12_e4[idx]])
        ay = 0.25 * (a[w12_e5[idx]] + a[w12_e6[idx]] + a[w12_e7[idx]] + a[w12_e8[idx]])
        az = 0.25 * (a[w12_e9[idx]] + a[w12_e10[idx]] + a[w12_e11[idx]] + a[w12_e12[idx]])

        bz = 0.5 * (b[w12_q1[idx]] + b[w12_q2[idx]])
        by = 0.5 * (b[w12_q3[idx]] + b[w12_q4[idx]])
        bx = 0.5 * (b[w12_q5[idx]] + b[w12_q6[idx]])

        res[idx] = ax * bx - ay * by + az * bz
    end
end

@kernel function kernel_wedge_dd_01_cached_3d!(
    res,
    @Const(dd0_bp), @Const(dd0_bn), @Const(dd0_qmask),
    @Const(f), @Const(a),
)
    q = @index(Global)
    @inbounds begin
        mask = dd0_qmask[q]
        z = zero(eltype(f))

        pos = ifelse(Bool(mask & Int8(1)),               f[dd0_bp[q]], z)
        neg = ifelse(Bool((mask >> Int8(1)) & Int8(1)), f[dd0_bn[q]], z)

        nvalid = (mask & Int8(1)) + ((mask >> Int8(1)) & Int8(1))
        avg = ifelse(
            nvalid == Int8(2), (pos + neg) * 0.5,
            ifelse(nvalid == Int8(1), pos + neg, z),
        )

        res[q] = avg * a[q]
    end
end

# ============================================================================
# Cached wedge product interface
# ============================================================================

function wedge_product!(
    res::AbstractVector{FT},
    ::Val{1},
    ::Val{1},
    cache::UniformDECCache3D,
    a::AbstractVector{FT},
    b::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_wedge_11_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
        a, b,
        ndrange = cache.nquads_,
    )
    return res
end

function wedge_product(
    ::Val{1},
    ::Val{1},
    cache::UniformDECCache3D,
    a::AbstractVector{FT},
    b::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return wedge_product!(res, Val(1), Val(1), cache, a, b; workgroup_size = workgroup_size)
end

function wedge_product!(
    res::AbstractVector{FT},
    ::Val{1},
    ::Val{2},
    cache::UniformDECCache3D,
    a::AbstractVector{FT},
    b::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_wedge_12_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.w12_e1,  cache.w12_e2,  cache.w12_e3,  cache.w12_e4,
        cache.w12_e5,  cache.w12_e6,  cache.w12_e7,  cache.w12_e8,
        cache.w12_e9,  cache.w12_e10, cache.w12_e11, cache.w12_e12,
        cache.w12_q1,  cache.w12_q2,  cache.w12_q3,
        cache.w12_q4,  cache.w12_q5,  cache.w12_q6,
        a, b,
        ndrange = cache.nboids_,
    )
    return res
end

function wedge_product(
    ::Val{1},
    ::Val{2},
    cache::UniformDECCache3D,
    a::AbstractVector{FT},
    b::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    return wedge_product!(res, Val(1), Val(2), cache, a, b; workgroup_size = workgroup_size)
end

wedge_product(
    ::Val{2},
    ::Val{1},
    cache::UniformDECCache3D,
    b::AbstractVector{FT},
    a::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat} =
    wedge_product(Val(1), Val(2), cache, a, b; workgroup_size = workgroup_size)

function wedge_product_dd!(
    res::AbstractVector{FT},
    ::Val{0},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT},
    a::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(res)
    kernel = kernel_wedge_dd_01_cached_3d!(backend, workgroup_size)
    kernel(
        res,
        cache.dd0_bp, cache.dd0_bn, cache.dd0_qmask,
        f, a,
        ndrange = cache.nquads_,
    )
    return res
end

function wedge_product_dd(
    ::Val{0},
    ::Val{1},
    cache::UniformDECCache3D,
    f::AbstractVector{FT},
    a::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat}
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return wedge_product_dd!(res, Val(0), Val(1), cache, f, a; workgroup_size = workgroup_size)
end

wedge_product_dd(
    ::Val{1},
    ::Val{0},
    cache::UniformDECCache3D,
    a::AbstractVector{FT},
    f::AbstractVector{FT};
    workgroup_size::Int = 256,
) where {FT <: AbstractFloat} =
    wedge_product_dd(Val(0), Val(1), cache, f, a; workgroup_size = workgroup_size)