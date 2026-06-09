using KernelAbstractions

### Exterior Derivatives ###

@kernel function kernel_exterior_derivative_zero!(res, s, @Const(f))
    idx = @index(Global)
    x, y, z, align = edge_to_coord(s, idx)
  
    @inbounds res[idx] = f[tgt(s, x, y, z, align)] - f[src(s, x, y, z, align)]
end
  
@kernel function kernel_exterior_derivative_one!(res, s, @Const(f))
    idx = @index(Global)
    x, y, z, align = quad_to_coord(s, idx)
  
    e1, e2, e3, e4 = quad_edges(s, x, y, z, align)
    @inbounds res[idx] = f[e1] + f[e2] - f[e3] - f[e4]
end
  
@kernel function kernel_exterior_derivative_two!(res, s, @Const(f))
    idx = @index(Global)
    x, y, z = boid_to_coord(s, idx)
    q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)
    @inbounds res[idx] = f[q1] - f[q2] + f[q3] - f[q4] + f[q5] - f[q6]
end

# TODO: Change all these to get_backend

function exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_zero!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_one!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative!(res, ::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_exterior_derivative_two!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function exterior_derivative(op::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function exterior_derivative(op::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function exterior_derivative(op::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Hodge Star ###

@kernel function kernel_hodge_star_zero!(res, s, @Const(f))
    idx = @index(Global) 
    v_idx = idx
    x, y, z = vert_to_coord(s, v_idx)

    ratio = dual_boid_volume(s, x, y, z)
    @inbounds res[idx] = f[v_idx] * ratio
end

@kernel function kernel_hodge_star_one!(res, s, @Const(f))
    idx = @index(Global)
    e_idx = idx
    x, y, z, align = edge_to_coord(s, e_idx)

    ratio = dual_quad_area(s, x, y, z, align) / edge_len(s, align)
    @inbounds res[idx] = f[e_idx] * ratio
end

@kernel function kernel_hodge_star_two!(res, s, @Const(f))
    idx = @index(Global) 
    q_idx = idx 
    x, y, z, align = quad_to_coord(s, q_idx)

    ratio = dual_edge_length(s, x, y, z, align) / quad_area(s, align)
    @inbounds res[idx] = f[q_idx] * ratio
end

@kernel function kernel_hodge_star_three!(res, s, @Const(f))
    idx = @index(Global) 
    b_idx = idx 
    x, y, z = boid_to_coord(s, b_idx)
    
    ratio = 1.0 / boid_volume(s)
    @inbounds res[idx] = f[b_idx] * ratio
end

# TODO: Change all these to get_backend

function hodge_star!(res, ::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_hodge_star_zero!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res, ::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_hodge_star_one!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res, ::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_hodge_star_two!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star!(res, ::Val{3}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_hodge_star_three!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Inverse Hodge Star ###

@kernel function kernel_inv_hodge_star_zero!(res, s, @Const(f))
    idx = @index(Global)
    b_idx = idx
    x, y, z = vert_to_coord(s, idx)

    ratio = 1.0 / dual_boid_volume(s, x, y, z)
    @inbounds res[idx] = f[b_idx] * ratio
end

@kernel function kernel_inv_hodge_star_one!(res, s, @Const(f))
    idx = @index(Global)
    q_idx = idx
    x, y, z, align = edge_to_coord(s, idx)

    ratio = edge_len(s, align) / dual_quad_area(s, x, y, z, align)
    @inbounds res[idx] = f[q_idx] * ratio
end

@kernel function kernel_inv_hodge_star_two!(res, s, @Const(f))
    idx = @index(Global)
    e_idx = idx
    x, y, z, align = quad_to_coord(s, idx)
    
    ratio = quad_area(s, align) / dual_edge_length(s, x, y, z, align)
    @inbounds res[idx] = f[e_idx] * ratio
end

@kernel function kernel_inv_hodge_star_three!(res, s, @Const(f))
    idx = @index(Global)
    v_idx = idx
    
    ratio = boid_volume(s)
    @inbounds res[idx] = f[v_idx] * ratio
end

# TODO: Change all these to get_backend

function inv_hodge_star!(res, ::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_zero!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res, ::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_one!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res, ::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_two!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star!(res, ::Val{3}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_inv_hodge_star_three!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function inv_hodge_star(op::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function inv_hodge_star(op::Val{3}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nboids(s))
    inv_hodge_star!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Dual Derivative ###

@kernel function kernel_dual_exterior_derivative_zero!(res, s, @Const(f))
    idx = @index(Global)
    FT = eltype(f)
    x, y, z, align = quad_to_coord(s, idx)
    (b_indices, b_valid) = quad_boids(s, x, y, z, align)
    
    val1 = b_valid[1] ? f[b_indices[1]] : zero(FT)
    val2 = b_valid[2] ? f[b_indices[2]] : zero(FT)

    @inbounds res[idx] = val2 - val1
end
  
@kernel function kernel_dual_exterior_derivative_one!(res, s, @Const(f))
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
  
@kernel function kernel_dual_exterior_derivative_two!(res, s, @Const(f))
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

# TODO: Change all these to get_backend

function dual_exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_dual_exterior_derivative_zero!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function dual_exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_dual_exterior_derivative_one!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function dual_exterior_derivative!(res, ::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_dual_exterior_derivative_two!(backend, workgroup_size)
    kernel(res, s, f, ndrange=size(res))
end

function dual_exterior_derivative(op::Val{0}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    dual_exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function dual_exterior_derivative(op::Val{1}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), ne(s))
    dual_exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

function dual_exterior_derivative(op::Val{2}, s::UniformCubicalComplex3D, f; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nv(s))
    dual_exterior_derivative!(res, op, s, f; workgroup_size=workgroup_size)
    return res
end

### Wedge Product ###

@kernel function kernel_wedge_product_11!(res, s, @Const(a), @Const(b))
    idx = @index(Global)
    x, y, z, align = quad_to_coord(s, idx)
    e1, e2, e3, e4 = quad_edges(s, x, y, z, align)
    
    # X_ALIGN first pair Y, second pair Z
    # Y_ALIGN first pair Z, second pair X (negate)
    # Z_ALIGN first pair X, second pair Y 
    @inbounds begin
        a1 = 0.5 * (a[e1] + a[e3]); a2 = 0.5 * (a[e2] + a[e4])
        b1 = 0.5 * (b[e1] + b[e3]); b2 = 0.5 * (b[e2] + b[e4])
        tmp = a1 * b2 - a2 * b1
        res[idx] = ifelse(align == Y_ALIGN, -tmp, tmp)
    end
end

@kernel function kernel_wedge_product_12!(res, s, @Const(a), @Const(b))
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

function wedge_product!(res, ::Val{1}, ::Val{1}, s::UniformCubicalComplex3D, a, b; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_wedge_product_11!(backend, workgroup_size)
    kernel(res, s, a, b, ndrange=size(res))
end

function wedge_product!(res, ::Val{1}, ::Val{2}, s::UniformCubicalComplex3D, a, b; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_wedge_product_12!(backend, workgroup_size)
    kernel(res, s, a, b, ndrange=size(res))
end

function wedge_product(op1::Val{1}, op2::Val{1}, s::UniformCubicalComplex3D, a, b; workgroup_size = 256)
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, eltype(a), nquads(s))
    wedge_product!(res, op1, op2, s, a, b; workgroup_size=workgroup_size)
    return res
end

function wedge_product(op1::Val{1}, op2::Val{2}, s::UniformCubicalComplex3D, a, b; workgroup_size = 256)
    backend = get_backend(a)
    res = KernelAbstractions.zeros(backend, eltype(a), nboids(s))
    wedge_product!(res, op1, op2, s, a, b; workgroup_size=workgroup_size)
    return res
end
wedge_product(::Val{2}, ::Val{1}, s::UniformCubicalComplex3D, a, b; workgroup_size = 256) = wedge_product(Val(1), Val(2), s, a, b; workgroup_size=workgroup_size)

### Dual Wedge Product ###

@kernel function kernel_wedge_product_dd_01!(res, s, @Const(f), @Const(a))
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

# TODO: Change all these to get_backend

function wedge_product_dd!(res, ::Val{0}, ::Val{1}, s::UniformCubicalComplex3D, f, a; workgroup_size = 256)
    backend = get_backend(res)
    kernel = kernel_wedge_product_dd_01!(backend, workgroup_size)
    kernel(res, s, f, a, ndrange=size(res))
end

function wedge_product_dd(op1::Val{0}, op2::Val{1}, s::UniformCubicalComplex3D, f, a; workgroup_size = 256)
    backend = get_backend(f)
    res = KernelAbstractions.zeros(backend, eltype(f), nquads(s))
    wedge_product_dd!(res, op1, op2, s, f, a; workgroup_size=workgroup_size)
    return res
end

### Sharp and Flat Operators ###

# TODO: Fill out this function
@kernel function kernel_sharp_dd(X, Y, Z, s, @Const(f))
    idx = @index(Global)
    x, y, z = boid_to_coord(s, idx)
    FT = eltype(f)

    q1, q2, q3, q4, q5, q6 = boid_quads(s, x, y, z)



end