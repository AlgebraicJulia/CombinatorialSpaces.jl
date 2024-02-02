module FastDEC
using LinearAlgebra: Diagonal, dot, norm, cross
using StaticArrays: SVector
using SparseArrays: sparse
using LinearAlgebra
using Base.Iterators
using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra.CSets
using ..SimplicialSets, ..DiscreteExteriorCalculus

export dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star, dec_wedge_product, dec_c_wedge_product, dec_p_wedge_product, dec_c_wedge_product!

"""
    dec_p_wedge_product(::Type{Tuple{0,1}}, sd::EmbeddedDeltaDualComplex1D)

Precomputes values for the wedge product between a 0 and 1-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{0,1}}, sd::EmbeddedDeltaDualComplex1D)
    return (hcat(convert(Vector{Int32}, sd[:∂v0])::Vector{Int32}, convert(Vector{Int32}, sd[:∂v1])::Vector{Int32}), simplices(1, sd))
end

"""
    dec_p_wedge_product(::Type{Tuple{0,1}}, sd::EmbeddedDeltaDualComplex2D)

Precomputes values for the wedge product between a 0 and 1-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{0,1}}, sd::EmbeddedDeltaDualComplex2D)
    return (hcat(convert(Vector{Int32}, sd[:∂v0])::Vector{Int32}, convert(Vector{Int32}, sd[:∂v1])::Vector{Int32}), simplices(1, sd))
end

# XXX: This assumes that the dual vertice on an edge is always the midpoint
# TODO: Add options to change 0.5 to a different float
"""
    dec_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack)

Computes the wedge product between a 0 and 1-form.
Use the precomputational "p" varient for the wedge_terms parameter. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack)
    primal_vertices, simples = val_pack

    @inbounds for i in simples
        wedge_terms[i] = 0.5 * α[i] * (f[primal_vertices[i, 1]] + f[primal_vertices[i, 2]])
    end

    return wedge_terms
end

"""
    dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

Precomputes values for the wedge product between a 0 and 2-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

    simples = triangles(sd)

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_0 = @view sd[:D_∂v0]

    dual_edges_2 = @view sd[:D_∂e2]
    dual_v_1 = @view sd[:D_∂v1]

    dv = @view sd[:dual_area]
    vols = @view sd[:area]

    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    primal_vertices = Array{Int32}(undef, 6, ntriangles(sd))
    coeffs = Array{float_type}(undef, 6, ntriangles(sd))

    row_idx_in_col = ones(Int8, ntriangles(sd))
    shift::Int = nv(sd) + ne(sd)

    @inbounds for dual_tri in eachindex(dual_edges_1)
        primal_tri = dual_v_0[dual_edges_1[dual_tri]] - shift
        row_idx = row_idx_in_col[primal_tri]

        primal_vertices[row_idx, primal_tri] = dual_v_1[dual_edges_2[dual_tri]]
        coeffs[row_idx, primal_tri] = dv[dual_tri] / vols[primal_tri]

        row_idx_in_col[primal_tri] += 1
    end

    return (primal_vertices, coeffs, simples)
end

"""
    dec_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack)

Computes the wedge product between a 0 and 2-form.
Use the precomputational "p" varient for the wedge_terms parameter. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack)
    pv, coeffs, simples = val_pack
    @inbounds for i in simples
        wedge_terms[i] = α[i] * (coeffs[1, i] * f[pv[1, i]] + coeffs[2, i] * f[pv[2, i]]
                                 + coeffs[3, i] * f[pv[3, i]] + coeffs[4, i] * f[pv[4, i]]
                                 + coeffs[5, i] * f[pv[5, i]] + coeffs[6, i] * f[pv[6, i]])
    end

    return wedge_terms
end

"""
    dec_p_wedge_product(::Type{Tuple{1,1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

Precomputes values for the wedge product between a 1 and 1-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{1,1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    simples = simplices(2, sd)

    areas = @view sd[:area]
    d_areas = @view sd[:dual_area]

    coeffs = Array{float_type}(undef, 3, ntriangles(sd))

    shift = ntriangles(sd)
    @inbounds for i in 1:ntriangles(sd)
        area = areas[i]
        coeffs[1, i] = (d_areas[i] + d_areas[i+shift]) / area
        coeffs[2, i] = (d_areas[i+2*shift] + d_areas[i+3*shift]) / area
        coeffs[3, i] = (d_areas[i+4*shift] + d_areas[i+5*shift]) / area
    end
    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    e = Array{Int32}(undef, 3, ntriangles(sd))

    e[1, :] = ∂(2, 0, sd)
    e[2, :] = ∂(2, 1, sd)
    e[3, :] = ∂(2, 2, sd)

    return (e, coeffs, simples)
end

"""
    dec_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, f, α, val_pack)

Computes the wedge product between a 1 and 1-form.
Use the precomputational "p" varient for the wedge_terms parameter. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, α, β, val_pack)
    e, coeffs, simples = val_pack

    @inbounds for i in simples
        ae0, ae1, ae2 = α[e[1, i]], α[e[2, i]], α[e[3, i]]
        be0, be1, be2 = β[e[1, i]], β[e[2, i]], β[e[3, i]]

        c1, c2, c3 = coeffs[1, i], coeffs[2, i], coeffs[3, i]

        wedge_terms[i] = (c1 * (ae2 * be1 - ae1 * be2)
                        + c2 * (ae2 * be0 - ae0 * be2)
                        + c3 * (ae1 * be0 - ae0 * be1))

    end

    return wedge_terms
end

"""
    dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack) where {m,n}

Computes the wedge product between two forms.
Use the precomputational "p" varient for the wedge_terms parameter. 
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack) where {m,n}
    # The last item in the val_pack should always be the range of simplices
    wedge_terms = zeros(last(last(val_pack)))
    return dec_c_wedge_product!(Tuple{m,n}, wedge_terms, α, β, val_pack)
end

dec_wedge_product(m::Int, n::Int, sd::HasDeltaSet) = dec_wedge_product(Tuple{m,n}, sd::HasDeltaSet)

"""
    dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)

Returns a function that computes the wedge product between two 0-forms.
"""
function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)
    (f, g) -> f .* g
end

"""
    dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where {k}

Returns a function that computes wedge product between a k and a 0-form.
"""
function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd)
    (α, g) -> dec_c_wedge_product(Tuple{0,k}, g, α, val_pack)
end

"""
    dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where {k}

Returns a function that computes the wedge product between a 0 and a k-form.
"""
function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd)
    (f, β) -> dec_c_wedge_product(Tuple{0,k}, f, β, val_pack)
end

"""
    dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)

Returns a function that computes the wedge product between a 1 and a 1-form.
"""
function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)
    val_pack = dec_p_wedge_product(Tuple{1,1}, sd)
    (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, val_pack)
end

# Boundary Operators
dec_boundary(n::Int, sd::HasDeltaSet) = sparse(dec_p_boundary(Val{n}, sd)...)

dec_p_boundary(::Type{Val{k}}, sd::HasDeltaSet; negate::Bool=false) where {k} =
    dec_p_derivbound(Val{k - 1}, sd, transpose=true, negate=negate)

# Dual Derivative Operators
dec_dual_derivative(n::Int, sd::HasDeltaSet) = sparse(dec_p_dual_derivative(Val{n}, sd)...)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet1D) =
    dec_p_boundary(Val{1}, sd, negate=true)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet2D) =
    dec_p_boundary(Val{2}, sd)

dec_p_dual_derivative(::Type{Val{1}}, sd::HasDeltaSet2D) =
    dec_p_boundary(Val{1}, sd, negate=true)

# Exterior Derivative Operators
dec_differential(n::Int, sd::HasDeltaSet) = sparse(dec_p_derivbound(Val{n}, sd)...)

function dec_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
    vec_size = 2 * ne(sd)

    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    I = Vector{Int32}(undef, vec_size)
    J = Vector{Int32}(undef, vec_size)

    V = Vector{Int8}(undef, vec_size)

    e_orient::Vector{Int8} = sd[:edge_orientation]
    for i in eachindex(e_orient)
        e_orient[i] = (e_orient[i] == 1 ? 1 : -1)
    end

    v0_list = @view sd[:∂v0]
    v1_list = @view sd[:∂v1]

    for i in edges(sd)
        j = 2 * i - 1

        I[j] = i
        I[j+1] = i

        J[j] = v0_list[i]
        J[j+1] = v1_list[i]

        sign_term = e_orient[i]

        V[j] = sign_term
        V[j+1] = -1 * sign_term
    end

    if (transpose)
        I, J = J, I
    end
    if (negate)
        V .= -1 .* V
    end

    (I, J, V)
end

function dec_p_derivbound(::Type{Val{1}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
    vec_size = 3 * ntriangles(sd)

    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    I = Vector{Int32}(undef, vec_size)
    J = Vector{Int32}(undef, vec_size)

    V = Vector{Int8}(undef, vec_size)

    tri_sign_list::Vector{Int8} = sign(2, sd)

    e_orient::Vector{Int8} = sd[:edge_orientation]
    for i in eachindex(e_orient)
        e_orient[i] = (e_orient[i] == 1 ? 1 : -1)
    end

    e0_list = @view sd[:∂e0]
    e1_list = @view sd[:∂e1]
    e2_list = @view sd[:∂e2]

    for i in triangles(sd)
        j = 3 * i - 2

        I[j] = i
        I[j+1] = i
        I[j+2] = i

        tri_sign = tri_sign_list[i]

        J[j] = e0_list[i]
        J[j+1] = e1_list[i]
        J[j+2] = e2_list[i]

        edge_sign_0 = e_orient[e0_list[i]]
        edge_sign_1 = e_orient[e1_list[i]]
        edge_sign_2 = e_orient[e2_list[i]]

        V[j] = edge_sign_0 * tri_sign
        V[j+1] = -1 * edge_sign_1 * tri_sign
        V[j+2] = edge_sign_2 * tri_sign

    end
    if (transpose)
        I, J = J, I
    end
    if (negate)
        V .= -1 .* V
    end

    (I, J, V)
end

# Diagonal Hodges 

function dec_p_hodge_diag(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type
    num_v_sd = nv(sd)

    hodge_diag_0 = zeros(float_type, num_v_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]

    for (d_edge_idx, v1) in enumerate(v1_list)
        if (1 <= v1 <= num_v_sd)
            hodge_diag_0[v1] += dual_lengths[d_edge_idx]
        end
    end
    return hodge_diag_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type
    vols::Vector{float_type} = volume(Val{1}, sd, edges(sd))
    return 1 ./ vols
end


function dec_p_hodge_diag(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    hodge_diag_0 = zeros(float_type, nv(sd))

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_1 = @view sd[:D_∂v1]
    dual_areas = @view sd[:dual_area]

    for (dual_tri, dual_edges) in enumerate(dual_edges_1)
        v = dual_v_1[dual_edges]
        hodge_diag_0[v] += dual_areas[dual_tri]
    end
    return hodge_diag_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    num_v_sd = nv(sd)
    num_e_sd = ne(sd)

    hodge_diag_1 = zeros(float_type, num_e_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]
    lengths = @view sd[:length]

    for (d_edge_idx, v1) in enumerate(v1_list)
        v1_shift = v1 - num_v_sd
        if (1 <= v1_shift <= num_e_sd)
            hodge_diag_1[v1_shift] += dual_lengths[d_edge_idx] / lengths[v1_shift]
        end
    end
    return hodge_diag_1
end

function dec_p_hodge_diag(::Type{Val{2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    tri_areas::Vector{float_type} = sd[:area]
    return 1 ./ tri_areas
end

dec_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge()) = dec_hodge_star(Val{n}, sd, hodge)
dec_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) = dec_hodge_star(Val{n}, sd, DiagonalHodge())
dec_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) = dec_hodge_star(Val{n}, sd, GeometricHodge())

dec_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where {k} =
    Diagonal(dec_p_hodge_diag(Val{k}, sd))

# These are Geometric Hodges 
# TODO: Still need better implementation for Hodge 1 in 2D
dec_hodge_star(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) =
    dec_hodge_star(Val{0}, sd, DiagonalHodge())

dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) =
    dec_hodge_star(Val{1}, sd, DiagonalHodge())

dec_hodge_star(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) =
    dec_hodge_star(Val{0}, sd, DiagonalHodge())

dec_hodge_star(::Type{Val{2}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) =
    dec_hodge_star(Val{2}, sd, DiagonalHodge())

crossdot(v1, v2) = begin
    v1v2 = cross(v1, v2)
    norm(v1v2) * (last(v1v2) == 0 ? 1 : sign(last(v1v2)))
end

function dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, point_type}, ::GeometricHodge) where {float_type, point_type}
    
    I = Vector{Int32}(undef, ntriangles(sd) * 9)
    J = Vector{Int32}(undef, ntriangles(sd) * 9)
    V = Vector{float_type}(undef, ntriangles(sd) * 9)

    # Reversed by contruction
    tri_edges_1 = @view sd[:∂e2]
    tri_edges_2 = @view sd[:∂e1]
    tri_edges_3 = @view sd[:∂e0]
    tri_edges = [tri_edges_1, tri_edges_2, tri_edges_3]
    
    edge_centers = @view sd[:edge_center]
    tri_centers = @view sd[:tri_center]
    tgts = @view sd[:∂v0]
    srcs = @view sd[:∂v1]

    # Regular points are contained in first nv(sd) spots
    # TODO: Decide if to use view or not, using a view means less memory but slightly slower
    dual_points::Vector{point_type} = sd[:dual_point]

    evt = Vector{point_type}(undef, 3)
    dvt = Vector{point_type}(undef, 3)

    idx = 0

    @inbounds for t in triangles(sd)
        dual_point_tct = dual_points[tri_centers[t]]
        for i in 1:3
            tri_edge = tri_edges[i][t]
            evt[i] = dual_points[tgts[tri_edge]] - dual_points[srcs[tri_edge]]
            dvt[i] = dual_point_tct - dual_points[edge_centers[tri_edge]]
        end
        dvt[2] *= -1

        # This relative orientation needs to be redefined for each triangle in the
        # case that the mesh has multiple independent connected components
        cross_ev_dv = cross(evt[1], dvt[1])
        rel_orient = (last(cross_ev_dv) == 0 ? 1 : sign(last(cross_ev_dv)))
        for i in 1:3
            diag_cross = crossdot(evt[i], dvt[i])
            if diag_cross != 0.0
                idx += 1
                I[idx] = tri_edges[i][t]
                J[idx] = tri_edges[i][t]
                V[idx] = rel_orient * diag_cross / dot(evt[i], evt[i])
            end
        end

        for p ∈ ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
            diag_dot = dot(evt[p[1]], dvt[p[1]]) / dot(evt[p[1]], evt[p[1]])
            val = diag_dot * dot(evt[p[1]], evt[p[3]]) 
            if val != 0.0
                idx += 1
                I[idx] = tri_edges[p[1]][t]
                J[idx] = tri_edges[p[2]][t]
                V[idx] = rel_orient * val / crossdot(evt[p[2]], evt[p[3]])
            end
        end
    end

    view_I = @view I[1:idx]
    view_J = @view J[1:idx]
    view_V = @view V[1:idx]

    sparse(view_I, view_J, view_V)
end

dec_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge()) = dec_inv_hodge_star(Val{n}, sd, hodge)
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) = dec_inv_hodge_star(Val{n}, sd, DiagonalHodge())
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) = dec_inv_hodge_star(Val{n}, sd, GeometricHodge())

# These are Diagonal Inverse Hodges
function dec_inv_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where {k}
    hdg = dec_p_hodge_diag(Val{k}, sd)
    mult_term = iseven(k * (ndims(sd) - k)) ? 1 : -1
    hdg .= (1 ./ hdg) .* mult_term
    return Diagonal(hdg)
end

# These are Geometric Inverse Hodges
dec_inv_hodge_star(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) =
    dec_inv_hodge_star(Val{0}, sd, DiagonalHodge())

dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) =
    dec_inv_hodge_star(Val{1}, sd, DiagonalHodge())

dec_inv_hodge_star(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) =
    dec_inv_hodge_star(Val{0}, sd, DiagonalHodge())

function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge)
    hdg_lu = LinearAlgebra.factorize(-1 * dec_hodge_star(1, sd, GeometricHodge()))
    x -> hdg_lu \ x
end

dec_inv_hodge_star(::Type{Val{2}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) =
    dec_inv_hodge_star(Val{2}, sd, DiagonalHodge())
end