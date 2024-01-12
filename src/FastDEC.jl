module FastDEC
using LinearAlgebra: Diagonal, dot, norm, cross
using StaticArrays: SVector
using SparseArrays: sparse
using LinearAlgebra
using Base.Iterators
using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra.CSets
using ..SimplicialSets, ..DiscreteExteriorCalculus

export dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star, dec_wedge_product

# TODO: This relies on the assumption of a well ordering of the 
# the dual space simplices. If changed, use dec_p_wedge_product_zero
function dec_p_wedge_product(::Type{Tuple{0,1}}, sd; float_type=Float64)
    return (hcat(convert(Vector{Int32}, sd[:∂v0]), convert(Vector{Int32}, sd[:∂v1])), simplices(1, sd))
    # return (hcat(sd[:∂v0], sd[:∂v1]), simplices(1, sd))
end

# TODO: This relies on the assumption of a well ordering of the 
# the dual space simplices. If changed, use dec_c_wedge_product_zero
# TODO: This assumes that the dual vertice on an edge is always the midpoint
# TODO: Add options to change 0.5 to a different float
function dec_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack)
    primal_vertices, simples = val_pack

    wedge_terms .= 0.5 .* α
    @inbounds for i in simples
        wedge_terms[i] *= (f[primal_vertices[i, 1]] + f[primal_vertices[i, 2]])
    end

    return wedge_terms
end

function dec_p_wedge_product(::Type{Tuple{0,2}}, sd::HasDeltaSet; float_type=Float64)

    simples = simplices(2, sd)

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_0 = @view sd[:D_∂v0]

    dual_edges_2 = @view sd[:D_∂e2]
    dual_v_1 = @view sd[:D_∂v1]

    dv = @view sd[:dual_area]
    vols = @view sd[:area]

    # TODO: This is assuming that meshes don't have too many entries
    # This type should be settable by the user and default set to Int32
    primal_vertices = Array{Int32}(undef, 6, ntriangles(sd))
    coeffs = Array{float_type}(undef, 6, ntriangles(sd))

    row_idx_in_col = ones(Int8, ntriangles(sd))
    shift::Int = nv(sd) + ne(sd)

    for dual_tri in eachindex(dual_edges_1)
        primal_tri = dual_v_0[dual_edges_1[dual_tri]] - shift
        row_idx = row_idx_in_col[primal_tri]

        primal_vertices[row_idx, primal_tri] = dual_v_1[dual_edges_2[dual_tri]]
        coeffs[row_idx, primal_tri] = dv[dual_tri] / vols[primal_tri]

        row_idx_in_col[primal_tri] += 1
    end

    return (primal_vertices, coeffs, simples)
end

function dec_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack)
    pv, coeffs, simples = val_pack

    # TODO: May want to move this to be in the loop in case the coeffs width does change
    # Can use the below code in the preallocation to determine if we do have to recompute
    # the width at every step or if we can just compute it once.
    # all(map(x -> length(coeffs[x]), simples) .== length(coeffs[1]))
    @inbounds for i in simples
        wedge_terms[i] = α[i] * (coeffs[1, i] * f[pv[1, i]] + coeffs[2, i] * f[pv[2, i]]
                                 + coeffs[3, i] * f[pv[3, i]] + coeffs[4, i] * f[pv[4, i]]
                                 + coeffs[5, i] * f[pv[5, i]] + coeffs[6, i] * f[pv[6, i]])
    end

    return wedge_terms
end

# TODO: This relies on a well established ordering for 
# the dual space simplices. If changed, use dec_p_wedge_product_ones_safe
function dec_p_wedge_product(::Type{Tuple{1,1}}, sd; float_type=Float64)
    simples = simplices(2, sd)

    areas = @view sd[:area]
    d_areas = @view sd[:dual_area]

    coeffs = Array{float_type}(undef, 3, ntriangles(sd))

    shift = ntriangles(sd)
    for i in 1:ntriangles(sd)
        area = areas[i]
        coeffs[1, i] = (d_areas[i] + d_areas[i+shift]) / area
        coeffs[2, i] = (d_areas[i+2*shift] + d_areas[i+3*shift]) / area
        coeffs[3, i] = (d_areas[i+4*shift] + d_areas[i+5*shift]) / area
    end
    # TODO: This is assuming that meshes don't have too many entries
    # This type should be settable by the user and default set to Int32
    # e = Array{Int64}(undef, 3, ntriangles(sd))
    e = Array{Int32}(undef, 3, ntriangles(sd))

    e[1, :] = ∂(2, 0, sd)
    e[2, :] = ∂(2, 1, sd)
    e[3, :] = ∂(2, 2, sd)

    return (e, coeffs, simples)
end

function dec_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, α, β, val_pack)
    e, coeffs, simples = val_pack

    @inbounds for i in simples
        ae0, ae1, ae2 = α[e[1, i]], α[e[2, i]], α[e[3, i]]
        be0, be1, be2 = β[e[1, i]], β[e[2, i]], β[e[3, i]]

        wedge_terms[i] += (coeffs[1, i] * (ae2 * be1 - ae1 * be2)
                           + coeffs[2, i] * (ae2 * be0 - ae0 * be2)
                           + coeffs[3, i] * (ae1 * be0 - ae0 * be1))
    end

    return wedge_terms
end

function dec_c_wedge_product(::Type{Tuple{m,n}}, f, α, val_pack) where {m,n}
    # The last item in the val_pack should always be the range of simplices
    wedge_terms = zeros(last(last(val_pack)))
    return dec_c_wedge_product!(Tuple{m,n}, wedge_terms, f, α, val_pack)
end

dec_wedge_product(m::Int, n::Int, sd::HasDeltaSet) = dec_wedge_product(Tuple{m,n}, sd::HasDeltaSet)

function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet, float_type=Float64)
    (f, g) -> f .* g
end

function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet; float_type=Float64) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd, float_type=float_type)
    (α, g) -> dec_c_wedge_product(Tuple{0,k}, g, α, val_pack)
end

function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet; float_type=Float64) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd, float_type=float_type)
    (f, β) -> dec_c_wedge_product(Tuple{0,k}, f, β, val_pack)
end

function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D; float_type=Float64)
    val_pack = dec_p_wedge_product(Tuple{1,1}, sd, float_type=float_type)
    (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, val_pack)
end

# Boundary Operators
dec_boundary(n::Int, sd::HasDeltaSet) = sparse(dec_p_boundary(Val{n}, sd)...)

dec_p_boundary(::Type{Val{k}}, sd::HasDeltaSet; negate=false) where {k} =
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

function dec_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose=false, negate=false)
    vec_size = 2 * ne(sd)

    # TODO: This is assuming that meshes don't have too many entries
    # This type should be settable by the user and default set to Int32
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

function dec_p_derivbound(::Type{Val{1}}, sd::HasDeltaSet; transpose=false, negate=false)
    vec_size = 3 * ntriangles(sd)

    # TODO: This is assuming that meshes don't have too many entries
    # This type should be settable by the user and default set to Int32
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

# TODO: Check this Hodge with a 1D mesh
function dec_p_hodge_diag(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D; float_type=Float64)
    num_v_sd = nv(sd)

    hodge_diag_0 = zeros(float_type, num_v_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]

    for d_edge_idx in eachindex(v1_list)
        v1 = v1_list[d_edge_idx]
        if (1 <= v1 <= num_v_sd)
            hodge_diag_0[v1] += dual_lengths[d_edge_idx]
        end
    end
    return hodge_diag_0
end

# TODO: Check this Hodge with a 1D mesh
function dec_p_hodge_diag(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D; float_type=Float64)
    vols::Vector{float_type} = volume(Val{1}, sd, edges(sd))
    return 1 ./ vols
end


function dec_p_hodge_diag(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D; float_type=Float64)
    hodge_diag_0 = zeros(float_type, nv(sd))

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_1 = @view sd[:D_∂v1]
    dual_areas = @view sd[:dual_area]

    for dual_tri in eachindex(dual_edges_1)
        v = dual_v_1[dual_edges_1[dual_tri]]
        hodge_diag_0[v] += dual_areas[dual_tri]
    end
    return hodge_diag_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D; float_type=Float64)
    num_v_sd = nv(sd)
    num_e_sd = ne(sd)

    hodge_diag_1 = zeros(float_type, num_e_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]
    lengths = @view sd[:length]

    for d_edge_idx in eachindex(v1_list)
        v1_shift = v1_list[d_edge_idx] - num_v_sd
        if (1 <= v1_shift <= num_e_sd)
            hodge_diag_1[v1_shift] += dual_lengths[d_edge_idx] / lengths[v1_shift]
        end
    end
    return hodge_diag_1
end

function dec_p_hodge_diag(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D; float_type=Float64)
    # tri_areas = @view sd[:area]
    tri_areas::Vector{float_type} = sd[:area]
    return 1 ./ tri_areas
end

dec_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge(), float_type=Float64) = dec_hodge_star(Val{n}, sd, hodge, float_type=float_type)
dec_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge; float_type=Float64) = dec_hodge_star(Val{n}, sd, DiagonalHodge(), float_type=float_type)
dec_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge; float_type=Float64) = dec_hodge_star(Val{n}, sd, GeometricHodge(), float_type=float_type)

dec_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge; float_type=Float64) where {k} =
    Diagonal(dec_p_hodge_diag(Val{k}, sd, float_type=float_type))

# These are Geometric Hodges 
# TODO: Still need implementation for Hodge 1 in 2D
dec_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge; float_type=Float64) =
    dec_hodge_star(Val{0}, sd, DiagonalHodge(), float_type=float_type)

dec_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge; float_type=Float64) =
    dec_hodge_star(Val{1}, sd, DiagonalHodge(), float_type=float_type)

dec_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type=Float64) =
    dec_hodge_star(Val{0}, sd, DiagonalHodge(), float_type=float_type)

dec_hodge_star(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type=Float64) =
    dec_hodge_star(Val{2}, sd, DiagonalHodge(), float_type=float_type)

crossdot(v1, v2) = begin
    v1v2 = cross(v1, v2)
    norm(v1v2) * (last(v1v2) == 0 ? 1.0 : sign(last(v1v2)))
end

function dec_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type::DataType = Float64)

    I = Vector{Int32}()
    J = Vector{Int32}()
    V = Vector{float_type}()

    edge_centers = @view sd[:edge_center]
    tri_centers = @view sd[:tri_center]

    # points::Vector{Point3{Float64}} = sd[:point]
    # dual_points::Vector{Point3{Float64}} = sd[:dual_point]

    #TODO: Figure out how to type these since both Point2D and Point3D can be used
    points = sd[:point]
    dual_points = sd[:dual_point]

    tgts = @view sd[:∂v0]
    srcs = @view sd[:∂v1]

    tri_signs::Vector{Int8} = sign(2, sd)

    tri_edges = Array{Int32}(undef, 3, ntriangles(sd))
    # Reversed by contruction
    tri_edges[1, :] = sd[:∂e2]
    tri_edges[2, :] = sd[:∂e1]
    tri_edges[3, :] = sd[:∂e0]

    evt = points[tgts[tri_edges]] .- points[srcs[tri_edges]]
    tct = dual_points[tri_centers]
    dvt = dual_points[edge_centers[tri_edges]]
    for i in 1:3
        dvt[i, :] .= tct .- dvt[i, :]
    end
    dvt[2, :] .= dvt[2, :] .* -1

    for t in triangles(sd)
        e = tri_edges[:, t]
        ev = evt[:, t]
        dv = dvt[:, t]

        diag_dot = map(1:3) do i
            dot(ev[i], dv[i]) / dot(ev[i], ev[i])
        end

        # This relative orientation needs to be redefined for each triangle in the
        # case that the mesh has multiple independent connected components
        rel_orient = 0.0
        for i in 1:3
            diag_cross = tri_signs[t] * crossdot(ev[i], dv[i]) /
                         dot(ev[i], ev[i])
            if diag_cross != 0.0
                # Decide the orientation of the mesh relative to z-axis (see crossdot)
                # For optimization, this could be moved out of this loop
                if rel_orient == 0.0
                    rel_orient = sign(diag_cross)
                end

                push!(I, e[i])
                push!(J, e[i])
                push!(V, diag_cross * rel_orient)
            end
        end

        for p ∈ ((1, 2, 3), (1, 3, 2), (2, 1, 3),
            (2, 3, 1), (3, 1, 2), (3, 2, 1))
            val = rel_orient * tri_signs[t] * diag_dot[p[1]] *
                  dot(ev[p[1]], ev[p[3]]) / crossdot(ev[p[2]], ev[p[3]])
            if val != 0.0
                push!(I, e[p[1]])
                push!(J, e[p[2]])
                push!(V, val)
            end
        end
    end
    sparse(I, J, V)
end

dec_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge(), float_type=Float64) = dec_inv_hodge_star(Val{n}, sd, hodge, float_type=float_type)
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge; float_type=Float64) = dec_inv_hodge_star(Val{n}, sd, DiagonalHodge(), float_type=float_type)
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge; float_type=Float64) = dec_inv_hodge_star(Val{n}, sd, GeometricHodge(), float_type=float_type)

# These are Diagonal Inverse Hodges
function dec_inv_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge; float_type=Float64) where {k}
    hdg = dec_p_hodge_diag(Val{k}, sd, float_type=float_type)
    mult_term = iseven(k * (ndims(sd) - k)) ? 1 : -1
    hdg .= (1 ./ hdg) .* mult_term
    return Diagonal(hdg)
end

# These are Geometric Inverse Hodges
dec_inv_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge; float_type=Float64) =
    dec_inv_hodge_star(Val{0}, sd, DiagonalHodge(), float_type=float_type)

dec_inv_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge; float_type=Float64) =
    dec_inv_hodge_star(Val{1}, sd, DiagonalHodge(), float_type=float_type)

dec_inv_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type=Float64) =
    dec_inv_hodge_star(Val{0}, sd, DiagonalHodge(), float_type=float_type)

function dec_inv_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type=Float64)
    hdg_lu = LinearAlgebra.factorize(-1 * dec_hodge_star(1, sd, GeometricHodge(), float_type=float_type))
    x -> hdg_lu \ x
end

dec_inv_hodge_star(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge; float_type=Float64) =
    dec_inv_hodge_star(Val{2}, sd, DiagonalHodge(), float_type=float_type)

end