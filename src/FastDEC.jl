""" FastDEC operator implementations

These operator implementations are meant to be optimized versions of some 
operators founds in SimplicialSets and DiscreteExteriorCalculus. They rely on 
the fact that the creation of the dual mesh ACSet is structured in a certain way. 

The current operators directly implemented are boundary, exterior derivative, hodge star
and wedge product.

If those ACSets are edited or altered after creation, these operators may not work.

For added performance, memory bounds checking is also turned off for some operators.
Please ensure that any inputs provided are what is expected by the operator to avoid 
memory corruption and crashes.
"""

module FastDEC
using LinearAlgebra: Diagonal, dot, norm, cross
using StaticArrays: SVector
using SparseArrays: sparse
using Base.Iterators

using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra.CSets

using ..SimplicialSets, ..DiscreteExteriorCalculus

export fast_boundary, fast_d, fast_dual_derivative, fast_hodge_star, fast_inv_hodge_star, fast_wedge_product

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

const VForm = SimplexForm{0}
const EForm = SimplexForm{1}
const TriForm = SimplexForm{2}
const TetForm = SimplexForm{3}

const DualV = DualSimplex{0}
const DualE = DualSimplex{1}
const DualTri = DualSimplex{2}

#= const fast_boundary = fast_∂

const fast_coboundary = fast_d
const fast_exterior_derivative = fast_d

const fast_hodge_star = fast_⋆

const fast_inv_hodge_star = fast_⋆⁻¹

const fast_wedge_product = fast_∧ =#

# Boundary Operators
fast_boundary(sd::HasDeltaSet, x::SimplexForm{n}) where n = SimplexForm{n-1}(fast_boundary(n::Int, sd::HasDeltaSet) * x.data)

fast_boundary(n::Int, sd::HasDeltaSet) = sparse(fast_p_boundary(Val{n}, sd)...)

fast_p_boundary(::Type{Val{k}}, sd::HasDeltaSet; negate = false) where k = 
    fast_p_derivbound(Val{k - 1}, sd, transpose = true, negate = negate)

# Dual Derivative Operators
fast_dual_derivative(sd::HasDeltaSet, x::DualForm{n}) where n = DualForm{n-1}(sparse(fast_p_dual_derivative(Val{n}, sd)...) * x.data)

fast_dual_derivative(n::Int, sd::HasDeltaSet) = sparse(fast_p_dual_derivative(Val{n}, sd)...)

fast_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet1D) = 
    fast_p_boundary(Val{1}, sd, negate = true)

fast_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet2D) = 
    fast_p_boundary(Val{2}, sd)

fast_p_dual_derivative(::Type{Val{1}}, sd::HasDeltaSet2D) = 
    fast_p_boundary(Val{1}, sd, negate = true)

# Exterior Derivative Operators
fast_d(sd::HasDeltaSet, x::SimplexForm{n}) where n = SimplexForm{n+1}(fast_d(n::Int, sd::HasDeltaSet) * x.data)

fast_d(n::Int, sd::HasDeltaSet) = sparse(fast_p_derivbound(Val{n}, sd)...)

function fast_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose = false, negate = false)
    vec_size = 2 * ne(sd)

    I = Vector{Int64}(undef, vec_size)
    J = Vector{Int64}(undef, vec_size)
    V = Vector{Int64}(undef, vec_size)

    e_orient::Vector{Int8} = sd[:edge_orientation]
    for i in eachindex(e_orient)
        e_orient[i] = (e_orient[i] == 1 ? 1 : -1)
    end

    v0_list = @view sd[:∂v0]
    v1_list = @view sd[:∂v1]

    for i in edges(sd)
        j = 2 * i - 1

        I[j] = i
        I[j + 1] = i

        J[j] = v0_list[i]
        J[j + 1] = v1_list[i]

        sign_term = e_orient[i]

        V[j] = sign_term
        V[j + 1] = -1 * sign_term
    end
    
    if(transpose)
        I, J = J, I
    end
    if(negate)
        V .= -1 .* V
    end
    
    (I, J, V)
end

function fast_p_derivbound(::Type{Val{1}}, sd::HasDeltaSet; transpose = false, negate = false)
    vec_size = 3 * ntriangles(sd)

    I = Vector{Int64}(undef, vec_size)
    J = Vector{Int64}(undef, vec_size)
    V = Vector{Int64}(undef, vec_size)

    tri_sign_list::Vector{Int64} = sign(2, sd)
    
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
        I[j + 1] = i
        I[j + 2] = i

        tri_sign = tri_sign_list[i]

        J[j] = e0_list[i]
        J[j + 1] = e1_list[i]
        J[j + 2] = e2_list[i]

        edge_sign_0 = e_orient[e0_list[i]]
        edge_sign_1 = e_orient[e1_list[i]]
        edge_sign_2 = e_orient[e2_list[i]]

        V[j] = edge_sign_0 * tri_sign
        V[j + 1] = -1 * edge_sign_1 * tri_sign
        V[j + 2] = edge_sign_2 * tri_sign

    end
    if(transpose)
        I, J = J, I
    end
    if(negate)
        V .= -1 .* V
    end

    (I, J, V)
end

# TODO: This relies on the assumption of a well ordering of the 
# the dual space simplices. If changed, use fast_p_wedge_product_zero
function fast_p_wedge_product(::Type{Tuple{0, 1}}, sd)
    return (hcat(sd[:∂v0], sd[:∂v1]), simplices(1, sd))
end

# TODO: This relies on the assumption of a well ordering of the 
# the dual space simplices. If changed, use fast_c_wedge_product_zero
# TODO: This assumes that the dual vertice on an edge is always the midpoint
function fast_c_wedge_product!(::Type{Tuple{0, 1}}, wedge_terms, f, α, val_pack)
    primal_vertices, simples = val_pack

    # wedge_terms = Vector{Float64}(undef, last(simples))
    wedge_terms .= 0.5 .* α
    @inbounds for i in simples
        wedge_terms[i] *= (f[primal_vertices[i, 1]] + f[primal_vertices[i, 2]])
    end

    return wedge_terms
end

function fast_p_wedge_product(::Type{Tuple{0, 2}}, sd::HasDeltaSet)

    simples = simplices(2, sd)

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_0 = @view sd[:D_∂v0]

    dual_edges_2 = @view sd[:D_∂e2]
    dual_v_1 = @view sd[:D_∂v1]

    dv = @view sd[:dual_area]
    vols = @view sd[:area]

    primal_vertices = Array{Int64}(undef, 6, ntriangles(sd))
    coeffs = Array{Float64}(undef, 6, ntriangles(sd))

    row_idx_in_col = ones(Int8, ntriangles(sd))
    shift::Int64 = nv(sd) + ne(sd)
    
    for dual_tri in eachindex(dual_edges_1)
        primal_tri = dual_v_0[dual_edges_1[dual_tri]] - shift
        row_idx = row_idx_in_col[primal_tri]
        
        primal_vertices[row_idx, primal_tri] = dual_v_1[dual_edges_2[dual_tri]]
        coeffs[row_idx, primal_tri] = dv[dual_tri] / vols[primal_tri]

        row_idx_in_col[primal_tri] += 1;
    end

    return (primal_vertices, coeffs, simples)
end

function fast_c_wedge_product!(::Type{Tuple{0, 2}}, wedge_terms, f, α, val_pack)
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

function fast_p_wedge_product(::Type{Tuple{0, k}}, sd) where k
    simples = simplices(k, sd)
    subsimples = map(x -> subsimplices(k, sd, x), simples)
    primal_vertices = map(x -> primal_vertex(k, sd, x), subsimples)

    vols = volume(k,sd,simples)
    dual_vols = map(y -> dual_volume(k,sd,y), subsimples)
    coeffs = dual_vols ./ vols
    return (primal_vertices, coeffs, simples)
end

function fast_c_wedge_product!(::Type{Tuple{0, k}}, wedge_terms, f, α, val_pack) where k
    primal_vertices, coeffs, simples = val_pack

    # TODO: May want to move this to be in the loop in case the coeffs width does change
    # Can use the below code in the preallocation to determine if we do have to recompute
    # the width at every step or if we can just compute it once.
    # all(map(x -> length(coeffs[x]), simples) .== length(coeffs[1]))
    width_iter = 1:length(coeffs[1])
    @inbounds for i in simples
                for j in width_iter
                    wedge_terms[i] += coeffs[i][j] * f[primal_vertices[i][j]]
                end
                wedge_terms[i] *= α[i]
    end
    
    return wedge_terms
end

# TODO: This relies on a well established ordering for the dual space simplices.
function fast_p_wedge_product(::Type{Tuple{1, 1}}, sd)
    simples = simplices(2, sd)

    areas = @view sd[:area]
    d_areas = @view sd[:dual_area]

    coeffs = Array{Float64}(undef, 3, ntriangles(sd))

    shift = ntriangles(sd)
    for i in 1:ntriangles(sd)
        area = areas[i]
        coeffs[1, i] = (d_areas[i] + d_areas[i + shift]) / area
        coeffs[2, i] = (d_areas[i + 2 * shift] + d_areas[i + 3 * shift]) / area
        coeffs[3, i] = (d_areas[i + 4 * shift] + d_areas[i + 5 * shift]) / area
    end

    e = Array{Int64}(undef, 3, ntriangles(sd))
    e[1, :] = ∂(2,0,sd)
    e[2, :] = ∂(2,1,sd)
    e[3, :] = ∂(2,2,sd)

    return (e, coeffs, simples)
end

function fast_c_wedge_product!(::Type{Tuple{1, 1}}, wedge_terms, α, β, val_pack)
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

function fast_c_wedge_product(::Type{Tuple{m, n}}, f, α, val_pack) where {m, n}
    # The last item in the val_pack should always be the range of simplices
    wedge_terms = zeros(last(last(val_pack)))
    return fast_c_wedge_product!(Tuple{m, n}, wedge_terms, f, α, val_pack)
end

fast_wedge_product(sd::HasDeltaSet, α::SimplexForm{k}, β::SimplexForm{l}) where {k,l} = 
    SimplexForm{k+l}(fast_wedge_product(k, l, sd)(α, β))

fast_wedge_product(m::Int, n::Int, sd::HasDeltaSet) = fast_wedge_product(Tuple{m,n}, sd::HasDeltaSet)

function fast_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)
    (f, g) -> f .* g
end

function fast_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where k
    val_pack = fast_p_wedge_product(Tuple{0, k}, sd)
    (α, g) -> fast_c_wedge_product(Tuple{0, k}, g, α, val_pack)
end

function fast_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where k
    val_pack = fast_p_wedge_product(Tuple{0, k}, sd)
    (f, β) -> fast_c_wedge_product(Tuple{0, k}, f, β, val_pack)
end

function fast_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)
    val_pack = fast_p_wedge_product(Tuple{1,1}, sd)
    (α, β) -> fast_c_wedge_product(Tuple{1,1}, α, β,val_pack)
end

# These are Diagonal Hodges 

# TODO: Check this Hodge with a 1D mesh
function fast_p_hodge_diag(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D)
    num_v_sd = nv(sd)

    hodge_diag_0 = zeros(num_v_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]
    
    for d_edge_idx in eachindex(v1_list)
        v1 = v1_list[d_edge_idx]
        if(1 <= v1 <= num_v_sd)
            hodge_diag_0[v1] += dual_lengths[d_edge_idx]
        end
    end
    return hodge_diag_0
end

# TODO: Check this Hodge with a 1D mesh
function fast_p_hodge_diag(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D)
    return 1 ./ volume(Val{1}, sd, edges(sd))
end

function fast_p_hodge_diag(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D)
    hodge_diag_0 = zeros(nv(sd))

    dual_edges_1 = @view sd[:D_∂e1]
    dual_v_1 = @view sd[:D_∂v1]
    dual_areas = @view sd[:dual_area]

    for dual_tri in eachindex(dual_edges_1)
        v = dual_v_1[dual_edges_1[dual_tri]]
        hodge_diag_0[v] += dual_areas[dual_tri]
    end
    return hodge_diag_0
end

function fast_p_hodge_diag(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D)
    num_v_sd = nv(sd)
    num_e_sd = ne(sd)

    hodge_diag_1 = zeros(num_e_sd)

    v1_list = @view sd[:D_∂v1]
    dual_lengths = @view sd[:dual_length]
    lengths = @view sd[:length]

    for d_edge_idx in eachindex(v1_list)
        v1_shift = v1_list[d_edge_idx] - num_v_sd
        if(1 <= v1_shift <= num_e_sd)
            hodge_diag_1[v1_shift] += dual_lengths[d_edge_idx] / lengths[v1_shift]
        end
    end
    return hodge_diag_1
end

function fast_p_hodge_diag(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D)
    tri_areas = @view sd[:area]
    return 1 ./ tri_areas
end

fast_hodge_star(n::Int, sd::HasDeltaSet; hodge = GeometricHodge()) = fast_hodge_star(Val{n}, sd, hodge) 
fast_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) = fast_hodge_star(Val{n}, sd, DiagonalHodge())
fast_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) = fast_hodge_star(Val{n}, sd, GeometricHodge())

fast_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where k = 
    Diagonal(fast_p_hodge_diag(Val{k}, sd))

# These are Geometric Hodges 
# Implementation for Hodge 1 in 2D is below
fast_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge) = 
    fast_hodge_star(Val{0}, sd, DiagonalHodge())

fast_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge) = 
    fast_hodge_star(Val{1}, sd, DiagonalHodge())

fast_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge) = 
    fast_hodge_star(Val{0}, sd, DiagonalHodge())

fast_hodge_star(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge) = 
    fast_hodge_star(Val{2}, sd, DiagonalHodge())

#= function crossdot(a, b)
    x, y, z = 1, 2, 3
    c_x = a[y] * b[z] - a[z] * b[y]
    c_y = a[z] * b[x] - a[x] * b[z]
    c_z = a[x] * b[y] - a[y] * b[x]

    flipbit = (c_z == 0 ? 1.0 : sign(c_z))
    c_norm = sqrt(c_x^2 + c_y^2 + c_z^2)
    return c_norm * flipbit
end =#

crossdot(v1, v2) = begin
    v1v2 = cross(v1, v2)
    norm(v1v2) * (last(v1v2) == 0 ? 1.0 : sign(last(v1v2)))
  end
  

function fast_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge)

    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()

    rel_orient::Float64 = 0.0

    edge_centers = @view sd[:edge_center]
    tri_centers = @view sd[:tri_center]

    points::Vector{Point3D} = sd[:point]
    dual_points::Vector{Point3D} = sd[:dual_point]

    # points = sd[:point]
    # dual_points = sd[:dual_point]

    tgts = @view sd[:∂v0]
    srcs = @view sd[:∂v1]

    tri_signs::Vector{Int64} = sign(2, sd)

    for t in triangles(sd)
      e = reverse(triangle_edges(sd, t))
      ev = points[tgts[e]] .- points[srcs[e]]

      tc = dual_points[tri_centers[t]]
  
      dv = map(enumerate(dual_points[edge_centers[e]])) do (i,v)
        (tc - v) * (i == 2 ? -1 : 1)
      end

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
  
      for p ∈ ((1,2,3), (1,3,2), (2,1,3),
               (2,3,1), (3,1,2), (3,2,1))
        val = rel_orient * tri_signs[t] * diag_dot[p[1]] *
                dot(ev[p[1]], ev[p[3]]) / crossdot(ev[p[2]], ev[p[3]])
        if val != 0.0
          push!(I, e[p[1]])
          push!(J, e[p[2]])
          push!(V, val)
        end
      end
    end
    sparse(I,J,V)
end
  
fast_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge = GeometricHodge()) = fast_inv_hodge_star(Val{n}, sd, hodge)
fast_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) = fast_inv_hodge_star(Val{n}, sd, DiagonalHodge())
fast_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) = fast_inv_hodge_star(Val{n}, sd, GeometricHodge())

# These are Diagonal Inverse Hodges
function fast_inv_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where k
    hdg = fast_p_hodge_diag(Val{k}, sd)
    mult_term = iseven(k*(ndims(sd)-k)) ? 1 : -1
    hdg .= (1 ./ hdg) .* mult_term
    return Diagonal(hdg)
end

# These are Geometric Inverse Hodges
fast_inv_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge) = 
    fast_inv_hodge_star(Val{0}, sd, DiagonalHodge())

fast_inv_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex1D, ::GeometricHodge) = 
    fast_inv_hodge_star(Val{1}, sd, DiagonalHodge())

fast_inv_hodge_star(::Type{Val{0}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge) = 
    fast_inv_hodge_star(Val{0}, sd, DiagonalHodge())

# TODO: Change this hodge to dec hodge when implemented
function fast_inv_hodge_star(::Type{Val{1}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge)
    hdg_lu = LinearAlgebra.factorize(fast_hodge_star(1, sd, GeometricHodge()))
    x -> hdg_lu \ x
end

fast_inv_hodge_star(::Type{Val{2}}, sd::AbstractDeltaDualComplex2D, ::GeometricHodge) = 
    fast_inv_hodge_star(Val{2}, sd, DiagonalHodge())

end