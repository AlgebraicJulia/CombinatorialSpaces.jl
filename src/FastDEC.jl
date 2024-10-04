""" The discrete exterior calculus (DEC) with high performance in mind.

This module provides similar fuctionality to the DiscreteExteriorCalculus module
but uses assumptions about the ACSet mesh structure to greatly improve performance.
Some operators, like the exterior derivative are returned as sparse matrices while others,
like the wedge product, are instead returned as functions that will compute the product.
"""
module FastDEC
using LinearAlgebra: Diagonal, dot, norm, cross
using StaticArrays: SVector, MVector
using SparseArrays: sparse, spzeros, SparseMatrixCSC
using LinearAlgebra
using Base.Iterators
using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra.CSets
using ..SimplicialSets, ..DiscreteExteriorCalculus
import ..DiscreteExteriorCalculus: ∧
import ..SimplicialSets: numeric_sign

export dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star, dec_wedge_product, dec_c_wedge_product, dec_p_wedge_product, dec_c_wedge_product!,
       dec_wedge_product_pd, dec_wedge_product_dp, ∧,
       interior_product_dd, ℒ_dd,
       dec_wedge_product_dd,
       Δᵈ,
       avg₀₁, avg_01, avg₀₁_mat, avg_01_mat
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

"""    dec_p_wedge_product(::Type{Tuple{0,1}}, sd::EmbeddedDeltaDualComplex2D)

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
"""    dec_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack)

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

"""    dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

Precomputes values for the wedge product between a 0 and 2-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient.
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    primal_vertices = Array{Int32}(undef, 6, ntriangles(sd))
    coeffs = Array{float_type}(undef, 6, ntriangles(sd))

    shift::Int = ntriangles(sd)

    @inbounds for primal_tri in triangles(sd)
      for dual_tri_idx in 1:6
        dual_tri_real = primal_tri + (dual_tri_idx - 1) * shift

        primal_vertices[dual_tri_idx, primal_tri] = sd[sd[dual_tri_real, :D_∂e2], :D_∂v1]
        coeffs[dual_tri_idx, primal_tri] = sd[dual_tri_real, :dual_area] / sd[primal_tri, :area]
      end
  end


    return (primal_vertices, coeffs, triangles(sd))
end

"""    dec_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack)

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

"""    dec_p_wedge_product(::Type{Tuple{1,1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

Precomputes values for the wedge product between a 1 and 1-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient.
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{1,1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    simples =

    coeffs = Array{float_type}(undef, 3, ntriangles(sd))

    shift = ntriangles(sd)
    @inbounds for i in 1:ntriangles(sd)
        area = 2 * sd[i, :area]
        coeffs[1, i] = (sd[i, :dual_area] + sd[i+shift, :dual_area]) / area
        coeffs[2, i] = (sd[i+2*shift, :dual_area] + sd[i+3*shift, :dual_area]) / area
        coeffs[3, i] = (sd[i+4*shift, :dual_area] + sd[i+5*shift, :dual_area]) / area
    end
    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    e = Array{Int32}(undef, 3, ntriangles(sd))

    e[1, :] = ∂(2, 0, sd)
    e[2, :] = ∂(2, 1, sd)
    e[3, :] = ∂(2, 2, sd)

    return (e, coeffs, triangles(sd))
end

"""    dec_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, f, α, val_pack)

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

"""    dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack) where {m,n}

Computes the wedge product between two forms.
Use the precomputational "p" varient for the wedge_terms parameter.
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack) where {m,n}
    # The last item in the val_pack should always be the range of simplices
    wedge_terms = zeros(eltype(α), last(last(val_pack)))
    return dec_c_wedge_product!(Tuple{m,n}, wedge_terms, α, β, val_pack)
end

dec_wedge_product(m::Int, n::Int, sd::HasDeltaSet) = dec_wedge_product(Tuple{m,n}, sd::HasDeltaSet)

"""    dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)

Returns a function that computes the wedge product between two 0-forms.
"""
function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)
    (f, g) -> f .* g
end

"""    dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where {k}

Returns a function that computes wedge product between a k and a 0-form.
"""
function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd)
    (α, g) -> dec_c_wedge_product(Tuple{0,k}, g, α, val_pack)
end

"""    dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where {k}

Returns a function that computes the wedge product between a 0 and a k-form.
"""
function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where {k}
    val_pack = dec_p_wedge_product(Tuple{0,k}, sd)
    (f, β) -> dec_c_wedge_product(Tuple{0,k}, f, β, val_pack)
end

"""    dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)

Returns a function that computes the wedge product between a 1 and a 1-form.
"""
function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)
    val_pack = dec_p_wedge_product(Tuple{1,1}, sd)
    (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, val_pack)
end

"""    function wedge_dd_01_mat(sd::HasDeltaSet)

Returns a matrix that can be multiplied to a dual 0-form, before being
elementwise-multiplied by a dual 1-form, encoding the wedge product.
"""
function wedge_dd_01_mat(sd::HasDeltaSet)
  m = spzeros(ne(sd), ntriangles(sd))
  for e in edges(sd)
    des = elementary_duals(1,sd,e)
    dvs = sd[des, :D_∂v0]
    tris = only.(incident(sd, dvs, :tri_center))
    ws = sd[des, :dual_length] ./ sum(sd[des, :dual_length])
    for (w,t) in zip(ws,tris)
      m[e,t] = w
    end
  end
  m
end

"""    dec_wedge_product_dd(::Type{Tuple{0,1}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a dual
0-form and a dual 1-form.
"""
function dec_wedge_product_dd(::Type{Tuple{0,1}}, sd::HasDeltaSet)
  m = wedge_dd_01_mat(sd)
  (f,g) -> (m * f) .* g
end

"""    dec_wedge_product_dd(::Type{Tuple{1,0}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a dual
1-form and a dual 0-form.
"""
function dec_wedge_product_dd(::Type{Tuple{1,0}}, sd::HasDeltaSet)
  m = wedge_dd_01_mat(sd)
  (f,g) -> f .* (m * g)
end

"""    function wedge_pd_01_mat(sd::HasDeltaSet)

Returns a matrix that can be multiplied to a primal 0-form, before being
elementwise-multiplied by a dual 1-form, encoding the wedge product.

This function assumes barycentric means and performs bilinear interpolation. It
is not known if this definition has appeared in the literature or any code.
"""
function wedge_pd_01_mat(sd::HasDeltaSet)
  m = spzeros(ne(sd), nv(sd))
  for e in edges(sd)
    α, β = edge_vertices(sd,e)
    des = elementary_duals(1,sd,e)
    dvs = sd[des, :D_∂v0]
    tris = only.(incident(sd, dvs, :tri_center))
    γδ = map(tris) do t
      only(filter(x -> x ∉ [α,β], triangle_vertices(sd,t)))
    end
    ws = sd[des, :dual_length] ./ sum(sd[des, :dual_length])
    for (w,l) in zip(ws,γδ)
      m[e,α] += w*5/12
      m[e,β] += w*5/12
      m[e,l] += w*2/12
    end
  end
  m
end

"""    dec_wedge_product_dp(::Type{Tuple{1,0}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a dual
1-form and a primal 0-form.

This function assumes barycentric means and performs bilinear interpolation. It
is not known if this definition has appeared in the literature or any code.
"""
function dec_wedge_product_dp(::Type{Tuple{1,0}}, sd::HasDeltaSet)
  m = wedge_pd_01_mat(sd)
  (f,g) -> f .* (m * g)
end

"""    function dec_wedge_product_pd(::Type{Tuple{0,1}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a primal
0-form and a dual 1-form.

This function assumes barycentric means and performs bilinear interpolation. It
is not known if this definition has appeared in the literature or any code.
"""
function dec_wedge_product_pd(::Type{Tuple{0,1}}, sd::HasDeltaSet)
  m = wedge_pd_01_mat(sd)
  (g,f) -> (m * g) .* f
end

"""    dec_wedge_product_pd(::Type{Tuple{1,1}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a primal
1-form and a dual 1-form.
"""
function dec_wedge_product_pd(::Type{Tuple{1,1}}, sd::HasDeltaSet)
  ♭♯_m = ♭♯_mat(sd)
  Λ_cached = dec_wedge_product(Tuple{1, 1}, sd)
  (f, g) -> Λ_cached(f, ♭♯_m * g)
end

"""    dec_wedge_product_dp(::Type{Tuple{1,1}}, sd::HasDeltaSet)

Returns a cached function that computes the wedge product between a dual 1-form
and a primal 1-form.
"""
function dec_wedge_product_dp(::Type{Tuple{1,1}}, sd::HasDeltaSet)
  ♭♯_m = ♭♯_mat(sd)
  Λ_cached = dec_wedge_product(Tuple{1, 1}, sd)
  (f, g) -> Λ_cached(♭♯_m * f, g)
end

"""    ∧(s::HasDeltaSet, α::SimplexForm{1}, β::DualForm{1})

Wedge product of a primal 1-form and a dual 1-form.

Chain the musical isomorphisms to interpolate the dual 1-form to a primal
1-form, using the linear least squares ♯. Then use the CombinatorialSpaces
version of the Hirani primal-primal weddge.
"""
∧(s::HasDeltaSet, α::SimplexForm{1}, β::DualForm{1}) =
  dec_wedge_product_pd(Tuple{1,1}, s)(α, β)

"""    ∧(s::HasDeltaSet, α::DualForm{1}, β::SimplexForm{1})

Wedge product of a dual 1-form and a primal 1-form.

Chain the musical isomorphisms to interpolate the dual 1-form to a primal
1-form. Then use the CombinatorialSpaces version of the Hirani primal-primal
weddge (without explicitly dividing by 2.)
"""
∧(s::HasDeltaSet, α::DualForm{1}, β::SimplexForm{1}) =
  dec_wedge_product_dp(Tuple{1,1}, s)(α, β)


# Boundary Operators
"""
    dec_boundary(n::Int, sd::HasDeltaSet)

Gives the boundary operator (as a matrix) for `(n+1)`-simplices to `(n)`-simplices
"""
dec_boundary(n::Int, sd::HasDeltaSet) = sparse(dec_p_boundary(Val{n}, sd)...)

dec_p_boundary(::Type{Val{k}}, sd::HasDeltaSet; negate::Bool=false) where {k} =
    dec_p_derivbound(Val{k - 1}, sd, transpose=true, negate=negate)

# Dual Derivative Operators
"""
    dec_dual_derivative(n::Int, sd::HasDeltaSet)

Gives the dual exterior derivative (as a matrix) between dual `n`-simplices and dual `(n+1)`-simplices
"""
dec_dual_derivative(n::Int, sd::HasDeltaSet) = sparse(dec_p_dual_derivative(Val{n}, sd)...)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet1D) =
    dec_p_boundary(Val{1}, sd, negate=true)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet2D) =
    dec_p_boundary(Val{2}, sd)

dec_p_dual_derivative(::Type{Val{1}}, sd::HasDeltaSet2D) =
    dec_p_boundary(Val{1}, sd, negate=true)

# Exterior Derivative Operators
"""
    dec_differential(n::Int, sd::HasDeltaSet)

Gives the exterior derivative (as a matrix) between `n`-simplices and `(n+1)`-simplices
"""
dec_differential(n::Int, sd::HasDeltaSet) = sparse(dec_p_derivbound(Val{n}, sd)...)

function dec_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
    vec_size = 2 * ne(sd)

    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    I = Vector{Int32}(undef, vec_size)
    J = Vector{Int32}(undef, vec_size)

    V = Vector{Int8}(undef, vec_size)

    for i in edges(sd)
        j = 2 * i - 1

        I[j] = i
        I[j+1] = i

        J[j] = sd[i, :∂v0]
        J[j+1] = sd[i, :∂v1]

        sign_term = numeric_sign(sd[i, :edge_orientation]::Bool)

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

    for i in triangles(sd)
        j = 3 * i - 2

        I[j] = i
        I[j+1] = i
        I[j+2] = i

        tri_sign = numeric_sign(sd[i, :tri_orientation]::Bool)

        J[j] = sd[i, :∂e0]
        J[j+1] = sd[i, :∂e1]
        J[j+2] = sd[i, :∂e2]

        edge_sign_0 = numeric_sign(sd[sd[i, :∂e0], :edge_orientation]::Bool)
        edge_sign_1 = numeric_sign(sd[sd[i, :∂e1], :edge_orientation]::Bool)
        edge_sign_2 = numeric_sign(sd[sd[i, :∂e2], :edge_orientation]::Bool)

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

    for d_edge_idx in parts(sd, :DualE)
      v1 = sd[d_edge_idx, :D_∂v1]
      if (1 <= v1 <= num_v_sd)
          hodge_diag_0[v1] += sd[d_edge_idx, :dual_length]
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

    for dual_tri in parts(sd, :DualTri)
      v = sd[sd[dual_tri, :D_∂e1], :D_∂v1]
      hodge_diag_0[v] += sd[dual_tri, :dual_area]
    end
    return hodge_diag_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    num_v_sd = nv(sd)
    num_e_sd = ne(sd)

    hodge_diag_1 = zeros(float_type, num_e_sd)

    for d_edge_idx in parts(sd, :DualE)
      v1_shift = sd[d_edge_idx, :D_∂v1] - num_v_sd
      if (1 <= v1_shift <= num_e_sd)
          hodge_diag_1[v1_shift] += sd[d_edge_idx, :dual_length] / sd[v1_shift, :length]
      end
    end
    return hodge_diag_1
end

function dec_p_hodge_diag(::Type{Val{2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    signed_tri_areas::Vector{float_type} = sd[:area] .* sign(2,sd)
    return 1 ./ signed_tri_areas
end

"""
    dec_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge())

Gives the hodge matrix between `n`-simplices and dual 'n'-simplices.
"""
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

    evt = MVector{3, point_type}(undef)
    dvt = MVector{3, point_type}(undef)

    idx = 0

    @inbounds for t in triangles(sd)
        dual_point_tct = sd[sd[t, :tri_center], :dual_point]
        for i in 1:3
            tri_edge = tri_edges[i][t]
            evt[i] = sd[sd[tri_edge, :∂v0], :dual_point] - sd[sd[tri_edge, :∂v1], :dual_point]
            dvt[i] = dual_point_tct - sd[sd[tri_edge, :edge_center], :dual_point]
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

"""
    dec_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge())

Gives the inverse hodge matrix between dual `n`-simplices and 'n'-simplices.
"""
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

"""    function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 1-form, return their interior product as a dual 0-form.
"""
function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)
  ihs1 = dec_inv_hodge_star(Val{1}, s, GeometricHodge())
  Λ11 = dec_wedge_product_pd(Tuple{1,1}, s)
  hs2 = dec_hodge_star(Val{2}, s, GeometricHodge())

  (f,g) -> hs2 * Λ11(ihs1(g), f)
end

"""    function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 2-form, return their interior product as a dual 1-form.
"""
function interior_product_dd(::Type{Tuple{1,2}}, s::SimplicialSets.HasDeltaSet)
  ihs0 = dec_inv_hodge_star(Val{0}, s, GeometricHodge())
  hs1 = dec_hodge_star(Val{1}, s, GeometricHodge())
  ♭♯_m = ♭♯_mat(s)
  Λ01_m = wedge_pd_01_mat(s)
  (f,g) -> hs1 * ♭♯_m * ((Λ01_m * ihs0 * g) .* f)
end

"""    function ℒ_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 1-form, return their lie derivative as a dual 1-form.
"""
function ℒ_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)
  # ℒ := -diuv - iduv
  d0 = dec_dual_derivative(0, s)
  d1 = dec_dual_derivative(1, s)
  i1 = interior_product_dd(Tuple{1,1}, s)
  i2 = interior_product_dd(Tuple{1,2}, s)

  (f,g) ->
    -(d0 * i1(f,g)) -
      i2(f,d1 * g)
end

const lie_derivative_dd = ℒ_dd

"""    function Δᵈ_mat(::Type{Val{0}}, s::SimplicialSets.HasDeltaSet)

Return a function matrix encoding the dual 0-form Laplacian.
"""
function Δᵈ(::Type{Val{0}}, s::SimplicialSets.HasDeltaSet)
  dd0 = dec_dual_derivative(0, s);
  ihs1 = dec_inv_hodge_star(1, s, GeometricHodge());
  d1 = dec_differential(1,s);
  hs2 = dec_hodge_star(2, s, GeometricHodge());
  m = hs2 * d1
  x -> hs2 * d1 * ihs1(dd0 * x)
end

"""    function Δᵈ_mat(::Type{Val{2}}, s::SimplicialSets.HasDeltaSet)

Return a function matrix encoding the dual 1-form Laplacian.
"""
function Δᵈ(::Type{Val{1}}, s::SimplicialSets.HasDeltaSet)
  dd0 = dec_dual_derivative(0, s);
  ihs1 = dec_inv_hodge_star(1, s, GeometricHodge());
  d1 = dec_differential(1,s);
  hs2 = dec_hodge_star(2, s, GeometricHodge());
  dd1 = dec_dual_derivative(1, s);
  ihs0 = dec_inv_hodge_star(0, s, GeometricHodge());
  d0 = dec_differential(0,s);
  hs1 = dec_hodge_star(1, s, GeometricHodge());
  m = hs1 * d0 * ihs0 * dd1
  n = dd0 * hs2 * d1
  x -> begin
    m * x +
    n * ihs1(x)
  end
end

function avg₀₁_mat(s::HasDeltaSet, float_type)
  d0 = dec_differential(0,s)
  avg_mat = SparseMatrixCSC{float_type, Int32}(d0)
  avg_mat.nzval .= 0.5
  avg_mat
end

""" Averaging matrix from 0-forms to 1-forms.

Given a 0-form, this matrix computes a 1-form by taking the mean of value stored on the faces of each edge.

This matrix can be used to implement a wedge product: `(avg₀₁(s)*X) .* Y` where `X` is a 0-form and `Y` a 1-form, assuming the center of an edge is halfway between its endpoints.

See also [`avg₀₁`](@ref).
"""
avg₀₁_mat(s::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type =
  avg₀₁_mat(s, float_type)
avg₀₁_mat(s::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type =
  avg₀₁_mat(s, float_type)

"""    avg₀₁(s::HasDeltaSet, α::SimplexForm{0})

Turn a 0-form into a 1-form by averaging data stored on the face of an edge.

See also [`avg₀₁_mat`](@ref).
"""
avg₀₁(s::HasDeltaSet, α::SimplexForm{0}) = avg₀₁_mat(s) * α

""" Alias for the averaging operator [`avg₀₁`](@ref).
"""
const avg_01 = avg₀₁

""" Alias for the averaging matrix [`avg₀₁_mat`](@ref).
"""
const avg_01_mat = avg₀₁_mat

end
