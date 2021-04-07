""" Dual complexes for simplicial sets in one, two, and three dimensions.
"""
module DualSimplicialSets
export DualSimplex, DualV, DualE, DualTri, DualChain, DualForm, DualVectorField,
  AbstractDeltaDualComplex1D, DeltaDualComplex1D,
  OrientedDeltaDualComplex1D, EmbeddedDeltaDualComplex1D,
  AbstractDeltaDualComplex2D, DeltaDualComplex2D,
  OrientedDeltaDualComplex2D, EmbeddedDeltaDualComplex2D,
  SimplexCenter, Barycenter, Circumcenter, Incenter, geometric_center,
  subsimplices, primal_vertex, elementary_duals, dual_boundary, dual_derivative,
  ⋆, hodge_star, δ, codifferential, Δ, laplace_beltrami, ♭, flat,
  ∧, wedge_product, interior_product, interior_product_flat,
  lie_derivative, lie_derivative_flat,
  vertex_center, edge_center, triangle_center, dual_triangle_vertices,
  dual_point, dual_volume, subdivide_duals!

import Base: ndims
using LinearAlgebra: Diagonal, dot
using SparseArrays
using StaticArrays: @SVector, SVector

using Catlab, Catlab.CategoricalAlgebra.CSets
using Catlab.CategoricalAlgebra.FinSets: deleteat
using ..ArrayUtils, ..SimplicialSets
using ..SimplicialSets: DeltaCategory1D, DeltaCategory2D, CayleyMengerDet,
  operator_nz, ∂_nz, d_nz, cayley_menger, negate
import ..SimplicialSets: ∂, d, volume

abstract type DiscreteFlat end
struct DPPFlat <: DiscreteFlat end

# 1D dual complex
#################

# Should be expressed using a coproduct of two copies of `DeltaCategory1D`.

@present SchemaDualComplex1D <: DeltaCategory1D begin
  # Dual vertices and edges.
  (DualV, DualE)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V, DualV)
  edge_center::Hom(E, DualV)

  # Every primal edge is subdivided into two dual edges.
  #
  # (∂v0_dual, ∂v1_dual)::Hom(E,DualE)
  #
  # ∂v0_dual ⋅ D_∂v1 == tgt ⋅ vertex_center
  # ∂v1_dual ⋅ D_∂v1 == src ⋅ vertex_center
  # ∂v0_dual ⋅ D_∂v0 == edge_center
  # ∂v1_dual ⋅ D_∂v0 == edge_center
  #
  # We could, and arguably should, track these through dedicated morphisms, as
  # in the commented code above. We don't because it scales badly in dimension:
  # an ``n``-simplex is subdivided into ``n!`` sub-simplices. So we would need 6
  # morphisms for triangles and 24 for tets, and an even larger number of
  # equations to say how everything fits together. Moreover, in practice, the
  # incidence data for the dual complex suffices to construct the dual cells of
  # primal simplices.
end

""" Abstract type for dual complex of a 1D delta set.
"""
const AbstractDeltaDualComplex1D = AbstractACSetType(SchemaDualComplex1D)

""" Dual complex of a one-dimensional delta set.

The data structure includes both the primal complex and the dual complex, as
well as the mapping between them.
"""
const DeltaDualComplex1D = CSetType(SchemaDualComplex1D,
                                    index=[:src,:tgt,:D_∂v0,:D_∂v1])

""" Dual vertex corresponding to center of primal vertex.
"""
vertex_center(s::AbstractACSet, args...) = s[args..., :vertex_center]

""" Dual vertex corresponding to center of primal edge.
"""
edge_center(s::AbstractACSet, args...) = s[args..., :edge_center]

subsimplices(::Type{Val{1}}, s::AbstractACSet, e::Int) =
  SVector{2}(incident(s, edge_center(s, e), :D_∂v0))

primal_vertex(::Type{Val{1}}, s::AbstractACSet, e...) = s[e..., :D_∂v1]

elementary_duals(::Type{Val{0}}, s::AbstractDeltaDualComplex1D, v::Int) =
  incident(s, vertex_center(s,v), :D_∂v1)
elementary_duals(::Type{Val{1}}, s::AbstractDeltaDualComplex1D, e::Int) =
  SVector(edge_center(s,e))

""" Boundary dual vertices of a dual triangle.

This accessor assumes that the simplicial identities for the dual hold.
"""
function dual_triangle_vertices(s::AbstractACSet, t...)
  SVector(s[s[t..., :D_∂e1], :D_∂v1],
          s[s[t..., :D_∂e0], :D_∂v1],
          s[s[t..., :D_∂e0], :D_∂v0])
end

# 1D oriented dual complex
#-------------------------

@present SchemaOrientedDualComplex1D <: SchemaDualComplex1D begin
  Orientation::Data
  edge_orientation::Attr(E, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
end

""" Oriented dual complex of an oriented 1D delta set.
"""
const OrientedDeltaDualComplex1D = ACSetType(SchemaOrientedDualComplex1D,
                                             index=[:src,:tgt,:D_∂v0,:D_∂v1])

dual_boundary_nz(::Type{Val{1}}, s::AbstractDeltaDualComplex1D, x::Int) =
  # Boundary vertices of dual 1-cell ↔
  # Dual vertices for cofaces of (edges incident to) primal vertex.
  d_nz(Val{0}, s, x)

dual_derivative_nz(::Type{Val{0}}, s::AbstractDeltaDualComplex1D, x::Int) =
  negatenz(∂_nz(Val{1}, s, x))

negatenz((I, V)) = (I, negate.(V))

""" Construct 1D dual complex from 1D delta set.
"""
function (::Type{S})(t::AbstractDeltaSet1D) where S <: AbstractDeltaDualComplex1D
  s = S()
  copy_parts!(s, t)
  make_dual_simplices_1d!(s)
  return s
end

""" Make dual vertice and edges for dual complex of dimension ≧ 1.

Although zero-dimensional duality is geometrically trivial (subdividing a vertex
gives back the same vertex), we treat the dual vertices as disjoint from the
primal vertices. Thus, a dual vertex is created for every primal vertex.

If the primal complex is oriented, an orientation is induced on the dual
complex. The dual edges are oriented relative to the primal edges they subdivide
(Hirani 2003, PhD thesis, Ch. 2, last sentence of Remark 2.5.1).
"""
function make_dual_simplices_1d!(s::AbstractACSet)
  # Make dual vertices and edges.
  s[:vertex_center] = vcenters = add_parts!(s, :DualV, nv(s))
  s[:edge_center] = ecenters = add_parts!(s, :DualV, ne(s))
  D_edges = map((0,1)) do i
    add_parts!(s, :DualE, ne(s);
               D_∂v0 = ecenters, D_∂v1 = view(vcenters, ∂(1,i,s)))
  end

  # Orient elementary dual edges.
  if has_subpart(s, :edge_orientation)
    edge_orient = s[:edge_orientation]
    s[D_edges[1], :D_edge_orientation] = negate.(edge_orient)
    s[D_edges[2], :D_edge_orientation] = edge_orient
  end

  D_edges
end

# 1D embedded dual complex
#-------------------------

@present SchemaEmbeddedDualComplex1D <: SchemaOrientedDualComplex1D begin
  (Real, Point)::Data
  point::Attr(V, Point)
  length::Attr(E, Real)
  dual_point::Attr(DualV, Point)
  dual_length::Attr(DualE, Real)
end

""" Embedded dual complex of an embedded 1D delta set.

Although they are redundant information, the lengths of the primal and dual
edges are precomputed and stored.
"""
const EmbeddedDeltaDualComplex1D = ACSetType(SchemaEmbeddedDualComplex1D,
                                             index=[:src,:tgt,:D_∂v0,:D_∂v1])

""" Point associated with dual vertex of complex.
"""
dual_point(s::AbstractACSet, args...) = s[args..., :dual_point]

struct PrecomputedVol end

volume(::Type{Val{n}}, s::EmbeddedDeltaDualComplex1D, x) where n =
  volume(Val{n}, s, x, PrecomputedVol())
dual_volume(::Type{Val{n}}, s::EmbeddedDeltaDualComplex1D, x) where n =
  dual_volume(Val{n}, s, x, PrecomputedVol())

volume(::Type{Val{1}}, s::AbstractACSet, e, ::PrecomputedVol) = s[e, :length]
dual_volume(::Type{Val{1}}, s::AbstractACSet, e, ::PrecomputedVol) =
  s[e, :dual_length]

dual_volume(::Type{Val{1}}, s::AbstractACSet, e::Int, ::CayleyMengerDet) =
  volume(dual_point(s, SVector(s[e,:D_∂v0], s[e,:D_∂v1])))

hodge_diag(::Type{Val{0}}, s::AbstractDeltaDualComplex1D, v::Int) =
  sum(dual_volume(Val{1}, s, elementary_duals(Val{0},s,v)))
hodge_diag(::Type{Val{1}}, s::AbstractDeltaDualComplex1D, e::Int) =
  1 / volume(Val{1},s,e)

""" Compute geometric subdivision for embedded dual complex.

Supports different methods of subdivision through the choice of geometric
center, as defined by [`geometric_center`](@ref). In particular, barycentric
subdivision and circumcentric subdivision are supported.
"""
function subdivide_duals!(s::EmbeddedDeltaDualComplex1D, args...)
  subdivide_duals_1d!(s, args...)
  precompute_volumes_1d!(s)
end

function subdivide_duals_1d!(s::AbstractACSet, alg)
  for v in vertices(s)
    s[vertex_center(s,v), :dual_point] = point(s, v)
  end
  for e in edges(s)
    s[edge_center(s,e), :dual_point] = geometric_center(
      point(s, edge_vertices(s, e)), alg)
  end
end

function precompute_volumes_1d!(s::AbstractACSet)
  for e in edges(s)
    s[e, :length] = volume(1,s,e,CayleyMengerDet())
  end
  for e in parts(s, :DualE)
    s[e, :dual_length] = dual_volume(1,s,e,CayleyMengerDet())
  end
end

# 2D dual complex
#################

# Should be expressed using a coproduct of two copies of `DeltaCategory2D` or
# perhaps a pushout of `SchemaDualComplex2D` and `DeltaCategory1D`.

@present SchemaDualComplex2D <: DeltaCategory2D begin
  # Dual vertices, edges, and triangles.
  (DualV, DualE, DualTri)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)
  (D_∂e0, D_∂e1, D_∂e2)::Hom(DualTri, DualE)

  # Simplicial identities for dual simplices.
  D_∂e1 ⋅ D_∂v1 == D_∂e2 ⋅ D_∂v1
  D_∂e0 ⋅ D_∂v1 == D_∂e2 ⋅ D_∂v0
  D_∂e0 ⋅ D_∂v0 == D_∂e1 ⋅ D_∂v0

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V, DualV)
  edge_center::Hom(E, DualV)
  tri_center::Hom(Tri, DualV)
end

""" Abstract type for dual complex of a 2D delta set.
"""
const AbstractDeltaDualComplex2D = AbstractACSetType(SchemaDualComplex2D)

""" Dual complex of a two-dimensional delta set.
"""
const DeltaDualComplex2D = CSetType(SchemaDualComplex2D,
  index=[:src, :tgt, :∂e0, :∂e1, :∂e2, :D_∂v0, :D_∂v1, :D_∂e0, :D_∂e1, :D_∂e2])

""" Dual vertex corresponding to center of primal triangle.
"""
triangle_center(s::AbstractACSet, args...) = s[args..., :tri_center]

subsimplices(::Type{Val{2}}, s::AbstractACSet, t::Int) =
  SVector{6}(incident(s, triangle_center(s,t), @SVector [:D_∂e1, :D_∂v0]))

primal_vertex(::Type{Val{2}}, s::AbstractACSet, t...) =
  primal_vertex(Val{1}, s, s[t..., :D_∂e2])

elementary_duals(::Type{Val{0}}, s::AbstractDeltaDualComplex2D, v::Int) =
  incident(s, vertex_center(s,v), @SVector [:D_∂e1, :D_∂v1])
elementary_duals(::Type{Val{1}}, s::AbstractDeltaDualComplex2D, e::Int) =
  incident(s, edge_center(s,e), :D_∂v1)
elementary_duals(::Type{Val{2}}, s::AbstractDeltaDualComplex2D, t::Int) =
  SVector(triangle_center(s,t))

# 2D oriented dual complex
#-------------------------

@present SchemaOrientedDualComplex2D <: SchemaDualComplex2D begin
  Orientation::Data
  edge_orientation::Attr(E, Orientation)
  tri_orientation::Attr(Tri, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
  D_tri_orientation::Attr(DualTri, Orientation)
end

""" Oriented dual complex of an oriented 2D delta set.
"""
const OrientedDeltaDualComplex2D = ACSetType(SchemaOrientedDualComplex2D,
  index=[:src, :tgt, :∂e0, :∂e1, :∂e2, :D_∂v0, :D_∂v1, :D_∂e0, :D_∂e1, :D_∂e2])

dual_boundary_nz(::Type{Val{1}}, s::AbstractDeltaDualComplex2D, x::Int) =
  # Boundary vertices of dual 1-cell ↔
  # Dual vertices for cofaces of (triangles incident to) primal edge.
  negatenz(d_nz(Val{1}, s, x))
dual_boundary_nz(::Type{Val{2}}, s::AbstractDeltaDualComplex2D, x::Int) =
  # Boundary edges of dual 2-cell ↔
  # Dual edges for cofaces of (edges incident to) primal vertex.
  d_nz(Val{0}, s, x)

dual_derivative_nz(::Type{Val{0}}, s::AbstractDeltaDualComplex2D, x::Int) =
  ∂_nz(Val{2}, s, x)
dual_derivative_nz(::Type{Val{1}}, s::AbstractDeltaDualComplex2D, x::Int) =
  negatenz(∂_nz(Val{1}, s, x))

""" Construct 2D dual complex from 2D delta set.
"""
function (::Type{S})(t::AbstractDeltaSet2D) where S <: AbstractDeltaDualComplex2D
  s = S()
  copy_parts!(s, t)
  make_dual_simplices_2d!(s)
  return s
end

""" Make dual simplices for dual complex of dimension ≧ 2.

If the primal complex is oriented, an orientation is induced on the dual
complex. The elementary dual edges are oriented following (Hirani, 2003, Example
2.5.2) or (Desbrun et al, 2005, Table 1) and the dual triangles are oriented
relative to the primal triangles they subdivide.
"""
function make_dual_simplices_2d!(s::AbstractACSet)
  # Make dual vertices and edges.
  D_edges01 = make_dual_simplices_1d!(s)
  s[:tri_center] = tri_centers = add_parts!(s, :DualV, ntriangles(s))
  D_edges12 = map((0,1,2)) do e
    add_parts!(s, :DualE, ntriangles(s);
               D_∂v0=tri_centers, D_∂v1=edge_center(s, ∂(2,e,s)))
  end
  D_edges02 = map(triangle_vertices(s)) do vs
    add_parts!(s, :DualE, ntriangles(s);
               D_∂v0=tri_centers, D_∂v1=vertex_center(s, vs))
  end

  # Make dual triangles.
  # Counterclockwise order in drawing with vertices 0, 1, 2 from left to right.
  D_triangle_schemas = ((0,1,1),(0,2,1),(1,2,0),(1,0,1),(2,0,0),(2,1,0))
  D_triangles = map(D_triangle_schemas) do (v,e,ev)
    add_parts!(s, :DualTri, ntriangles(s);
               D_∂e0=D_edges12[e+1], D_∂e1=D_edges02[v+1],
               D_∂e2=view(D_edges01[ev+1], ∂(2,e,s)))
  end

  if has_subpart(s, :tri_orientation)
    # Orient elementary dual triangles.
    tri_orient = s[:tri_orientation]
    rev_tri_orient = negate.(tri_orient)
    for (i, D_tris) in enumerate(D_triangles)
      s[D_tris, :D_tri_orientation] = isodd(i) ? rev_tri_orient : tri_orient
    end

    # Orient elementary dual edges.
    for e in (0,1,2)
      s[D_edges12[e+1], :D_edge_orientation] = relative_sign.(
        s[∂(2,e,s), :edge_orientation],
        isodd(e) ? rev_tri_orient : tri_orient)
    end
    # Remaining dual edges are oriented arbitrarily.
    s[lazy(vcat, D_edges02...), :D_edge_orientation] = one(eltype(tri_orient))
  end

  D_triangles
end

relative_sign(x, y) = sign(x*y)
relative_sign(x::Bool, y::Bool) = (x && y) || (!x && !y)

# 2D embedded dual complex
#-------------------------

@present SchemaEmbeddedDualComplex2D <: SchemaOrientedDualComplex2D begin
  (Real, Point)::Data
  point::Attr(V, Point)
  length::Attr(E, Real)
  area::Attr(Tri, Real)
  dual_point::Attr(DualV, Point)
  dual_length::Attr(DualE, Real)
  dual_area::Attr(DualTri, Real)
end

""" Embedded dual complex of an embedded 12 delta set.

Although they are redundant information, the lengths and areas of the
primal/dual edges and triangles are precomputed and stored.
"""
const EmbeddedDeltaDualComplex2D = ACSetType(SchemaEmbeddedDualComplex2D,
  index=[:src, :tgt, :∂e0, :∂e1, :∂e2, :D_∂v0, :D_∂v1, :D_∂e0, :D_∂e1, :D_∂e2])

volume(::Type{Val{n}}, s::EmbeddedDeltaDualComplex2D, x) where n =
  volume(Val{n}, s, x, PrecomputedVol())
dual_volume(::Type{Val{n}}, s::EmbeddedDeltaDualComplex2D, x) where n =
  dual_volume(Val{n}, s, x, PrecomputedVol())

volume(::Type{Val{2}}, s::AbstractACSet, t, ::PrecomputedVol) = s[t, :area]
dual_volume(::Type{Val{2}}, s::AbstractACSet, t, ::PrecomputedVol) =
  s[t, :dual_area]

function dual_volume(::Type{Val{2}}, s::AbstractACSet, t::Int, ::CayleyMengerDet)
  dual_vs = SVector(s[s[t, :D_∂e1], :D_∂v1],
                    s[s[t, :D_∂e2], :D_∂v0],
                    s[s[t, :D_∂e0], :D_∂v0])
  volume(dual_point(s, dual_vs))
end

hodge_diag(::Type{Val{0}}, s::AbstractDeltaDualComplex2D, v::Int) =
  sum(dual_volume(Val{2}, s, elementary_duals(Val{0},s,v)))
hodge_diag(::Type{Val{1}}, s::AbstractDeltaDualComplex2D, e::Int) =
  sum(dual_volume(Val{1}, s, elementary_duals(Val{1},s,e))) / volume(Val{1},s,e)
hodge_diag(::Type{Val{2}}, s::AbstractDeltaDualComplex2D, t::Int) =
  1 / volume(Val{2},s,t)

function ♭(s::AbstractDeltaDualComplex2D, X::AbstractVector, ::DPPFlat)
  # XXX: Creating this lookup table shouldn't be necessary. Of course, we could
  # index `tri_center` but that shouldn't be necessary either. Rather, we should
  # loop over incident triangles instead of the elementary duals, which just
  # happens to be inconvenient.
  tri_map = Dict{Int,Int}(triangle_center(s,t) => t for t in triangles(s))

  map(edges(s)) do e
    e_vec = (point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
    dual_edges = elementary_duals(1,s,e)
    dual_lengths = dual_volume(1, s, dual_edges)
    mapreduce(+, dual_edges, dual_lengths) do dual_e, dual_length
      X_vec = X[tri_map[s[dual_e, :D_∂v0]]]
      dual_length * dot(X_vec, e_vec)
    end / sum(dual_lengths)
  end
end

function ∧(::Type{Tuple{1,1}}, s::AbstractACSet, α, β, x::Int)
  # XXX: This calculation of the volume coefficients is awkward due to the
  # design decision described in `SchemaDualComplex1D`.
  dual_vs = vertex_center(s, triangle_vertices(s, x))
  dual_es = sort(SVector{6}(incident(s, triangle_center(s, x), :D_∂v0)),
                 by=e -> s[e,:D_∂v1] .== dual_vs, rev=true)[1:3]
  coeffs = map(dual_es) do e
    sum(dual_volume(2, s, SVector{2}(incident(s, e, :D_∂e1))))
  end / volume(2, s, x)

  # Wedge product of two primal 1-forms, as in (Hirani 2003, Example 7.1.2).
  # This formula is not the same as (Hirani 2003, Equation 7.1.2) but it is
  # equivalent.
  e0, e1, e2 = ∂(2,0,s,x), ∂(2,1,s,x), ∂(2,2,s,x)
  dot(coeffs, SVector(α[e2] * β[e1] - α[e1] * β[e2],
                      α[e2] * β[e0] - α[e0] * β[e2],
                      α[e1] * β[e0] - α[e0] * β[e1])) / 2
end

function subdivide_duals!(s::EmbeddedDeltaDualComplex2D, args...)
  subdivide_duals_2d!(s, args...)
  precompute_volumes_2d!(s)
end

function subdivide_duals_2d!(s::AbstractACSet, alg)
  subdivide_duals_1d!(s, alg)
  for t in triangles(s)
    s[triangle_center(s,t), :dual_point] = geometric_center(
      point(s, triangle_vertices(s, t)), alg)
  end
end

function precompute_volumes_2d!(s::AbstractACSet)
  precompute_volumes_1d!(s)
  for t in triangles(s)
    s[t, :area] = volume(2,s,t,CayleyMengerDet())
  end
  for t in parts(s, :DualTri)
    s[t, :dual_area] = dual_volume(2,s,t,CayleyMengerDet())
  end
end

# General operators
###################

""" Wrapper for dual simplex or simplices of dimension `D`.

See also: [`DualV`](@ref), [`DualE`](@ref), [`DualTri`](@ref).
"""
@parts_array_struct DualSimplex{D}

""" Vertex in simplicial set: alias for `Simplex{0}`.
"""
const DualV = DualSimplex{0}

""" Edge in simplicial set: alias for `Simplex{1}`.
"""
const DualE = DualSimplex{1}

""" Triangle in simplicial set: alias for `Simplex{2}`.
"""
const DualTri = DualSimplex{2}

""" Wrapper for chain of dual cells of dimension `n`.

In an ``N``-dimensional complex, the elementary dual simplices of each
``n``-simplex together comprise the dual ``(N-n)``-cell of the simplex. Using
this correspondence, a basis for primal ``n``-chains defines the basis for dual
``(N-n)``-chains.

!!! note

    In (Hirani 2003, Definition 3.4.1), the duality operator assigns a certain
    sign to each elementary dual simplex. For us, all of these signs should be
    regarded as positive because we have already incorporated them into the
    orientation of the dual simplices.
"""
@vector_struct DualChain{n}

""" Wrapper for form, aka cochain, on dual cells of dimension `n`.
"""
@vector_struct DualForm{n}

""" Wrapper for vector field on dual vertices.
"""
@vector_struct DualVectorField

ndims(s::AbstractDeltaDualComplex1D) = 1
ndims(s::AbstractDeltaDualComplex2D) = 2

volume(s::AbstractACSet, x::DualSimplex{n}, args...) where n =
  dual_volume(Val{n}, s, x.data, args...)
@inline dual_volume(n::Int, s::AbstractACSet, args...) =
  dual_volume(Val{n}, s, args...)

""" List of dual simplices comprising the subdivision of a primal simplex.

A primal ``n``-simplex is always subdivided into ``n!`` dual ``n``-simplices,
not be confused with the [`elementary_duals`](@ref) which have complementary
dimension.

The returned list is ordered such that subsimplices with the same primal vertex
appear consecutively.
"""
subsimplices(s::AbstractACSet, x::Simplex{n}) where n =
  DualSimplex{n}(subsimplices(Val{n}, s, x.data))
@inline subsimplices(n::Int, s::AbstractACSet, args...) =
  subsimplices(Val{n}, s, args...)

""" Primal vertex associated with a dual simplex.
"""
primal_vertex(s::AbstractACSet, x::DualSimplex{n}) where n =
  V(primal_vertex(Val{n}, s, x.data))
@inline primal_vertex(n::Int, s::AbstractACSet, args...) =
  primal_vertex(Val{n}, s, args...)

""" List of elementary dual simplices corresponding to primal simplex.

In general, in an ``n``-dimensional complex, the elementary duals of primal
``k``-simplices are dual ``(n-k)``-simplices. Thus, in 1D dual complexes, the
elementary duals of...

- primal vertices are dual edges
- primal edges are (single) dual vertices

In 2D dual complexes, the elementary duals of...

- primal vertices are dual triangles
- primal edges are dual edges
- primal triangles are (single) dual triangles
"""
elementary_duals(s::AbstractACSet, x::Simplex{n}) where n =
  DualSimplex{ndims(s)-n}(elementary_duals(Val{n}, s, x.data))
@inline elementary_duals(n::Int, s::AbstractACSet, args...) =
  elementary_duals(Val{n}, s, args...)

""" Boundary of chain of dual cells.

Transpose of [`dual_derivative`](@ref).
"""
@inline dual_boundary(n::Int, s::AbstractACSet, args...) =
  dual_boundary(Val{n}, s, args...)
∂(s::AbstractACSet, x::DualChain{n}) where n =
  DualChain{n-1}(dual_boundary(Val{n}, s, x.data))

function dual_boundary(::Type{Val{n}}, s::AbstractACSet, args...) where n
  operator_nz(Int, nsimplices(ndims(s)-n+1,s),
              nsimplices(ndims(s)-n,s), args...) do x
    dual_boundary_nz(Val{n}, s, x)
  end
end

""" Discrete exterior derivative of dual form.

Transpose of [`dual_boundary`](@ref). For more info, see (Desbrun, Kanso, Tong,
2008: Discrete differential forms for computational modeling, §4.5).
"""
@inline dual_derivative(n::Int, s::AbstractACSet, args...) =
  dual_derivative(Val{n}, s, args...)
d(s::AbstractACSet, x::DualForm{n}) where n =
  DualForm{n+1}(dual_derivative(Val{n}, s, x.data))

function dual_derivative(::Type{Val{n}}, s::AbstractACSet, args...) where n
  operator_nz(Int, nsimplices(ndims(s)-n-1,s),
              nsimplices(ndims(s)-n,s), args...) do x
    dual_derivative_nz(Val{n}, s, x)
  end
end

""" Hodge star operator from primal ``n``-forms to dual ``N-n``-forms.

!!! warning

    Some authors, such as (Hirani 2003) and (Desbrun 2005), use the symbol ``⋆``
    for the duality operator on chains and the symbol ``*`` for the Hodge star
    operator on cochains. We do not explicitly define the duality operator and
    we use the symbol ``⋆`` for the Hodge star.
"""
⋆(s::AbstractACSet, x::SimplexForm{n}) where n =
  DualForm{ndims(s)-n}(⋆(Val{n}, s, x.data))
@inline ⋆(n::Int, s::AbstractACSet, args...) = ⋆(Val{n}, s, args...)

⋆(::Type{Val{n}}, s::AbstractACSet, form::AbstractVector) where n =
  applydiag(form) do x, a; a * hodge_diag(Val{n},s,x) end
⋆(::Type{Val{n}}, s::AbstractACSet) where n =
  Diagonal([ hodge_diag(Val{n},s,x) for x in simplices(n,s) ])

""" Alias for the Hodge star operator [`⋆`](@ref).
"""
const hodge_star = ⋆

""" Inverse Hodge star operator from dual ``N-n``-forms to primal ``n``-forms.

Confusingly, this is *not* the operator inverse of the Hodge star [`⋆`](@ref)
because it carries an extra global sign, in analogy to the smooth case
(Gillette, 2009, Notes on the DEC, Definition 2.27).
"""
@inline inv_hodge_star(n::Int, s::AbstractACSet, args...) =
  inv_hodge_star(Val{n}, s, args...)

function inv_hodge_star(::Type{Val{n}}, s::AbstractACSet,
                        form::AbstractVector) where n
  if iseven(n*(ndims(s)-n))
    applydiag(form) do x, a; a / hodge_diag(Val{n},s,x) end
  else
    applydiag(form) do x, a; -a / hodge_diag(Val{n},s,x) end
  end
end

""" Codifferential operator from primal ``n`` forms to primal ``n-1``-forms.
"""
δ(s::AbstractACSet, x::SimplexForm{n}) where n =
  SimplexForm{n-1}(δ(Val{n}, s, x.data))
@inline δ(n::Int, s::AbstractACSet, args...) = δ(Val{n}, s, args...)

function δ(::Type{Val{n}}, s::AbstractACSet, args...) where n
  # TODO: What is the right global sign?
  # sgn = iseven((n-1)*(ndims(s)-n+1)) ? +1 : -1
  operator_nz(Float64, nsimplices(n-1,s), nsimplices(n,s), args...) do x
    c = hodge_diag(Val{n}, s, x)
    I, V = dual_derivative_nz(Val{ndims(s)-n}, s, x)
    V = map(I, V) do i, a
      c * a / hodge_diag(Val{n-1}, s, i)
    end
    (I, V)
  end
end

""" Alias for the codifferential operator [`δ`](@ref).
"""
const codifferential = δ

""" Laplace-Beltrami operator on discrete forms.

The linear operator on primal ``n``-forms defined by ``Δ f := δ d f``, where
[`δ`](@ref) is the codifferential and [`d`](@ref) is the exterior derivative.
"""
Δ(s::AbstractACSet, x::SimplexForm{n}) where n =
  SimplexForm{n}(Δ(Val{n}, s, x.data))
@inline Δ(n::Int, s::AbstractACSet, args...) = Δ(Val{n}, s, args...)

Δ(::Type{Val{n}}, s::AbstractACSet, form::AbstractVector) where n =
  δ(Val{n+1}, s, d(Val{n}, s, form))
Δ(::Type{Val{n}}, s::AbstractACSet, Mat::Type=SparseMatrixCSC{Float64}) where n =
  δ(Val{n+1}, s, Mat) * d(Val{n}, s, Mat)

""" Alias for the Laplace-Beltrami operator [`Δ`](@ref).
"""
const laplace_beltrami = Δ

""" Flat operator converting vector fields to 1-forms.

A generic function for the discrete flat operators proposed by (Hirani 2003).
Currently only the DPP-flat is implemented.
"""
♭(s::AbstractACSet, X::DualVectorField) = EForm(♭(s, X.data, DPPFlat()))

""" Alias for the flat operator [`♭`](@ref).
"""
const flat = ♭

""" Wedge product of discrete forms.

The wedge product of a ``k``-form and an ``l``-form is a ``(k+l)``-form.

The DEC and related systems have several flavors of wedge product. This one is
the discrete primal-primal wedge product introduced in (Hirani, 2003, Chapter 7)
and (Desbrun et al 2005, Section 8). It depends on the geometric embedding and
requires the dual complex.
"""
∧(s::AbstractACSet, α::SimplexForm{k}, β::SimplexForm{l}) where {k,l} =
  SimplexForm{k+l}(∧(Tuple{k,l}, s, α.data, β.data))
@inline ∧(k::Int, l::Int, s::AbstractACSet, args...) = ∧(Tuple{k,l}, s, args...)

function ∧(::Type{Tuple{k,l}}, s::AbstractACSet, α, β) where {k,l}
  map(simplices(k+l, s)) do x
    ∧(Tuple{k,l}, s, α, β, x)
  end
end

∧(::Type{Tuple{0,0}}, s::AbstractACSet, f, g, x::Int) = f[x]*g[x]
∧(::Type{Tuple{k,0}}, s::AbstractACSet, α, g, x::Int) where k =
  wedge_product_zero(Val{k}, s, g, α, x)
∧(::Type{Tuple{0,k}}, s::AbstractACSet, f, β, x::Int) where k =
  wedge_product_zero(Val{k}, s, f, β, x)

""" Wedge product of a 0-form and a ``k``-form.
"""
function wedge_product_zero(::Type{Val{k}}, s::AbstractACSet,
                            f, α, x::Int) where k
  subs = subsimplices(k, s, x)
  vs = primal_vertex(k, s, subs)
  coeffs = map(x′ -> dual_volume(k,s,x′), subs) / volume(k,s,x)
  dot(coeffs, f[vs]) * α[x] / factorial(k)
end

""" Alias for the wedge product operator [`∧`](@ref).
"""
const wedge_product = ∧

""" Interior product of a vector field (or 1-form) and a ``n``-form.

Specifically, this operation is the primal-dual interior product defined in
(Hirani 2003, Section 8.2) and (Desbrun et al 2005, Section 10). Thus it takes a
primal vector field (or primal 1-form) and a dual ``n``-forms and then returns a
dual ``(n-1)``-form.
"""
interior_product(s::AbstractACSet, X♭::EForm, α::DualForm{n}) where n =
  DualForm{n-1}(interior_product_flat(Val{n}, s, X♭.data, α.data))

""" Interior product of a 1-form and a ``n``-form, yielding an ``(n-1)``-form.

Usually, the interior product is defined for vector fields; this function
assumes that the flat operator [`♭`](@ref) (not yet implemented for primal
vector fields) has already been applied to yield a 1-form.
"""
@inline interior_product_flat(n::Int, s::AbstractACSet, args...) =
  interior_product_flat(Val{n}, s, args...)

function interior_product_flat(::Type{Val{n}}, s::AbstractACSet,
                               X♭::AbstractVector, α::AbstractVector) where n
  # TODO: Global sign `iseven(n*n′) ? +1 : -1`
  n′ = ndims(s) - n
  hodge_star(n′+1,s, wedge_product(n′,1,s, inv_hodge_star(n′,s, α), X♭))
end

""" Lie derivative of ``n``-form with respect to a vector field (or 1-form).

Specifically, this is the primal-dual Lie derivative defined in (Hirani 2003,
Section 8.4) and (Desbrun et al 2005, Section 10).
"""
lie_derivative(s::AbstractACSet, X♭::EForm, α::DualForm{n}) where n =
  DualForm{n}(lie_derivative_flat(Val{n}, s, X♭, α.data))

""" Lie derivative of ``n``-form with respect to a 1-form.

Assumes that the flat operator [`♭`](@ref) has already been applied to the
vector field.
"""
@inline lie_derivative_flat(n::Int, s::AbstractACSet, args...) =
  lie_derivative_flat(Val{n}, s, args...)

function lie_derivative_flat(::Type{Val{n}}, s::AbstractACSet,
                             X♭::AbstractVector, α::AbstractVector) where n
  interior_product_flat(n+1, s, X♭, dual_derivative(n, s, α)) +
    dual_derivative(n-1, s, interior_product_flat(n, s, X♭, α))
end

# Euclidean geometry
####################

""" A notion of "geometric center" of a simplex.

See also: [`geometric_center`](@ref).
"""
abstract type SimplexCenter end

""" Calculate the center of simplex spanned by given points.

The first argument is a list of points and the second specifies the notion of
"center", via an instance of [`SimplexCenter`](@ref).
"""
function geometric_center end

""" Barycenter, aka centroid, of a simplex.
"""
struct Barycenter <: SimplexCenter end

function geometric_center(points, ::Barycenter)
  sum(points) / length(points)
end

""" Circumcenter, or center of circumscribed circle, of a simplex.

The circumcenter is calculated by inverting the Cayley-Menger matrix, as
explained by
[Westdendorp](https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm).
This method of calculation is also used in the package
[AlphaShapes.jl](https://github.com/harveydevereux/AlphaShapes.jl).
"""
struct Circumcenter <: SimplexCenter end

function geometric_center(points, ::Circumcenter)
  CM = cayley_menger(points...)
  barycentric_coords = inv(CM)[1,2:end]
  mapreduce(*, +, barycentric_coords, points)
end

""" Incenter, or center of inscribed circle, of a simplex.
"""
struct Incenter <: SimplexCenter end

function geometric_center(points, ::Incenter)
  length(points) > 2 || return geometric_center(points, Barycenter())
  face_volumes = map(i -> volume(deleteat(points, i)), eachindex(points))
  barycentric_coords = face_volumes / sum(face_volumes)
  mapreduce(*, +, barycentric_coords, points)
end

end
