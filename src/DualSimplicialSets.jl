""" Dual complexes for simplicial sets in one, two, and three dimensions.
"""
module DualSimplicialSets
export DualSimplex, DualV, DualE, DualTri, DualChain, DualForm,
  AbstractDeltaDualComplex1D, DeltaDualComplex1D,
  OrientedDeltaDualComplex1D, EmbeddedDeltaDualComplex1D,
  AbstractDeltaDualComplex2D, DeltaDualComplex2D,
  OrientedDeltaDualComplex2D, EmbeddedDeltaDualComplex2D,
  SimplexCenter, Barycenter, Circumcenter, Incenter, geometric_center,
  elementary_duals, dual_boundary, dual_derivative,
  ⋆, hodge_star, δ, codifferential, Δ, laplace_beltrami,
  vertex_center, edge_center, triangle_center, dual_triangle_vertices,
  dual_point, dual_volume, subdivide_duals!

import Base: ndims
using LinearAlgebra: Diagonal
using SparseArrays
using StaticArrays: @SVector, SVector

using Catlab, Catlab.CategoricalAlgebra.CSets
using Catlab.CategoricalAlgebra.FinSets: deleteat
using ..ArrayUtils, ..SimplicialSets
using ..SimplicialSets: DeltaCategory1D, DeltaCategory2D, CayleyMengerDet,
  operator_nz, ∂_nz, d_nz, cayley_menger, negate
import ..SimplicialSets: ∂, d, volume

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

elementary_duals(::Type{Val{0}}, s::AbstractDeltaDualComplex1D, v::Int) =
  incident(s, vertex_center(s,v), :D_∂v1)
elementary_duals(::Type{Val{1}}, s::AbstractDeltaDualComplex1D, e::Int) =
  SVector(edge_center(s,e))

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

elementary_duals(::Type{Val{0}}, s::AbstractDeltaDualComplex2D, v::Int) =
  incident(s, vertex_center(s,v), @SVector [:D_∂e1, :D_∂v1])
elementary_duals(::Type{Val{1}}, s::AbstractDeltaDualComplex2D, e::Int) =
  incident(s, edge_center(s,e), :D_∂v1)
elementary_duals(::Type{Val{2}}, s::AbstractDeltaDualComplex2D, t::Int) =
  SVector(triangle_center(s,t))

""" Boundary dual vertices of a  dual triangle

This accessor assumes that the simplicial identities for the dual hold
"""
function dual_triangle_vertices(s::AbstractDeltaDualComplex2D,t...)
  SVector(s[s[t..., :D_∂e1], :D_∂v1], s[s[t..., :D_∂e0], :D_∂v1], s[s[t..., :D_∂e0], :D_∂v0])
end

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

ndims(s::AbstractDeltaDualComplex1D) = 1
ndims(s::AbstractDeltaDualComplex2D) = 2

volume(s::AbstractACSet, x::DualSimplex{n}, args...) where n =
  dual_volume(Val{n}, s, x.data, args...)
@inline dual_volume(n::Int, s::AbstractACSet, args...) =
  dual_volume(Val{n}, s, args...)

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
  applydiag(form) do x; hodge_diag(Val{n},s,x) end
⋆(::Type{Val{n}}, s::AbstractACSet) where n =
  Diagonal([ hodge_diag(Val{n},s,x) for x in simplices(n,s) ])

""" Alias for the Hodge star operator [`⋆`](@ref).
"""
const hodge_star = ⋆

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
