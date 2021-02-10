""" Dual complexes for simplicial sets in one, two, and three dimensions.
"""
module DualSimplicialSets
export DualSimplex, DualV, DualE, DualTri, elementary_duals,
  AbstractDeltaDualComplex1D, DeltaDualComplex1D, OrientedDeltaDualComplex1D,
  AbstractDeltaDualComplex2D, DeltaDualComplex2D, OrientedDeltaDualComplex2D,
  vertex_center, edge_center, triangle_center

using StaticArrays: @SVector, SVector

using Catlab, Catlab.CategoricalAlgebra.CSets
using ..ArrayUtils
using ..SimplicialSets
using ..SimplicialSets: DeltaCategory1D, DeltaCategory2D

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

@present SchemaOrientedDualComplex1D <: SchemaDualComplex1D begin
  Orientation::Data
  edge_orientation::Attr(E, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
end

""" Oriented dual complex of an oriented 1D delta set.
"""
const OrientedDeltaDualComplex1D = ACSetType(SchemaOrientedDualComplex1D,
                                             index=[:src,:tgt,:D_∂v0,:D_∂v1])

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
               D_∂v0 = ecenters, D_∂v1 = view(vcenters, ∂₁(i,s)))
  end

  # Orient elementary dual edges.
  if has_subpart(s, :edge_orientation)
    edge_orient = s[:edge_orientation]
    s[D_edges[1], :D_edge_orientation] = negate.(edge_orient)
    s[D_edges[2], :D_edge_orientation] = edge_orient
  end

  D_edges
end

negate(x) = -x
negate(x::Bool) = !x

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
               D_∂v0=tri_centers, D_∂v1=edge_center(s, ∂₂(e,s)))
  end
  D_edges02 = map((0,1,2)) do v
    add_parts!(s, :DualE, ntriangles(s);
               D_∂v0=tri_centers, D_∂v1=vertex_center(s, triangle_vertex(v,s)))
  end

  # Make dual triangles.
  # Counterclockwise order in drawing with vertices 0, 1, 2 from left to right.
  D_triangle_schemas = ((0,1,1),(0,2,1),(1,2,0),(1,0,1),(2,0,0),(2,1,0))
  D_triangles = map(D_triangle_schemas) do (v,e,ev)
    add_parts!(s, :DualTri, ntriangles(s);
               D_∂e0=D_edges12[e+1], D_∂e1=D_edges02[v+1],
               D_∂e2=view(D_edges01[ev+1], ∂₂(e,s)))
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
        s[∂₂(e,s), :edge_orientation],
        isodd(e) ? rev_tri_orient : tri_orient)
    end
    # Remaining dual edges are oriented arbitrarily.
    s[lazy(vcat, D_edges02...), :D_edge_orientation] = one(eltype(tri_orient))
  end

  D_triangles
end

relative_sign(x, y) = sign(x*y)
relative_sign(x::Bool, y::Bool) = (x && y) || (!x && !y)

# General operators
###################

""" Wrapper for dual simplex or simplices of dimension `D`.

See also: [`DualV`](@ref), [`DualE`](@ref), [`DualTri`](@ref).
"""
@parts_array DualSimplex{D}

""" Vertex in simplicial set: alias for `Simplex{0}`.
"""
const DualV = DualSimplex{0}

""" Edge in simplicial set: alias for `Simplex{1}`.
"""
const DualE = DualSimplex{1}

""" Triangle in simplicial set: alias for `Simplex{2}`.
"""
const DualTri = DualSimplex{2}

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
@inline elementary_duals(s::AbstractDeltaDualComplex1D, x::Simplex{n}) where n =
  DualSimplex{1-n}(elementary_duals(Val{n}, s, x.data))
@inline elementary_duals(s::AbstractDeltaDualComplex2D, x::Simplex{n}) where n =
  DualSimplex{2-n}(elementary_duals(Val{n}, s, x.data))
@inline elementary_duals(n::Int, s::AbstractACSet, args...) =
  elementary_duals(Val{n}, s, args...)

end
