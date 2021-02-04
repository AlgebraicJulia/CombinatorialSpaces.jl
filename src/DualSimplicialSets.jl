""" Dual complexes for simplicial sets in one, two, and three dimensions.
"""
module DualSimplicialSets
export AbstractDeltaDualComplex1D, DeltaDualComplex1D,
  OrientedDeltaDualComplex1D,
  vertex_center, edge_center, elementary_duals

using StaticArrays: SVector

using Catlab, Catlab.CategoricalAlgebra.CSets
using ..SimplicialSets
using ..SimplicialSets: DeltaCategory1D, DeltaCategory2D

# 1D dual complex
#################

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

""" List of elementary dual simplices for primal simplex in 1D dual complex.

- elementary duals of primal vertices are dual edges
- elementary duals of primal edges are (single) dual vertices
"""
@inline elementary_duals(n::Int, s::AbstractDeltaDualComplex1D, args...) =
  elementary_duals(Val{n}, s, args...)

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
  make_dual_edges!(s)
  return s
end

""" Make dual edges in dual complex of dimension ≧ 1.

Also makes the dual vertices. Note that although zero-dimensional duality is
geometrically trivial (subdividing a vertex gives back the same vertex), the
dual vertices are disjoint from the primal vertices.

If the primal complex is oriented, an orientation is induced on the dual
complex. Dual vertices have no orientation. Dual edges are oriented relative to
the primal edges they subdivide (Hirani 2003, PhD thesis, Ch. 2, last sentence
of Remark 2.5.1).
"""
function make_dual_edges!(s::AbstractACSet)
  s[:vertex_center] = vcenters = add_parts!(s, :DualV, nv(s))
  s[:edge_center] = ecenters = add_parts!(s, :DualV, ne(s))
  D_edges0 = add_parts!(s, :DualE, ne(s);
                        D_∂v0 = ecenters, D_∂v1 = view(vcenters, ∂₁(0,s)))
  D_edges1 = add_parts!(s, :DualE, ne(s);
                        D_∂v0 = ecenters, D_∂v1 = view(vcenters, ∂₁(1,s)))
  if has_subpart(s, :edge_orientation)
    o = s[:edge_orientation]
    s[D_edges0, :D_edge_orientation] = negate.(o)
    s[D_edges1, :D_edge_orientation] = o
  end
end

negate(x) = -x
negate(x::Bool) = !x

end
