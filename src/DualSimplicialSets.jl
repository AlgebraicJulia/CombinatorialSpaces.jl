""" Duality for simplicial sets in one, two, and three dimensions.
"""
module DualSimplicialSets
export AbstractDualDeltaSet1D, DualDeltaSet1D

using Catlab, Catlab.CategoricalAlgebra.CSets
using ..SimplicialSets

# 1D dual simplicial sets
#########################

@present DualDeltaCategory1D <: DeltaCategory1D begin
  # Dual vertices and edges.
  (DualV, DualE)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V,DualV)
  edge_center::Hom(E,DualV)

  # Dual edges associated with primal edges.
  (∂v0_dual, ∂v1_dual)::Hom(E,DualE)
  ∂v0_dual ⋅ D_∂v1 == tgt ⋅ vertex_center
  ∂v1_dual ⋅ D_∂v1 == src ⋅ vertex_center
  ∂v0_dual ⋅ D_∂v0 == edge_center
  ∂v1_dual ⋅ D_∂v0 == edge_center
end

const AbstractDualDeltaSet1D = AbstractACSetType(DualDeltaCategory1D)

const DualDeltaSet1D = CSetType(DualDeltaCategory1D,
                                index=[:src,:tgt,:D_∂v0,:D_∂v1])

""" Construct 1D dual delta set from 1D delta set.
"""
function (::Type{S})(t::AbstractDeltaSet1D) where S <: AbstractDualDeltaSet1D
  s = S()
  copy_parts!(s, t)
  make_dual_edges!(s)
  return s
end

""" Make dual edges in dual delta set of dimension ≧ 1.

Also creates the dual vertices. Note that although zero-dimensional duality is
geometrically trivial (subdividing a vertex gives back the same vertex), all
vertices in the dual complex are disjoint from the primal vertices.
"""
function make_dual_edges!(s::AbstractACSet)
  s[:vertex_center] = vertex_vs = add_parts!(s, :DualV, nv(s))
  s[:edge_center] = edge_vs = add_parts!(s, :DualV, ne(s))
  s[:∂v0_dual] = add_parts!(s, :DualE, ne(s);
    D_∂v0 = edge_vs, D_∂v1 = view(vertex_vs, ∂₁(0,s)))
  s[:∂v1_dual] = add_parts!(s, :DualE, ne(s);
    D_∂v0 = edge_vs, D_∂v1 = view(vertex_vs, ∂₁(1,s)))
end

end
