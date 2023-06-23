""" Simplicial sets in one, two, and three dimensions.

For the time being, this module provides data structures only for [delta
sets](https://en.wikipedia.org/wiki/Delta_set), also known as [semi-simplicial
sets](https://ncatlab.org/nlab/show/semi-simplicial+set). These include the face
maps but not the degeneracy maps of a simplicial set. In the future we may add
support for simplicial sets. The analogy to keep in mind is that graphs are to
semi-simpicial sets as reflexive graphs are to simplicial sets.

Also provided are the fundamental operators on simplicial sets used in nearly
all geometric applications, namely the boundary and coboundary (discrete
exterior derivative) operators. For additional operators, see the
`DiscreteExteriorCalculus` module.
"""
module SimplicialSets
export Simplex, V, E, Tri, Tet, SimplexChain, VChain, EChain, TriChain, TetChain,
  SimplexForm, VForm, EForm, TriForm, TetForm, HasDeltaSet,
  HasDeltaSet1D, AbstractDeltaSet1D, DeltaSet1D, SchDeltaSet1D,
  OrientedDeltaSet1D, SchOrientedDeltaSet1D,
  EmbeddedDeltaSet1D, SchEmbeddedDeltaSet1D,
  HasDeltaSet2D, AbstractDeltaSet2D, DeltaSet2D, SchDeltaSet2D,
  OrientedDeltaSet2D, SchOrientedDeltaSet2D,
  EmbeddedDeltaSet2D, SchEmbeddedDeltaSet2D,
  HasDeltaSet3D, AbstractDeltaSet3D, DeltaSet3D, SchDeltaSet3D,
  OrientedDeltaSet3D, SchOrientedDeltaSet3D,
  EmbeddedDeltaSet3D, SchEmbeddedDeltaSet3D,
  ∂, boundary, coface, d, coboundary, exterior_derivative,
  simplices, nsimplices, point, volume,
  orientation, set_orientation!, orient!, orient_component!,
  src, tgt, nv, ne, vertices, edges, has_vertex, has_edge, edge_vertices,
  add_vertex!, add_vertices!, add_edge!, add_edges!,
  add_sorted_edge!, add_sorted_edges!,
  triangle_edges, triangle_vertices, ntriangles, triangles,
  add_triangle!, glue_triangle!, glue_sorted_triangle!,
  tetrahedron_triangles, tetrahedron_edges, tetrahedron_vertices, ntetrahedra,
  tetrahedra, add_tetrahedron!, glue_tetrahedron!, glue_sorted_tetrahedron!,
  is_manifold_like, nonboundaries, glue_sorted_tet_cube!

using LinearAlgebra: det
using SparseArrays
using StaticArrays: @SVector, SVector, SMatrix

using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra, Catlab.Graphs
import Catlab.Graphs: src, tgt, nv, ne, vertices, edges, has_vertex, has_edge,
  add_vertex!, add_vertices!, add_edge!, add_edges!
using ..ArrayUtils

# 0-D simplicial sets
#####################

@present SchDeltaSet0D(FreeSchema) begin
  V::Ob
end

""" Abstract type for C-sets that contain a delta set of some dimension.

This dimension could be zero, in which case the delta set consists only of
vertices (0-simplices).
"""
@abstract_acset_type HasDeltaSet

vertices(s::HasDeltaSet) = parts(s, :V)
nv(s::HasDeltaSet) = nparts(s, :V)
nsimplices(::Type{Val{0}}, s::HasDeltaSet) = nv(s)

has_vertex(s::HasDeltaSet, v) = has_part(s, :V, v)
add_vertex!(s::HasDeltaSet; kw...) = add_part!(s, :V; kw...)
add_vertices!(s::HasDeltaSet, n::Int; kw...) = add_parts!(s, :V, n; kw...)

# 1D simplicial sets
####################

@present SchDeltaSet1D <: SchDeltaSet0D begin
  E::Ob
  (∂v0, ∂v1)::Hom(E, V) # (∂₁(0), ∂₁(1))
end

""" Abstract type for C-sets that contain a one-dimensional delta set.
"""
@abstract_acset_type HasDeltaSet1D <: HasDeltaSet

""" Abstract type for one-dimensional delta sets, aka semi-simplicial sets.
"""
@abstract_acset_type AbstractDeltaSet1D <: HasDeltaSet1D

""" A one-dimensional delta set, aka semi-simplicial set.

Delta sets in 1D are isomorphic to graphs (in the category theorist's sense).
The source and target of an edge can be accessed using the face maps [`∂`](@ref)
(simplicial terminology) or `src` and `tgt` maps (graph-theoretic terminology).
More generally, this type implements the graphs interface in `Catlab.Graphs`.
"""
@acset_type DeltaSet1D(SchDeltaSet1D, index=[:∂v0,:∂v1]) <: AbstractDeltaSet1D

edges(s::HasDeltaSet1D) = parts(s, :E)
edges(s::HasDeltaSet1D, src::Int, tgt::Int) =
  (e for e in coface(1,1,s,src) if ∂(1,0,s,e) == tgt)

ne(s::HasDeltaSet1D) = nparts(s, :E)
nsimplices(::Type{Val{1}}, s::HasDeltaSet1D) = ne(s)

has_edge(s::HasDeltaSet1D, e) = has_part(s, :E, e)
has_edge(s::HasDeltaSet1D, src::Int, tgt::Int) =
  has_vertex(s, src) && any(e -> ∂(1,0,s,e) == tgt, coface(1,1,s,src))

src(s::HasDeltaSet1D, args...) = subpart(s, args..., :∂v1)
tgt(s::HasDeltaSet1D, args...) = subpart(s, args..., :∂v0)
face(::Type{Val{(1,0)}}, s::HasDeltaSet1D, args...) = subpart(s, args..., :∂v0)
face(::Type{Val{(1,1)}}, s::HasDeltaSet1D, args...) = subpart(s, args..., :∂v1)

coface(::Type{Val{(1,0)}}, s::HasDeltaSet1D, args...) = incident(s, args..., :∂v0)
coface(::Type{Val{(1,1)}}, s::HasDeltaSet1D, args...) = incident(s, args..., :∂v1)

""" Boundary vertices of an edge.
"""
edge_vertices(s::HasDeltaSet1D, e...) = SVector(∂(1,0,s,e...), ∂(1,1,s,e...))

add_edge!(s::HasDeltaSet1D, src::Int, tgt::Int; kw...) =
  add_part!(s, :E; ∂v1=src, ∂v0=tgt, kw...)

function add_edges!(s::HasDeltaSet1D, srcs::AbstractVector{Int},
                    tgts::AbstractVector{Int}; kw...)
  @assert (n = length(srcs)) == length(tgts)
  add_parts!(s, :E, n; ∂v1=srcs, ∂v0=tgts, kw...)
end

""" Add edge to simplicial set, respecting the order of the vertex IDs.
"""
add_sorted_edge!(s::HasDeltaSet1D, v₀::Int, v₁::Int; kw...) =
  add_edge!(s, min(v₀, v₁), max(v₀, v₁); kw...)

""" Add edges to simplicial set, respecting the order of the vertex IDs.
"""
function add_sorted_edges!(s::HasDeltaSet1D, vs₀::AbstractVector{Int},
                           vs₁::AbstractVector{Int}; kw...)
  add_edges!(s, min.(vs₀, vs₁), max.(vs₀, vs₁); kw...)
end

# 1D oriented simplicial sets
#----------------------------

@present SchOrientedDeltaSet1D <: SchDeltaSet1D begin
  Orientation::AttrType
  edge_orientation::Attr(E,Orientation)
end

""" A one-dimensional oriented delta set.

Edges are oriented from source to target when `edge_orientation` is
true/positive and from target to source when it is false/negative.
"""
@acset_type OrientedDeltaSet1D(SchOrientedDeltaSet1D,
                               index=[:∂v0,:∂v1]) <: AbstractDeltaSet1D

orientation(::Type{Val{1}}, s::HasDeltaSet1D, args...) =
  s[args..., :edge_orientation]
set_orientation!(::Type{Val{1}}, s::HasDeltaSet1D, e, orientation) =
  (s[e, :edge_orientation] = orientation)

function ∂_nz(::Type{Val{1}}, s::HasDeltaSet1D, e::Int)
  (edge_vertices(s, e), sign(1,s,e) * @SVector([1,-1]))
end

function d_nz(::Type{Val{0}}, s::HasDeltaSet1D, v::Int)
  e₀, e₁ = coface(1,0,s,v), coface(1,1,s,v)
  (lazy(vcat, e₀, e₁), lazy(vcat, sign(1,s,e₀), -sign(1,s,e₁)))
end

# 1D embedded simplicial sets
#----------------------------

@present SchEmbeddedDeltaSet1D <: SchOrientedDeltaSet1D begin
  Point::AttrType
  point::Attr(V, Point)
end

""" A one-dimensional, embedded, oriented delta set.
"""
@acset_type EmbeddedDeltaSet1D(SchEmbeddedDeltaSet1D,
                               index=[:∂v0,:∂v1]) <: AbstractDeltaSet1D

""" Point associated with vertex of complex.
"""
point(s::HasDeltaSet, args...) = s[args..., :point]

struct CayleyMengerDet end

volume(::Type{Val{n}}, s::EmbeddedDeltaSet1D, x) where n =
  volume(Val{n}, s, x, CayleyMengerDet())
volume(::Type{Val{1}}, s::HasDeltaSet1D, e::Int, ::CayleyMengerDet) =
  volume(point(s, edge_vertices(s, e)))

# 2D simplicial sets
####################

@present SchDeltaSet2D <: SchDeltaSet1D begin
  Tri::Ob
  (∂e0, ∂e1, ∂e2)::Hom(Tri,E) # (∂₂(0), ∂₂(1), ∂₂(2))

  # Simplicial identities.
  ∂e1 ⋅ ∂v1 == ∂e2 ⋅ ∂v1 # ∂₂(1) ⋅ ∂₁(1) == ∂₂(2) ⋅ ∂₁(1) == v₀
  ∂e0 ⋅ ∂v1 == ∂e2 ⋅ ∂v0 # ∂₂(0) ⋅ ∂₁(1) == ∂₂(2) ⋅ ∂₁(0) == v₁
  ∂e0 ⋅ ∂v0 == ∂e1 ⋅ ∂v0 # ∂₂(0) ⋅ ∂₁(0) == ∂₂(1) ⋅ ∂₁(0) == v₂
end

""" Abstract type for C-sets containing a 2D delta set.
"""
@abstract_acset_type HasDeltaSet2D <: HasDeltaSet1D

""" Abstract type for 2D delta sets.
"""
@abstract_acset_type AbstractDeltaSet2D <: HasDeltaSet2D

""" A 2D delta set, aka semi-simplicial set.

The triangles in a semi-simpicial set can be interpreted in several ways.
Geometrically, they are triangles (2-simplices) whose three edges are directed
according to a specific pattern, determined by the ordering of the vertices or
equivalently by the simplicial identities. This geometric perspective is present
through the subpart names `∂e0`, `∂e1`, and `∂e2` and through the boundary map
[`∂`](@ref). Alternatively, the triangle can be interpreted as a
higher-dimensional link or morphism, going from two edges in sequence (which
might be called `src2_first` and `src2_last`) to a transitive edge (say `tgt2`).
This is the shape of the binary composition operation in a category.
"""
@acset_type DeltaSet2D(SchDeltaSet2D,
                       index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2]) <: AbstractDeltaSet2D

triangles(s::HasDeltaSet2D) = parts(s, :Tri)
function triangles(s::HasDeltaSet2D, v₀::Int, v₁::Int, v₂::Int)
  # Note: This could be written in a more efficient way by using ∂ interspersed
  # with the calls to coface, similar to edges().

  # Note: A faster method could be written if the mesh is guaranteed to be
  # manifold-like, and it is guaranteed that v₀ < v₁ < v₂.

  # Note: This is a more "Catlab" approach to this problem:
  #homs = homomorphisms(
  #  representable(EmbeddedDeltaSet3D{Bool,Point3D}, SchEmbeddedDeltaSet3D, :Tri),
  #  s;
  #  initial=(V=[v₀, v₁, v₂],))
  #map(x -> only(x[:Tri].func), homs)
  # This is more elegant, exploiting the explicit representation of our axioms
  # from the schema, instead of implicitly exploiting them like below.
  # This method requires us to provide the schema for s as well, so we may have
  # to refactor around multiple dispatch, or always data-migrate to
  # SchDeltaSet2D.

  e₀s = coface(1,0,s,v₂) ∩ coface(1,1,s,v₁)
  isempty(e₀s) && return Int[]
  e₁s = coface(1,0,s,v₂) ∩ coface(1,1,s,v₀)
  isempty(e₁s) && return Int[]
  e₂s = coface(1,0,s,v₁) ∩ coface(1,1,s,v₀)
  isempty(e₂s) && return Int[]
  coface(2,0,s,e₀s...) ∩ coface(2,1,s,e₁s...) ∩ coface(2,2,s,e₂s...)
end

ntriangles(s::HasDeltaSet2D) = nparts(s, :Tri)
nsimplices(::Type{Val{2}}, s::HasDeltaSet2D) = ntriangles(s)

face(::Type{Val{(2,0)}}, s::HasDeltaSet2D, args...) = subpart(s, args..., :∂e0)
face(::Type{Val{(2,1)}}, s::HasDeltaSet2D, args...) = subpart(s, args..., :∂e1)
face(::Type{Val{(2,2)}}, s::HasDeltaSet2D, args...) = subpart(s, args..., :∂e2)

coface(::Type{Val{(2,0)}}, s::HasDeltaSet2D, args...) = incident(s, args..., :∂e0)
coface(::Type{Val{(2,1)}}, s::HasDeltaSet2D, args...) = incident(s, args..., :∂e1)
coface(::Type{Val{(2,2)}}, s::HasDeltaSet2D, args...) = incident(s, args..., :∂e2)

""" Boundary edges of a triangle.
"""
function triangle_edges(s::HasDeltaSet2D, t...)
  SVector(∂(2,0,s,t...), ∂(2,1,s,t...), ∂(2,2,s,t...))
end

""" Boundary vertices of a triangle.

This accessor assumes that the simplicial identities hold.
"""
function triangle_vertices(s::HasDeltaSet2D, t...)
  SVector(s[s[t..., :∂e1], :∂v1],
          s[s[t..., :∂e2], :∂v0],
          s[s[t..., :∂e1], :∂v0])
end

""" Add a triangle (2-simplex) to a simplicial set, given its boundary edges.

In the arguments to this function, the boundary edges have the order ``0 → 1``,
``1 → 2``, ``0 → 2``. i.e. (∂e₂, ∂e₀, ∂e₁).

!!! warning

    This low-level function does not check the simplicial identities. It is your
    responsibility to ensure they are satisfied. By contrast, triangles added
    using the function [`glue_triangle!`](@ref) always satisfy the simplicial
    identities, by construction. Thus it is often easier to use this function.
"""
add_triangle!(s::HasDeltaSet2D, src2_first::Int, src2_last::Int, tgt2::Int; kw...) =
  add_part!(s, :Tri; ∂e0=src2_last, ∂e1=tgt2, ∂e2=src2_first, kw...)

""" Glue a triangle onto a simplicial set, given its boundary vertices.

If a needed edge between two vertices exists, it is reused (hence the "gluing");
otherwise, it is created.

Note this function does not check whether a triangle [v₀,v₁,v₂] already exists.

Note that this function does not rearrange v₀, v₁, v₂ in the way that minimizes
the number of edges added. For example, if s is the DeltaSet with a single
triangle [1,2,3] and edges [1,2], [2,3], [1,3], then gluing triangle [3,1,4]
will add edges [3,1], [1,4], [3,4] so as to respect the simplicial identities.
Note that the edges [1,3] and [3,1] are distinct!
However, if the DeltaSet that one is creating is meant to be manifold-like, then
adding triangles using only the command [`glue_sorted_triangle!`](@ref)
guarantees that the minimal number of new edges are created.
# TODO: Reference a proof of the above claim.
"""
function glue_triangle!(s::HasDeltaSet2D, v₀::Int, v₁::Int, v₂::Int; kw...)
  add_triangle!(s, get_edge!(s, v₀, v₁), get_edge!(s, v₁, v₂),
                get_edge!(s, v₀, v₂); kw...)
end

function get_edge!(s::HasDeltaSet1D, src::Int, tgt::Int)
  es = edges(s, src, tgt)
  isempty(es) ? add_edge!(s, src, tgt) : first(es)
end

""" Glue a triangle onto a simplicial set, respecting the order of the vertices.
"""
function glue_sorted_triangle!(s::HasDeltaSet2D, v₀::Int, v₁::Int, v₂::Int; kw...)
  v₀, v₁, v₂ = sort(SVector(v₀, v₁, v₂))
  glue_triangle!(s, v₀, v₁, v₂; kw...)
end

# 2D oriented simplicial sets
#----------------------------

@present SchOrientedDeltaSet2D <: SchDeltaSet2D begin
  Orientation::AttrType
  edge_orientation::Attr(E,Orientation)
  tri_orientation::Attr(Tri,Orientation)
end

""" A two-dimensional oriented delta set.

Triangles are ordered in the cyclic order ``(0,1,2)`` when `tri_orientation` is
true/positive and in the reverse order when it is false/negative.
"""
@acset_type OrientedDeltaSet2D(SchOrientedDeltaSet2D,
                               index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2]) <: AbstractDeltaSet2D

orientation(::Type{Val{2}}, s::HasDeltaSet2D, args...) =
  s[args..., :tri_orientation]
set_orientation!(::Type{Val{2}}, s::HasDeltaSet2D, t, orientation) =
  (s[t, :tri_orientation] = orientation)

function ∂_nz(::Type{Val{2}}, s::HasDeltaSet2D, t::Int)
  edges = triangle_edges(s,t)
  (edges, sign(2,s,t) * sign(1,s,edges) .* @SVector([1,-1,1]))
end

function d_nz(::Type{Val{1}}, s::HasDeltaSet2D, e::Int)
  sgn = sign(1, s, e)
  t₀, t₁, t₂ = coface(2,0,s,e), coface(2,1,s,e), coface(2,2,s,e)
  (lazy(vcat, t₀, t₁, t₂),
   lazy(vcat, sgn*sign(2,s,t₀), -sgn*sign(2,s,t₁), sgn*sign(2,s,t₂)))
end

# 2D embedded simplicial sets
#----------------------------

@present SchEmbeddedDeltaSet2D <: SchOrientedDeltaSet2D begin
  Point::AttrType
  point::Attr(V, Point)
end

""" A two-dimensional, embedded, oriented delta set.
"""
@acset_type EmbeddedDeltaSet2D(SchEmbeddedDeltaSet2D,
                               index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2]) <: AbstractDeltaSet2D

volume(::Type{Val{n}}, s::EmbeddedDeltaSet2D, x) where n =
  volume(Val{n}, s, x, CayleyMengerDet())
volume(::Type{Val{2}}, s::HasDeltaSet2D, t::Int, ::CayleyMengerDet) =
  volume(point(s, triangle_vertices(s,t)))

# 3D simplicial sets
####################

@present SchDeltaSet3D <: SchDeltaSet2D begin
  Tet::Ob
  (∂t0, ∂t1, ∂t2, ∂t3)::Hom(Tet,Tri) # (∂₃(0), ∂₃(1), ∂₃(2), ∂₃(3))

  # Simplicial identities.
  ∂t3 ⋅ ∂e2 == ∂t2 ⋅ ∂e2
  ∂t3 ⋅ ∂e1 == ∂t1 ⋅ ∂e2
  ∂t3 ⋅ ∂e0 == ∂t0 ⋅ ∂e2

  ∂t2 ⋅ ∂e1 == ∂t1 ⋅ ∂e1
  ∂t2 ⋅ ∂e0 == ∂t0 ⋅ ∂e1

  ∂t1 ⋅ ∂e0 == ∂t0 ⋅ ∂e0
end

""" Abstract type for C-sets containing a 3D delta set.
"""
@abstract_acset_type HasDeltaSet3D <: HasDeltaSet2D

""" Abstract type for 3D delta sets.
"""
@abstract_acset_type AbstractDeltaSet3D <: HasDeltaSet3D

""" A 3D delta set, aka semi-simplicial set.

"""
@acset_type DeltaSet3D(SchDeltaSet3D,
                       index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:∂t0,:∂t1,:∂t2,:∂t3]) <: AbstractDeltaSet3D

tetrahedra(s::HasDeltaSet3D) = parts(s, :Tet)
ntetrahedra(s::HasDeltaSet3D) = nparts(s, :Tet)
nsimplices(::Type{Val{3}}, s::HasDeltaSet3D) = ntetrahedra(s)

face(::Type{Val{(3,0)}}, s::HasDeltaSet3D, args...) = subpart(s, args..., :∂t0)
face(::Type{Val{(3,1)}}, s::HasDeltaSet3D, args...) = subpart(s, args..., :∂t1)
face(::Type{Val{(3,2)}}, s::HasDeltaSet3D, args...) = subpart(s, args..., :∂t2)
face(::Type{Val{(3,3)}}, s::HasDeltaSet3D, args...) = subpart(s, args..., :∂t3)

coface(::Type{Val{(3,0)}}, s::HasDeltaSet3D, args...) = incident(s, args..., :∂t0)
coface(::Type{Val{(3,1)}}, s::HasDeltaSet3D, args...) = incident(s, args..., :∂t1)
coface(::Type{Val{(3,2)}}, s::HasDeltaSet3D, args...) = incident(s, args..., :∂t2)
coface(::Type{Val{(3,3)}}, s::HasDeltaSet3D, args...) = incident(s, args..., :∂t3)

""" Boundary triangles of a tetrahedron.
"""
function tetrahedron_triangles(s::HasDeltaSet3D, t...)
  SVector(∂(3,0,s,t...), ∂(3,1,s,t...), ∂(3,2,s,t...), ∂(3,3,s,t...))
end

""" Boundary edges of a tetrahedron.

This accessor assumes that the simplicial identities hold.
"""
function tetrahedron_edges(s::HasDeltaSet3D, t...)
  SVector(s[s[t..., :∂t0], :∂e0], # e₀
          s[s[t..., :∂t0], :∂e1], # e₁
          s[s[t..., :∂t0], :∂e2], # e₂
          s[s[t..., :∂t1], :∂e1], # e₃
          s[s[t..., :∂t1], :∂e2], # e₄
          s[s[t..., :∂t2], :∂e2]) # e₅
end

""" Boundary vertices of a tetrahedron.

This accessor assumes that the simplicial identities hold.
"""
function tetrahedron_vertices(s::HasDeltaSet3D, t...)
  SVector(s[s[s[t..., :∂t2], :∂e2], :∂v1], # v₀
          s[s[s[t..., :∂t2], :∂e2], :∂v0], # v₁
          s[s[s[t..., :∂t0], :∂e0], :∂v1], # v₂
          s[s[s[t..., :∂t0], :∂e0], :∂v0]) # v₃
end

""" Add a tetrahedron (3-simplex) to a simplicial set, given its boundary triangles.

!!! warning

    This low-level function does not check the simplicial identities. It is your
    responsibility to ensure they are satisfied. By contrast, tetrahedra added
    using the function [`glue_tetrahedron!`](@ref) always satisfy the simplicial
    identities, by construction. Thus it is often easier to use this function.
"""
add_tetrahedron!(s::HasDeltaSet3D, tri0::Int, tri1::Int, tri2::Int, tri3::Int; kw...) =
  add_part!(s, :Tet; ∂t0=tri0, ∂t1=tri1, ∂t2=tri2, ∂t3=tri3, kw...)

""" Glue a tetrahedron onto a simplicial set, given its boundary vertices.

If a needed triangle between two vertices exists, it is reused (hence the "gluing");
otherwise, it is created. Necessary 1-simplices are likewise glued.
"""
function glue_tetrahedron!(s::HasDeltaSet3D, v₀::Int, v₁::Int, v₂::Int, v₃::Int; kw...)
  # Note: There is a redundancy here in that the e.g. the first get_triangle!
  # guarantees that certain edges are already added, so some later calls to
  # get_edge! inside the following calls to get_triangle! don't actually need to
  # search using the edges() function for whether they have been added.
  add_tetrahedron!(s,
    get_triangle!(s, v₁, v₂, v₃), # t₀
    get_triangle!(s, v₀, v₂, v₃), # t₁
    get_triangle!(s, v₀, v₁, v₃), # t₂
    get_triangle!(s, v₀, v₁, v₂); # t₃
    kw...)
end

function get_triangle!(s::HasDeltaSet2D, v₀::Int, v₁::Int, v₂::Int)
  ts = triangles(s, v₀, v₁, v₂)
  isempty(ts) ? glue_triangle!(s, v₀, v₁, v₂) : first(ts)
end

""" Glue a tetrahedron onto a simplicial set, respecting the order of the vertices.
"""
function glue_sorted_tetrahedron!(s::HasDeltaSet3D, v₀::Int, v₁::Int, v₂::Int, v₃::Int; kw...)
  v₀, v₁, v₂, v₃ = sort(SVector(v₀, v₁, v₂, v₃))
  glue_tetrahedron!(s, v₀, v₁, v₂, v₃; kw...)
end

""" Glue a tetrahedralized cube onto a simplicial set, respecting the order of the vertices.

After sorting, the faces of the cube are:
1 5-4 0,
1 2-6 5,
1 2-3 0,
7 4-0 3,
7 3-2 6,
7 6-5 4,
For each face, the diagonal edge is between those vertices connected by a dash.
The internal diagonal is between vertices 1 and 7.
"""
function glue_sorted_tet_cube!(s::HasDeltaSet3D, v₀::Int, v₁::Int, v₂::Int,
  v₃::Int, v₄::Int, v₅::Int, v₆::Int, v₇::Int; kw...)
  v₀, v₁, v₂, v₃, v₄, v₅, v₆, v₇ = sort(SVector(v₀, v₁, v₂, v₃, v₄, v₅, v₆, v₇))
  glue_tetrahedron!(s, v₀, v₁, v₃, v₇; kw...),
  glue_tetrahedron!(s, v₁, v₂, v₃, v₇; kw...),
  glue_tetrahedron!(s, v₀, v₁, v₄, v₇; kw...),
  glue_tetrahedron!(s, v₁, v₂, v₆, v₇; kw...),
  glue_tetrahedron!(s, v₁, v₄, v₅, v₇; kw...),
  glue_tetrahedron!(s, v₁, v₅, v₆, v₇; kw...)
end

# 3D oriented simplicial sets
#----------------------------

@present SchOrientedDeltaSet3D <: SchDeltaSet3D begin
  Orientation::AttrType
  edge_orientation::Attr(E,Orientation)
  tri_orientation::Attr(Tri,Orientation)
  tet_orientation::Attr(Tet,Orientation)
end

""" A three-dimensional oriented delta set.
"""
@acset_type OrientedDeltaSet3D(SchOrientedDeltaSet3D,
                               index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:∂t0,:∂t1,:∂t2,:∂t3]) <: AbstractDeltaSet3D

orientation(::Type{Val{3}}, s::HasDeltaSet3D, args...) =
  s[args..., :tet_orientation]
set_orientation!(::Type{Val{3}}, s::HasDeltaSet3D, t, orientation) =
  (s[t, :tet_orientation] = orientation)

function ∂_nz(::Type{Val{3}}, s::HasDeltaSet3D, tet::Int)
  tris = tetrahedron_triangles(s, tet)
  (tris, sign(3,s,tet) * sign(2,s,tris) .* @SVector([1,-1,1,-1]))
end

function d_nz(::Type{Val{2}}, s::HasDeltaSet3D, tri::Int)
  t₀, t₁, t₂, t₃ = map(x -> coface(3,x,s,tri), 0:3)
  sgn = sign(2, s, tri)
  (lazy(vcat, t₀, t₁, t₂, t₃),
   lazy(vcat,
     sgn*sign(3,s,t₀), -sgn*sign(3,s,t₁), sgn*sign(3,s,t₂), -sgn*sign(3,s,t₃)))
end

# 3D embedded simplicial sets
#----------------------------

@present SchEmbeddedDeltaSet3D <: SchOrientedDeltaSet3D begin
  Point::AttrType
  point::Attr(V, Point)
end

""" A three-dimensional, embedded, oriented delta set.
"""
@acset_type EmbeddedDeltaSet3D(SchEmbeddedDeltaSet3D,
                               index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:∂t0,:∂t1,:∂t2,:∂t3]) <: AbstractDeltaSet3D

volume(::Type{Val{n}}, s::EmbeddedDeltaSet3D, x) where n =
  volume(Val{n}, s, x, CayleyMengerDet())
volume(::Type{Val{3}}, s::HasDeltaSet3D, t::Int, ::CayleyMengerDet) =
  volume(point(s, tetrahedron_vertices(s,t)))

# General operators
###################

""" Wrapper for simplex or simplices of dimension `n`.

See also: [`V`](@ref), [`E`](@ref), [`Tri`](@ref).
"""
@parts_array_struct Simplex{n}

""" Vertex in simplicial set: alias for `Simplex{0}`.
"""
const V = Simplex{0}

""" Edge in simplicial set: alias for `Simplex{1}`.
"""
const E = Simplex{1}

""" Triangle in simplicial set: alias for `Simplex{2}`.
"""
const Tri = Simplex{2}

""" Tetrahedron in simplicial set: alias for `Simplex{3}`.
"""
const Tet = Simplex{3}

""" Wrapper for chain of oriented simplices of dimension `n`.
"""
@vector_struct SimplexChain{n}

const VChain = SimplexChain{0}
const EChain = SimplexChain{1}
const TriChain = SimplexChain{2}
const TetChain = SimplexChain{3}

""" Wrapper for discrete form, aka cochain, in simplicial set.
"""
@vector_struct SimplexForm{n}

const VForm = SimplexForm{0}
const EForm = SimplexForm{1}
const TriForm = SimplexForm{2}
const TetForm = SimplexForm{3}

""" Simplices of given dimension in a simplicial set.
"""
@inline simplices(n::Int, s::HasDeltaSet) = 1:nsimplices(Val{n}, s)

""" Number of simplices of given dimension in a simplicial set.
"""
@inline nsimplices(n::Int, s::HasDeltaSet) = nsimplices(Val{n}, s)

""" Face map and boundary operator on simplicial sets.

Given numbers `n` and `0 <= i <= n` and a simplicial set of dimension at least
`n`, the `i`th face map is implemented by the call

```julia
∂(n, i, s, ...)
```

The boundary operator on `n`-faces and `n`-chains is implemented by the call

```julia
∂(n, s, ...)
```

Note that the face map returns *simplices*, while the boundary operator returns
*chains* (vectors in the free vector space spanned by oriented simplices).
"""
@inline ∂(i::Int, s::HasDeltaSet, x::Simplex{n}) where n =
  Simplex{n-1}(face(Val{(n,i)}, s, x.data))
@inline ∂(n::Int, i::Int, s::HasDeltaSet, args...) =
  face(Val{(n,i)}, s, args...)

@inline coface(i::Int, s::HasDeltaSet, x::Simplex{n}) where n =
  Simplex{n+1}(coface(Val{(n+1,i)}, s, x.data))
@inline coface(n::Int, i::Int, s::HasDeltaSet, args...) =
  coface(Val{(n,i)}, s, args...)

∂(s::HasDeltaSet, x::SimplexChain{n}) where n =
  SimplexChain{n-1}(∂(Val{n}, s, x.data))
@inline ∂(n::Int, s::HasDeltaSet, args...) = ∂(Val{n}, s, args...)

function ∂(::Type{Val{n}}, s::HasDeltaSet, args...) where n
  operator_nz(Int, nsimplices(n-1,s), nsimplices(n,s), args...) do x
    ∂_nz(Val{n}, s, x)
  end
end

""" Alias for the face map and boundary operator [`∂`](@ref).
"""
const boundary = ∂

""" The discrete exterior derivative, aka the coboundary operator.
"""
d(s::HasDeltaSet, x::SimplexForm{n}) where n =
  SimplexForm{n+1}(d(Val{n}, s, x.data))
@inline d(n::Int, s::HasDeltaSet, args...) = d(Val{n}, s, args...)

function d(::Type{Val{n}}, s::HasDeltaSet, args...) where n
  operator_nz(Int, nsimplices(n+1,s), nsimplices(n,s), args...) do x
    d_nz(Val{n}, s, x)
  end
end

""" Alias for the coboundary operator [`d`](@ref).
"""
const coboundary = d

""" Alias for the discrete exterior derivative [`d`](@ref).
"""
const exterior_derivative = d

""" Orientation of simplex.
"""
orientation(s::HasDeltaSet, x::Simplex{n}) where n =
  orientation(Val{n}, s, x.data)
@inline orientation(n::Int, s::HasDeltaSet, args...) =
  orientation(Val{n}, s, args...)

@inline Base.sign(n::Int, s::HasDeltaSet, args...) = sign(Val{n}, s, args...)
Base.sign(::Type{Val{n}}, s::HasDeltaSet, args...) where n =
  numeric_sign.(orientation(Val{n}, s, args...))

numeric_sign(x) = sign(x)
numeric_sign(x::Bool) = x ? +1 : -1

""" Set orientation of simplex.
"""
@inline set_orientation!(n::Int, s::HasDeltaSet, args...) =
  set_orientation!(Val{n}, s, args...)

""" ``n``-dimensional volume of ``n``-simplex in an embedded simplicial set.
"""
volume(s::HasDeltaSet, x::Simplex{n}, args...) where n =
  volume(Val{n}, s, x.data, args...)
@inline volume(n::Int, s::HasDeltaSet, args...) = volume(Val{n}, s, args...)

""" Convenience function for linear operator based on structural nonzero values.
"""
operator_nz(f, ::Type{T}, m::Int, n::Int,
            x::Int, Vec::Type=SparseVector{T}) where T = fromnz(Vec, f(x)..., m)
operator_nz(f, ::Type{T}, m::Int, n::Int,
            vec::AbstractVector) where T = applynz(f, vec, m, n)
operator_nz(f, ::Type{T}, m::Int, n::Int,
            Mat::Type=SparseMatrixCSC{T}) where T = fromnz(f, Mat, m, n)

# Consistent orientation
########################

""" Consistently orient simplices in a simplicial set, if possible.

Two simplices with a common face are *consistently oriented* if they induce
opposite orientations on the shared face. This function attempts to consistently
orient all simplices of a given dimension and returns whether this has been
achieved. Each connected component is oriently independently using the helper
function [`orient_component!`](@ref).
"""
orient!(s::AbstractDeltaSet1D) = orient!(s, E)
orient!(s::AbstractDeltaSet2D) = orient!(s, Tri)
orient!(s::AbstractDeltaSet3D) = orient!(s, Tet)

function orient!(s::HasDeltaSet, ::Type{Simplex{n}}) where n
  # Compute connected components as coequalizer of face maps.
  ndom, ncodom = nsimplices(n, s), nsimplices(n-1, s)
  face_maps = SVector{n+1}([ FinFunction(x -> ∂(n,i,s,x), ndom, ncodom)
                             for i in 0:n ])
  π = only(coequalizer(face_maps))

  # Choose an arbitrary representative of each component.
  reps = zeros(Int, length(codom(π)))
  for x in reverse(simplices(n, s))
    reps[π(∂(n,0,s,x))] = x
  end

  # Orient each component, starting at the chosen representative.
  init_orientation = one(attrtype_type(s, :Orientation))
  for x in reps
    orient_component!(s, Simplex{n}(x), init_orientation) || return false
  end
  true
end

""" Consistently orient simplices in the same connected component, if possible.

Given an ``n``-simplex and a choice of orientation for it, this function
attempts to consistently orient all ``n``-simplices that may be reached from it
by traversing ``(n-1)``-faces. The traversal is depth-first. If a consistent
orientation is possible, the function returns `true` and the orientations are
assigned; otherwise, it returns `false` and no orientations are changed.

If the simplicial set is not connected, the function [`orient!`](@ref) may be
more convenient.
"""
orient_component!(s::AbstractDeltaSet1D, e::Int, args...) =
  orient_component!(s, E(e), args...)
orient_component!(s::AbstractDeltaSet2D, t::Int, args...) =
  orient_component!(s, Tri(t), args...)
orient_component!(s::AbstractDeltaSet3D, t::Int, args...) =
  orient_component!(s, Tet(t), args...)

function orient_component!(s::HasDeltaSet, x::Simplex{n},
                           x_orientation::Orientation) where {n, Orientation}
  orientations = repeat(Union{Orientation,Nothing}[nothing], nsimplices(n, s))

  orient_stack = Vector{Pair{Int64, Orientation}}()

  push!(orient_stack, x[] => x_orientation)
  is_orientable = true
  while !isempty(orient_stack)
    x, target = pop!(orient_stack)
    current = orientations[x]
    if isnothing(current)
      # If not visited, set the orientation and add neighbors to stack.
      orientations[x] = target
      for i in 0:n, j in 0:n
        next = iseven(i+j) ? negate(target) : target
        for y in coface(n, j, s, ∂(n, i, s, x))
          y == x || push!(orient_stack, y=>next)
        end
      end
    elseif current != target
      is_orientable = false
      break
    end
  end

  if is_orientable
    component = findall(!isnothing, orientations)
    set_orientation!(n, s, component, orientations[component])
  end
  is_orientable
end

negate(x) = -x
negate(x::Bool) = !x

# Euclidean geometry
####################

""" ``n``-dimensional volume of ``n``-simplex spanned by given ``n+1`` points.
"""
function volume(points)
  CM = cayley_menger(points...)
  n = length(points) - 1
  sqrt(abs(det(CM)) / 2^n) / factorial(n)
end

""" Construct Cayley-Menger matrix for simplex spanned by given points.

For an ``n`-simplex, this is the ``(n+2)×(n+2)`` matrix that appears in the
[Cayley-Menger
determinant](https://en.wikipedia.org/wiki/Cayley-Menger_determinant).
"""
function cayley_menger(p0::V, p1::V) where V <: AbstractVector
  d01 = sqdistance(p0, p1)
  SMatrix{3,3}(0,  1,   1,
               1,  0,   d01,
               1,  d01, 0)
end
function cayley_menger(p0::V, p1::V, p2::V) where V <: AbstractVector
  d01, d12, d02 = sqdistance(p0, p1), sqdistance(p1, p2), sqdistance(p0, p2)
  SMatrix{4,4}(0,  1,   1,   1,
               1,  0,   d01, d02,
               1,  d01, 0,   d12,
               1,  d02, d12, 0)
end
function cayley_menger(p0::V, p1::V, p2::V, p3::V) where V <: AbstractVector
  d01, d12, d02 = sqdistance(p0, p1), sqdistance(p1, p2), sqdistance(p0, p2)
  d03, d13, d23 = sqdistance(p0, p3), sqdistance(p1, p3), sqdistance(p2, p3)
  SMatrix{5,5}(0,  1,   1,   1,   1,
               1,  0,   d01, d02, d03,
               1,  d01, 0,   d12, d13,
               1,  d02, d12, 0,   d23,
               1,  d03, d13, d23, 0)
end

""" Squared Euclidean distance between two points.
"""
sqdistance(x, y) = sum((x-y).^2)

# Manifold-like
###############

""" Test whether a given simplicial complex is manifold-like.

According to Hirani, "all simplices of dimension ``k`` with ``0 ≤ k ≤ n - 1`` 
must be the face of some simplex of dimension ``n`` in the complex." This 
function does not test that simplices do not overlap. Nor does it test that e.g.
two triangles that share 2 vertices share an edge. Nor does it test that e.g.
there is at most one triangle that connects 3 vertices. Nor does it test that
the delta set consists of a single component.
"""
is_manifold_like(s::AbstractDeltaSet1D) = is_manifold_like(s, E)
is_manifold_like(s::AbstractDeltaSet2D) = is_manifold_like(s, Tri)
is_manifold_like(s::AbstractDeltaSet3D) = is_manifold_like(s, Tet)

function is_manifold_like(s::HasDeltaSet, ::Type{Simplex{n}}) where n
  # The yth k-simplex c is not a face of an (k+1)-simplex if the yth column of
  # the exterior derivative matrix is all zeros.
  foreach(0:n-1) do k
    any(iszero, eachcol(d(k,s))) && return false
  end
  true
end

""" Find the simplices which are not a face of another.

For an n-D oriented delta set, return a vector of 0 through n-1 chains
consisting of the simplices that are not the face of another. Note that since
n-simplices in an n-D oriented delta set are never the face of an (n+1)-simplex,
these are excluded.

We choose the term "nonboundaries" so as not to be confused with the term
"nonface", defined as those faces that are not in a simplical complex, whose
corresponding monomials are the basis of the Stanley-Reisner ideal.
"""
nonboundaries(s::AbstractDeltaSet1D) = nonboundaries(s, E)
nonboundaries(s::AbstractDeltaSet2D) = nonboundaries(s, Tri)
nonboundaries(s::AbstractDeltaSet3D) = nonboundaries(s, Tet)

function nonboundaries(s::HasDeltaSet, ::Type{Simplex{n}}) where n
  # The yth k-simplex c is not a face of an (k+1)-simplex if the yth column of
  # the exterior derivative matrix is all zeros.
  map(0:n-1) do k
    SimplexChain{k}(findall(iszero, eachcol(d(k,s))))
  end
end

end
