""" Simplicial sets in one, two, and three dimensions.

For the time being, this module provides data structures only for [delta
sets](https://en.wikipedia.org/wiki/Delta_set), also known as [semi-simplicial
sets](https://ncatlab.org/nlab/show/semi-simplicial+set). These include the face
maps but not the degeneracy maps of a simplicial set. In the future we may add
support for simplicial sets. The analogy to keep in mind is that graphs are to
semi-simpicial sets as reflexive graphs are to simplicial sets.

Simplicial sets are inherently ordered structures. The "unordered" analogue of
simplicial sets are symmetric simplicial sets, sometimes called just "symmetric
sets." In one dimension, symmetric semi-simplicial sets are symmetric graphs.

This module does not implement symmetric simplicial sets as such. However,
symmetric sets can be simulated with simplicial sets by enforcing that the
ordering of the vertices of each face matches the ordering of the integer vertex
IDs. The simplicial set then "presents" a symmetric set in a canonical way. The
functions [`add_sorted_edge!`](@ref) and [`glue_sorted_triangle!`](@ref)
automatically sort their inputs to ensure that the ordering condition is
satisfied.
"""
module SimplicialSets
export ∂, boundary, d, coboundary, exterior_derivative,
  AbstractDeltaSet1D, DeltaSet1D, OrientedDeltaSet1D,
  ∂₁, src, tgt, edge_sign, nv, ne, vertices, edges, has_vertex, has_edge,
  add_vertex!, add_vertices!, add_edge!, add_edges!,
  add_sorted_edge!, add_sorted_edges!,
  AbstractDeltaSet2D, DeltaSet2D, OrientedDeltaSet2D,
  ∂₂, triangle_vertex, triangle_sign, ntriangles, triangles,
  add_triangle!, glue_triangle!, glue_sorted_triangle!

using SparseArrays
using StaticArrays: @SVector, SVector

using Catlab, Catlab.CategoricalAlgebra.CSets, Catlab.Graphs
using Catlab.Graphs.BasicGraphs: TheoryGraph, TheoryReflexiveGraph
using ..ArrayUtils

# 1D simplicial sets
####################

const DeltaCategory1D = TheoryGraph
const SimplexCategory1D = TheoryReflexiveGraph

""" Abstract type for 1D delta sets.
"""
const AbstractDeltaSet1D = AbstractGraph

""" A one-dimensional delta set, aka semi-simplicial set.

Delta sets in 1D are the same as graphs, and this type is just an alias for
`Graph`. The boundary operator [`∂₁`](@ref) translates the graph-theoretic
terminology into simplicial terminology.
"""
const DeltaSet1D = Graph

""" Face map on edges and boundary operator on 1-chains in simplicial set.
"""
@inline ∂₁(i::Int, s::AbstractACSet, args...) =
  ∂₁(Val{i}, s::AbstractACSet, args...)

∂₁(::Type{Val{0}}, s::AbstractACSet, args...) = subpart(s, args..., :tgt)
∂₁(::Type{Val{1}}, s::AbstractACSet, args...) = subpart(s, args..., :src)

@inline ∂₁_inv(i::Int, s::AbstractACSet, args...) =
  ∂₁_inv(Val{i}, s::AbstractACSet, args...)

∂₁_inv(::Type{Val{0}}, s::AbstractACSet, args...) = incident(s, args..., :tgt)
∂₁_inv(::Type{Val{1}}, s::AbstractACSet, args...) = incident(s, args..., :src)

""" Add edge to simplicial set, respecting the order of the vertex IDs.
"""
add_sorted_edge!(s::AbstractACSet, v₀::Int, v₁::Int; kw...) =
  add_edge!(s, min(v₀, v₁), max(v₀, v₁); kw...)

""" Add edges to simplicial set, respecting the order of the vertex IDs.
"""
function add_sorted_edges!(s::AbstractACSet, vs₀::AbstractVector{Int},
                           vs₁::AbstractVector{Int}; kw...)
  add_edges!(s, min.(vs₀, vs₁), max.(vs₀, vs₁); kw...)
end

# 1D oriented simplicial sets
#----------------------------

@present OrientedDeltaSchema1D <: DeltaCategory1D begin
  Orientation::Data
  edge_orientation::Attr(E,Orientation)
end

""" A one-dimensional oriented delta set.

Edges are oriented from source to target when `edge_orientation` is
true/positive and from target to source when it is false/negative.
"""
const OrientedDeltaSet1D = ACSetType(OrientedDeltaSchema1D, index=[:src,:tgt])

""" Sign (±1) associated with edge orientation.
"""
edge_sign(s::AbstractACSet, args...) =
  numeric_sign.(s[args..., :edge_orientation])

numeric_sign(x) = sign(x)
numeric_sign(x::Bool) = x ? +1 : -1

∂₁(s::AbstractACSet, e::Int, Vec::Type=SparseVector{Int}) =
  fromnz(Vec, ∂₁nz(s,e)..., nv(s))
∂₁(s::AbstractACSet, echain::AbstractVector) =
  applynz(echain, nv(s), ne(s)) do e; ∂₁nz(s,e) end
∂₁(s::AbstractACSet, Mat::Type=SparseMatrixCSC{Int}) =
  fromnz(Mat, nv(s), ne(s)) do e; ∂₁nz(s,e) end

function ∂₁nz(s::AbstractACSet, e::Int)
  (SVector(∂₁(0,s,e), ∂₁(1,s,e)), edge_sign(s,e) * @SVector([1,-1]))
end

""" Coboundary operator on 0-forms.
"""
δ₀(s::AbstractACSet, vform::AbstractVector) =
  applynz(vform, ne(s), nv(s)) do v; δ₀nz(s,v) end
δ₀(s::AbstractACSet, Mat::Type=SparseMatrixCSC{Int}) =
  fromnz(Mat, ne(s), nv(s)) do v; δ₀nz(s,v) end

function δ₀nz(s::AbstractACSet, v::Int)
  e₀, e₁ = ∂₁_inv(0,s,v), ∂₁_inv(1,s,v)
  (lazy(vcat, e₀, e₁), lazy(vcat, edge_sign(s,e₀), -edge_sign(s,e₁)))
end

# 2D simplicial sets
####################

@present DeltaCategory2D <: DeltaCategory1D begin
  Tri::Ob
  (∂e0, ∂e1, ∂e2)::Hom(Tri,E) # (∂₂(0), ∂₂(1), ∂₂(2))

  # Simplicial identities.
  ∂e1 ⋅ src == ∂e2 ⋅ src # ∂₂(1) ⋅ ∂₁(1) == ∂₂(2) ⋅ ∂₁(1) == v₀
  ∂e0 ⋅ src == ∂e2 ⋅ tgt # ∂₂(0) ⋅ ∂₁(1) == ∂₂(2) ⋅ ∂₁(0) == v₁
  ∂e0 ⋅ tgt == ∂e1 ⋅ tgt # ∂₂(0) ⋅ ∂₁(0) == ∂₂(1) ⋅ ∂₁(0) == v₂
end

""" Abstract type for 2D delta sets.
"""
const AbstractDeltaSet2D = AbstractACSetType(DeltaCategory2D)

""" A 2D delta set, aka semi-simplicial set.

The triangles in a semi-simpicial set can be interpreted in several ways.
Geometrically, they are triangles (2-simplices) whose three edges are directed
according to a specific pattern, determined by the ordering of the vertices or
equivalently by the simplicial identities. This geometric perspective is present
through the subpart names `∂e0`, `∂e1`, and `∂e2` and through the boundary map
`[∂₂](@ref)`. Alternatively, the triangle can be interpreted as a
higher-dimensional link or morphism, going from two edges in sequence (which
might be called `src2_first` and `src2_last`) to a transitive edge (say `tgt2`).
This is the shape of the binary composition operation in a category.
"""
const DeltaSet2D = CSetType(DeltaCategory2D,
                            index=[:src, :tgt, :∂e0, :∂e1, :∂e2])

""" Face map on triangles and boundary operator on 2-chains in simplicial set.
"""
@inline ∂₂(i::Int, s::AbstractACSet, args...) =
  ∂₂(Val{i}, s::AbstractACSet, args...)

∂₂(::Type{Val{0}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e0)
∂₂(::Type{Val{1}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e1)
∂₂(::Type{Val{2}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e2)

∂₂_inv(i::Int, s::AbstractACSet, args...) =
  ∂₂_inv(Val{i}, s::AbstractACSet, args...)

∂₂_inv(::Type{Val{0}}, s::AbstractACSet, args...) = incident(s, args..., :∂e0)
∂₂_inv(::Type{Val{1}}, s::AbstractACSet, args...) = incident(s, args..., :∂e1)
∂₂_inv(::Type{Val{2}}, s::AbstractACSet, args...) = incident(s, args..., :∂e2)

triangles(s::AbstractACSet) = parts(s, :Tri)
ntriangles(s::AbstractACSet) = nparts(s, :Tri)

""" Boundary vertex of a triangle.

This accessor assumes that the simplicial identities hold.
"""
@inline triangle_vertex(i::Int, s::AbstractACSet, args...) =
  triangle_vertex(Val{i}, s, args...)

triangle_vertex(::Type{Val{0}}, s::AbstractACSet, args...) =
  s[s[args..., :∂e1], :src]
triangle_vertex(::Type{Val{1}}, s::AbstractACSet, args...) =
  s[s[args..., :∂e2], :tgt]
triangle_vertex(::Type{Val{2}}, s::AbstractACSet, args...) =
  s[s[args..., :∂e1], :tgt]

""" Add a triangle (2-simplex) to a simplicial set, given its boundary edges.

In the arguments to this function, the boundary edges have the order ``0 → 1``,
``1 → 2``, ``0 -> 2``.

!!! warning

    This low-level function does not check the simplicial identities. It is your
    responsibility to ensure they are satisfied. By contrast, triangles added
    using the function [`glue_triangle!`](@ref) always satisfy the simplicial
    identities, by construction. Thus it is often easier to use this function.
"""
add_triangle!(s::AbstractACSet, src2_first::Int, src2_last::Int, tgt2::Int; kw...) =
  add_part!(s, :Tri; ∂e0=src2_last, ∂e1=tgt2, ∂e2=src2_first, kw...)

""" Glue a triangle onto a simplicial set, given its boundary vertices.

If a needed edge between two vertices exists, it is reused (hence the "gluing");
otherwise, it is created.
"""
function glue_triangle!(s::AbstractACSet, v₀::Int, v₁::Int, v₂::Int; kw...)
  add_triangle!(s, get_edge!(s, v₀, v₁), get_edge!(s, v₁, v₂),
                get_edge!(s, v₀, v₂); kw...)
end

function get_edge!(s::AbstractACSet, src::Int, tgt::Int)
  es = edges(s, src, tgt)
  isempty(es) ? add_edge!(s, src, tgt) : first(es)
end

""" Glue a triangle onto a simplicial set, respecting the order of the vertices.
"""
function glue_sorted_triangle!(s::AbstractACSet, v₀::Int, v₁::Int, v₂::Int; kw...)
  v₀, v₁, v₂ = sort(SVector(v₀, v₁, v₂))
  glue_triangle!(s, v₀, v₁, v₂; kw...)
end

# 2D oriented simplicial sets
#----------------------------

@present OrientedDeltaSchema2D <: DeltaCategory2D begin
  Orientation::Data
  edge_orientation::Attr(E,Orientation)
  tri_orientation::Attr(Tri,Orientation)
end

""" A two-dimensional oriented delta set.

Triangles are ordered in the cyclic order ``(0,1,2)`` (with numbers defined by
[`triangle_vertex`](@ref)) when `tri_orientation` is true/positive and in the
reverse order when it is false/negative.
"""
const OrientedDeltaSet2D = ACSetType(OrientedDeltaSchema2D,
                                     index=[:src, :tgt, :∂e0, :∂e1, :∂e2])

""" Sign (±1) associated with triangle orientation.
"""
triangle_sign(s::AbstractACSet, args...) =
  numeric_sign.(s[args..., :tri_orientation])

∂₂(s::AbstractACSet, t::Int, Vec::Type=SparseVector{Int}) =
  fromnz(Vec, ∂₂nz(s,t)..., ne(s))
∂₂(s::AbstractACSet, tchain::AbstractVector) =
  applynz(tchain, ne(s), ntriangles(s)) do t; ∂₂nz(s,t) end
∂₂(s::AbstractACSet, Mat::Type=SparseMatrixCSC{Int}) =
  fromnz(Mat, ne(s), ntriangles(s)) do t; ∂₂nz(s,t) end

function ∂₂nz(s::AbstractACSet, t::Int)
  edges = SVector(∂₂(0,s,t), ∂₂(1,s,t), ∂₂(2,s,t))
  (edges, triangle_sign(s,t) * edge_sign(s,edges) .* @SVector([1,-1,1]))
end

""" Coboundary operator on 1-forms.
"""
δ₁(s::AbstractACSet, eform::AbstractVector) =
  applynz(eform, ntriangles(s), ne(s)) do e; δ₁nz(s,e) end
δ₁(s::AbstractACSet, Mat::Type=SparseMatrixCSC{Int}) =
  fromnz(Mat, ntriangles(s), ne(s)) do e; δ₁nz(s,e) end

function δ₁nz(s::AbstractACSet, e::Int)
  sgn = edge_sign(s, e)
  t₀, t₁, t₂ = ∂₂_inv(0,s,e), ∂₂_inv(1,s,e), ∂₂_inv(2,s,e)
  (lazy(vcat, t₀, t₁, t₂),
   lazy(vcat, sgn*triangle_sign(s,t₀),
        -sgn*triangle_sign(s,t₁), sgn*triangle_sign(s,t₂)))
end

# General operators
###################

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
@inline ∂(n::Int, i::Int, s::AbstractACSet, args...) =
  ∂(Val{n}, Val{i}, s::AbstractACSet, args...)

∂(::Type{Val{1}}, i::Type, s::AbstractACSet, args...) = ∂₁(i, s, args...)
∂(::Type{Val{2}}, i::Type, s::AbstractACSet, args...) = ∂₂(i, s, args...)

@inline ∂(n::Int, s::AbstractACSet, args...) =
  ∂(Val{n}, s::AbstractACSet, args...)

∂(::Type{Val{1}}, s::AbstractACSet, args...) = ∂₁(s, args...)
∂(::Type{Val{2}}, s::AbstractACSet, args...) = ∂₂(s, args...)

""" Alias for the face map and boundary operator [`∂`](@ref).
"""
const boundary = ∂

""" The discrete exterior derivative, aka the coboundary operator.
"""
@inline d(n::Int, s::AbstractACSet, args...) =
  d(Val{n}, s::AbstractACSet, args...)

d(::Type{Val{0}}, s::AbstractACSet, args...) = δ₀(s, args...)
d(::Type{Val{1}}, s::AbstractACSet, args...) = δ₁(s, args...)

""" Alias for the coboundary operator [`d`](@ref).
"""
const coboundary = d

""" Alias for the discrete exterior derivative [`d`](@ref).
"""
const exterior_derivative = d

end
