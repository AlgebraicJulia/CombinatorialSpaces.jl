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
export Simplex, V, E, Tri, SimplexChain, VChain, EChain, TriChain,
  SimplexForm, VForm, EForm, TriForm,
  AbstractDeltaSet1D, DeltaSet1D, OrientedDeltaSet1D, EmbeddedDeltaSet1D,
  AbstractDeltaSet2D, DeltaSet2D, OrientedDeltaSet2D, EmbeddedDeltaSet2D,
  ∂, boundary, coface, d, coboundary, exterior_derivative,
  simplices, nsimplices, point, volume, orient!, orient_component!,
  src, tgt, nv, ne, vertices, edges, has_vertex, has_edge,
  edge_vertices, edge_sign, add_vertex!, add_vertices!, add_edge!, add_edges!,
  add_sorted_edge!, add_sorted_edges!,
  triangle_vertices, triangle_sign, ntriangles, triangles,
  add_triangle!, glue_triangle!, glue_sorted_triangle!

using LinearAlgebra: det
using SparseArrays
using StaticArrays: @SVector, SVector, SMatrix

using Catlab, Catlab.CategoricalAlgebra,
  Catlab.CategoricalAlgebra.FinSets, Catlab.Graphs
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

nsimplices(::Type{Val{0}}, s) = nv(s)
nsimplices(::Type{Val{1}}, s) = ne(s)

face(::Type{Val{(1,0)}}, s::AbstractACSet, args...) = subpart(s, args..., :tgt)
face(::Type{Val{(1,1)}}, s::AbstractACSet, args...) = subpart(s, args..., :src)

coface(::Type{Val{(1,0)}}, s::AbstractACSet, args...) = incident(s, args..., :tgt)
coface(::Type{Val{(1,1)}}, s::AbstractACSet, args...) = incident(s, args..., :src)

""" Boundary vertices of an edge.
"""
edge_vertices(s::AbstractACSet, e...) = SVector(∂(1,0,s,e...), ∂(1,1,s,e...))

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

orientation(::Type{Val{1}}, s::AbstractACSet, args...) =
  s[args..., :edge_orientation]
set_orientation!(::Type{Val{1}}, s::AbstractACSet, e, orientation) =
  (s[e, :edge_orientation] = orientation)

""" Sign (±1) associated with edge orientation.
"""
edge_sign(s::AbstractACSet, args...) =
  numeric_sign.(s[args..., :edge_orientation])

numeric_sign(x) = sign(x)
numeric_sign(x::Bool) = x ? +1 : -1

function ∂_nz(::Type{Val{1}}, s::AbstractACSet, e::Int)
  (edge_vertices(s, e), edge_sign(s,e) * @SVector([1,-1]))
end

function d_nz(::Type{Val{0}}, s::AbstractACSet, v::Int)
  e₀, e₁ = coface(1,0,s,v), coface(1,1,s,v)
  (lazy(vcat, e₀, e₁), lazy(vcat, edge_sign(s,e₀), -edge_sign(s,e₁)))
end

# 1D embedded simplicial sets
#----------------------------

@present EmbeddedDeltaSchema1D <: OrientedDeltaSchema1D begin
  Point::Data
  point::Attr(V, Point)
end

""" A one-dimensional, embedded, oriented delta set.
"""
const EmbeddedDeltaSet1D = ACSetType(EmbeddedDeltaSchema1D, index=[:src,:tgt])

""" Point associated with vertex of complex.
"""
point(s::AbstractACSet, args...) = s[args..., :point]

struct CayleyMengerDet end

volume(::Type{Val{n}}, s::EmbeddedDeltaSet1D, x) where n =
  volume(Val{n}, s, x, CayleyMengerDet())
volume(::Type{Val{1}}, s::AbstractACSet, e::Int, ::CayleyMengerDet) =
  volume(point(s, edge_vertices(s, e)))

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
[`∂`](@ref). Alternatively, the triangle can be interpreted as a
higher-dimensional link or morphism, going from two edges in sequence (which
might be called `src2_first` and `src2_last`) to a transitive edge (say `tgt2`).
This is the shape of the binary composition operation in a category.
"""
const DeltaSet2D = CSetType(DeltaCategory2D,
                            index=[:src, :tgt, :∂e0, :∂e1, :∂e2])

triangles(s::AbstractACSet) = parts(s, :Tri)
ntriangles(s::AbstractACSet) = nparts(s, :Tri)
nsimplices(::Type{Val{2}}, s) = ntriangles(s)

face(::Type{Val{(2,0)}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e0)
face(::Type{Val{(2,1)}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e1)
face(::Type{Val{(2,2)}}, s::AbstractACSet, args...) = subpart(s, args..., :∂e2)

coface(::Type{Val{(2,0)}}, s::AbstractACSet, args...) = incident(s, args..., :∂e0)
coface(::Type{Val{(2,1)}}, s::AbstractACSet, args...) = incident(s, args..., :∂e1)
coface(::Type{Val{(2,2)}}, s::AbstractACSet, args...) = incident(s, args..., :∂e2)

""" Boundary vertices of a triangle.

This accessor assumes that the simplicial identities hold.
"""
function triangle_vertices(s::AbstractACSet, t...)
  SVector(s[s[t..., :∂e1], :src], s[s[t..., :∂e2], :tgt], s[s[t..., :∂e1], :tgt])
end

""" Add a triangle (2-simplex) to a simplicial set, given its boundary edges.

In the arguments to this function, the boundary edges have the order ``0 → 1``,
``1 → 2``, ``0 → 2``.

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

Triangles are ordered in the cyclic order ``(0,1,2)`` when `tri_orientation` is
true/positive and in the reverse order when it is false/negative.
"""
const OrientedDeltaSet2D = ACSetType(OrientedDeltaSchema2D,
                                     index=[:src, :tgt, :∂e0, :∂e1, :∂e2])

orientation(::Type{Val{2}}, s::AbstractACSet, args...) =
  s[args..., :tri_orientation]
set_orientation!(::Type{Val{2}}, s::AbstractACSet, t, orientation) =
  (s[t, :tri_orientation] = orientation)

""" Sign (±1) associated with triangle orientation.
"""
triangle_sign(s::AbstractACSet, args...) =
  numeric_sign.(s[args..., :tri_orientation])

function ∂_nz(::Type{Val{2}}, s::AbstractACSet, t::Int)
  edges = SVector(∂(2,0,s,t), ∂(2,1,s,t), ∂(2,2,s,t))
  (edges, triangle_sign(s,t) * edge_sign(s,edges) .* @SVector([1,-1,1]))
end

function d_nz(::Type{Val{1}}, s::AbstractACSet, e::Int)
  sgn = edge_sign(s, e)
  t₀, t₁, t₂ = coface(2,0,s,e), coface(2,1,s,e), coface(2,2,s,e)
  (lazy(vcat, t₀, t₁, t₂),
   lazy(vcat, sgn*triangle_sign(s,t₀),
        -sgn*triangle_sign(s,t₁), sgn*triangle_sign(s,t₂)))
end

# 2D embedded simplicial sets
#----------------------------

@present EmbeddedDeltaSchema2D <: OrientedDeltaSchema2D begin
  Point::Data
  point::Attr(V, Point)
end

""" A two-dimensional, embedded, oriented delta set.
"""
const EmbeddedDeltaSet2D = ACSetType(EmbeddedDeltaSchema2D,
                                     index=[:src, :tgt, :∂e0, :∂e1, :∂e2])

volume(::Type{Val{n}}, s::EmbeddedDeltaSet2D, x) where n =
  volume(Val{n}, s, x, CayleyMengerDet())
volume(::Type{Val{2}}, s::AbstractACSet, t::Int, ::CayleyMengerDet) =
  volume(point(s, triangle_vertices(s,t)))

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

""" Wrapper for chain of oriented simplices of dimension `n`.
"""
@vector_struct SimplexChain{n}

const VChain = SimplexChain{0}
const EChain = SimplexChain{1}
const TriChain = SimplexChain{2}

""" Wrapper for discrete form, aka cochain, in simplicial set.
"""
@vector_struct SimplexForm{n}

const VForm = SimplexForm{0}
const EForm = SimplexForm{1}
const TriForm = SimplexForm{2}

""" Simplices of given dimension in a simplicial set.
"""
@inline simplices(n::Int, s::AbstractACSet) = 1:nsimplices(Val{n}, s)

""" Number of simplices of given dimension in a simplicial set.
"""
@inline nsimplices(n::Int, s::AbstractACSet) = nsimplices(Val{n}, s)

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
@inline ∂(i::Int, s::AbstractACSet, x::Simplex{n}) where n =
  Simplex{n-1}(face(Val{(n,i)}, s, x.data))
@inline ∂(n::Int, i::Int, s::AbstractACSet, args...) =
  face(Val{(n,i)}, s, args...)

@inline coface(i::Int, s::AbstractACSet, x::Simplex{n}) where n =
  Simplex{n+1}(coface(Val{(n+1,i)}, s, x.data))
@inline coface(n::Int, i::Int, s::AbstractACSet, args...) =
  coface(Val{(n,i)}, s, args...)

∂(s::AbstractACSet, x::SimplexChain{n}) where n =
  SimplexChain{n-1}(∂(Val{n}, s, x.data))
@inline ∂(n::Int, s::AbstractACSet, args...) = ∂(Val{n}, s, args...)

function ∂(::Type{Val{n}}, s::AbstractACSet, args...) where n
  operator_nz(Int, nsimplices(n-1,s), nsimplices(n,s), args...) do x
    ∂_nz(Val{n}, s, x)
  end
end

""" Alias for the face map and boundary operator [`∂`](@ref).
"""
const boundary = ∂

""" The discrete exterior derivative, aka the coboundary operator.
"""
d(s::AbstractACSet, x::SimplexForm{n}) where n =
  SimplexForm{n+1}(d(Val{n}, s, x.data))
@inline d(n::Int, s::AbstractACSet, args...) = d(Val{n}, s, args...)

function d(::Type{Val{n}}, s::AbstractACSet, args...) where n
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

""" ``n``-dimensional volume of ``n``-simplex in an embedded simplicial set.
"""
volume(s::AbstractACSet, x::Simplex{n}, args...) where n =
  volume(Val{n}, s, x.data, args...)
@inline volume(n::Int, s::AbstractACSet, args...) = volume(Val{n}, s, args...)

""" Convenience function for linear operator based on structural nonzero values.
"""
operator_nz(f, ::Type{T}, m::Int, n::Int,
            x::Int, Vec::Type=SparseVector{T}) where T = fromnz(Vec, f(x)..., m)
operator_nz(f, ::Type{T}, m::Int, n::Int,
            vec::AbstractVector) where T = applynz(f, vec, m, n)
operator_nz(f, ::Type{T}, m::Int, n::Int,
            Mat::Type=SparseMatrixCSC{T}) where T = fromnz(f, Mat, m, n)

# Orientation
#############

""" Consistently orient simplices in a simplicial set, if possible.

Two simplices with a common face are *consistently oriented* if they induce
opposite orientations on the shared face. This function attempts to consistently
orient all simplices of a given dimension and returns whether this has been
achieved. Each connected component is oriently independently using the helper
function [`orient_component!`](@ref).
"""
orient!(s::AbstractDeltaSet1D) = orient!(s, E)
orient!(s::AbstractDeltaSet2D) = orient!(s, Tri)

function orient!(s::AbstractACSet, ::Type{Simplex{n}}) where n
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
  init_orientation = one(eltype(orientation(n, s)))
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

function orient_component!(s::AbstractACSet, x::Simplex{n},
                           x_orientation::Orientation) where {n, Orientation}
  orientations = repeat(Union{Orientation,Nothing}[nothing], nsimplices(n, s))

  function orient_neighbors!(x, target)
    current = orientations[x]
    if isnothing(current)
      # If not visited, set the orientation and recursively visit neighbors.
      orientations[x] = target
      for i in 0:n, j in 0:n
        next = iseven(i+j) ? negate(target) : target
        for y in coface(n, j, s, ∂(n, i, s, x))
          y == x || orient_neighbors!(y, next) || return false
        end
      end
      true
    else
      # If already visited, check that current and target orientations agree.
      current == target
    end
  end

  is_orientable = orient_neighbors!(x[], x_orientation)
  if is_orientable
    component = findall(!isnothing, orientations)
    set_orientation!(n, s, component, orientations[component])
  end
  is_orientable
end

@inline orientation(n::Int, s::AbstractACSet, args...) =
  orientation(Val{n}, s, args...)
@inline set_orientation!(n::Int, s::AbstractACSet, args...) =
  set_orientation!(Val{n}, s, args...)

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

""" Squared Euclidean distance between two points.
"""
sqdistance(x, y) = sum((x-y).^2)

end
