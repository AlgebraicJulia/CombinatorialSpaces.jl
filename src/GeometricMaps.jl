"""
A particularly simple definition for a geometric map from an n-truncated
simplicial set S to an n-truncated simplicial set T is just a function

S_0 -> T_n x Δ^n

This sends each vertex in S to a pair of an n-simplex in T and a point inside
the generic geometric n-simplex.

However, this naive definition can run into problems. Specifically, given a
k-simplex in S, its vertices must all be sent into the same n-simplex. But
this would imply that connected components in S must all be sent into the
same n-simplex. In order to fix this, we should change the definition to

S_0 -> \sum_{k=0}^n T_k x int(\Delta^k)

Then the condition for a k-simplex in S
"""

struct Point
  simplex::Int
  coordinates::Vector{Float64}
end

function vertices(d::DeltaSet, dim::Int, i::Int)
  if dim == 0
    [i]
  elseif dim == 1
    [src(d, i), tgt(d, i)]
  elseif dim == 2
    triangle_vertices(d, i)
  else
    error("we can't do above 2-dimensions")
  end
end

function vertices(d::DeltaSet, p::Point)
  vertices(d, length(p.coordinates) - 1, p.simplex)
end

function check_simplex_exists(
  codom::SimplicialComplex{D′}
  points::Vector{Point}
) where {D′}
  verts = vcat([vertices(codom.delta_set, p) for p in points])
  sort!(verts)
  unique!(verts)
  haskey(codom.complexes, verts) ||
    error("edge $e cannot map into a valid complex")
end

struct GeometricMap{D, D'}
  dom::SimplicialComplex{D}
  codom::SimplicialComplex{D'}
  values::Vector{Point}
  function GeometricMap(
    dom::SimplicialComplex{D},
    codom::SimplicialComplex{D'},
    values::Vector{Point}
  ) where {D <: AbstractDeltaSet1D, D'}

  end
end
