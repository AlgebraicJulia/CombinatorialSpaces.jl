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

struct GeometricMap
  values::Vector{Point}
end
