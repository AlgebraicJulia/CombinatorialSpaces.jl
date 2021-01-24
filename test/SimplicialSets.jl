module TestSimplicialSets
using Test

using SparseArrays

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces.SimplicialSets

""" Check that the semi-simplicial identities hold in dimension `n`.
"""
function is_semi_simplicial(s::AbstractACSet, n::Int)
  all(∂(n-1, i, s, ∂(n, j, s)) == ∂(n-1, j-1, s, ∂(n, i, s))
      for i in 1:n for j in (i+1):n)
end

# 1D simplicial sets
####################

s = SemiSimplicialSet1D()
add_vertices!(s, 4)
add_sorted_edge!(s, 2, 1)
add_sorted_edges!(s, [2,4], [3,3])
@test src(s) == [1,2,3]
@test tgt(s) == [2,3,4]
@test ∂₁(0, s) == [2,3,4]
@test ∂₁(1, s) == [1,2,3]

# 1D oriented simplicial sets
#----------------------------

s = OrientedSimplicialSet1D{Bool}()
add_vertices!(s, 4)
add_edges!(s, [1,2,3], [2,3,4], edge_orientation=[true,false,true])
@test ∂₁(s, 1) == [-1,1,0,0]
@test ∂₁(s, 2) == [0,1,-1,0]

# Boundary operator, dense vectors.
vvec = ∂₁(s, 1, Vector{Int})
@test !issparse(vvec)
@test vvec == [-1,1,0,0]
vvec = ∂₁(s, [1,-1,1])
@test !issparse(vvec)
@test vvec == [-1,0,0,1]

# Boundary operator, sparse vectors.
vvec = ∂₁(s, 1, SparseVector{Int})
@test issparse(vvec)
@test vvec == [-1,1,0,0]
vvec = ∂₁(s, sparsevec([1,-1,1]))
@test issparse(vvec)
@test vvec == [-1,0,0,1]

# 2D simplicial sets
####################

s = SemiSimplicialSet2D()
add_vertices!(s, 3)
glue_triangle!(s, 1, 2, 3)
@test is_semi_simplicial(s, 2)
@test ntriangles(s) == 1
@test map(i -> ∂₂(i, s, 1), (0,1,2)) == (2,3,1)
@test map(i -> triangle_vertex(s, i, 1), (0,1,2)) == (1,2,3)

s′ = SemiSimplicialSet2D()
add_vertices!(s′, 3)
glue_sorted_triangle!(s′, 2, 3, 1)
@test s′ == s

# Triangulated commutative square.
s = SemiSimplicialSet2D()
add_vertices!(s, 4)
glue_triangle!(s, 1, 2, 3)
glue_triangle!(s, 1, 4, 3)
@test ntriangles(s) == 2
@test triangles(s) == 1:2
@test ne(s) == 5
@test sort(map(Pair, src(s), tgt(s))) == [1=>2, 1=>3, 1=>4, 2=>3, 4=>3]

# 2D oriented simplicial sets
#----------------------------

# Triangle with matching edge orientations.
s = OrientedSimplicialSet2D{Bool}()
add_vertices!(s, 3)
add_sorted_edges!(s, [1,2,3], [2,3,1], edge_orientation=[true,true,false])
glue_triangle!(s, 1, 2, 3, tri_orientation=true)
@test ∂₂(s, 1) == [1,1,1]

# Triangulated square with consistent orientation.
s = OrientedSimplicialSet2D{Bool}()
add_vertices!(s, 4)
glue_triangle!(s, 1, 2, 3)
glue_triangle!(s, 1, 3, 4)
s[:edge_orientation] = true
s[:tri_orientation] = true
@test ∂₂(s, 1) == [1,1,-1,0,0]
@test ∂₂(s, [1,1]) == [1,1,0,1,-1] # 2-chain around perimeter.

end
