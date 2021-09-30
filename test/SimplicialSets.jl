module TestSimplicialSets
using Test

using SparseArrays, StaticArrays

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces.SimplicialSets

""" Check that the semi-simplicial identities hold in dimension `n`.
"""
function is_semi_simplicial(s::ACSet, n::Int)
  all(∂(n-1, i, s, ∂(n, j, s)) == ∂(n-1, j-1, s, ∂(n, i, s))
      for i in 0:n for j in (i+1):n)
end

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

# 1D simplicial sets
####################

s = DeltaSet1D()
add_vertices!(s, 4)
add_sorted_edge!(s, 2, 1)
add_sorted_edges!(s, [2,4], [3,3])
@test src(s) == [1,2,3]
@test tgt(s) == [2,3,4]
@test ∂(1, 0, s) == [2,3,4]
@test ∂(1, 1, s) == [1,2,3]
@test ∂(0, s, E(1))::V == V(2)
@test ∂(1, s, E([1,3]))::V == V([1,3])
@test coface(0, s, V(4))::E == E([3])

# 1D oriented simplicial sets
#----------------------------

s = OrientedDeltaSet1D{Bool}()
add_vertices!(s, 4)
add_edges!(s, [1,2,3], [2,3,4], edge_orientation=[true,false,true])
@test ∂(1, s, 1) == [-1,1,0,0]
@test ∂(1, s, 2) == [0,1,-1,0]

# Boundary operator, dense vectors.
vvec = ∂(1, s, 1, Vector{Int})
@test !issparse(vvec)
@test vvec == [-1,1,0,0]
vvec = ∂(1, s, [1,-1,1])
@test !issparse(vvec)
@test vvec == [-1,0,0,1]
@test ∂(s, EChain([1,-1,1]))::VChain == VChain(vvec)

# Boundary operator, sparse vectors.
vvec = ∂(1, s, 1, SparseVector{Int})
@test issparse(vvec)
@test vvec == [-1,1,0,0]
vvec = ∂(1, s, sparsevec([1,-1,1]))
@test issparse(vvec)
@test vvec == [-1,0,0,1]
@test ∂(s, EChain(sparsevec([1,-1,1]))) == VChain(vvec)
B = ∂(1, s)
@test issparse(B)
@test B*[1,-1,1] == [-1,0,0,1]

# Exterior derivative.
@test d(s, VForm([1,1,1,4]))::EForm == EForm([0,0,3])
@test d(s, VForm([4,1,0,0])) == EForm([-3,1,0])
@test d(0, s) == B'

# Consistent orientation.
s = OrientedDeltaSet1D{Bool}()
add_vertices!(s, 10)
add_edges!(s, 1:4, 2:5)
add_edges!(s, 6:9, 7:10)
@test orient_component!(s, 1, true)
@test orientation(s, E(1:4)) == trues(4)
@test orient_component!(s, 8, true)
@test orientation(s, E(1:8)) == trues(8)

s[:edge_orientation] = rand(Bool, 8)
@test orient!(s)
@test orientation(s, E(1:8)) == trues(8)

# 1D embedded simplicial sets
#----------------------------

s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(s, 2, point=[Point2D(-1, 0), Point2D(+1, 0)])
add_edge!(s, 1, 2, edge_orientation=true)
@test volume(1, s, 1) ≈ 2
@test volume(s, E(1)) ≈ 2

# 2D simplicial sets
####################

s = DeltaSet2D()
add_vertices!(s, 3)
glue_triangle!(s, 1, 2, 3)
@test is_semi_simplicial(s, 2)
@test ntriangles(s) == 1
@test map(i -> ∂(2, i, s, 1), (0,1,2)) == (2,3,1)
@test triangle_vertices(s, 1) == [1,2,3]

s′ = DeltaSet2D()
add_vertices!(s′, 3)
glue_sorted_triangle!(s′, 2, 3, 1)
@test s′ == s

# Triangulated commutative square.
s = DeltaSet2D()
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
s = OrientedDeltaSet2D{Bool}()
add_vertices!(s, 3)
add_sorted_edges!(s, [1,2,3], [2,3,1], edge_orientation=[true,true,false])
glue_triangle!(s, 1, 2, 3)
@test orient_component!(s, 1, true)
@test orientation(s, Tri(1)) == true
@test ∂(2, s, 1) == [1,1,1]
@test d(1, s, [45,3,34]) == [82]

# Triangulated square with consistent orientation.
s = OrientedDeltaSet2D{Bool}()
add_vertices!(s, 4)
glue_triangle!(s, 1, 2, 3)
glue_triangle!(s, 1, 3, 4)
s[:edge_orientation] = true
@test orient!(s)
@test orientation(s, Tri(1:2)) == trues(2)
@test ∂(2, s, 1) == [1,1,-1,0,0]
@test ∂(s, TriChain([1,1]))::EChain == EChain([1,1,0,1,-1])
@test d(s, EForm([45,3,34,0,0]))::TriForm == TriForm([14, 34]) # == [45+3-34, 34]
@test d(s, EForm([45,3,34,17,5])) == TriForm([14, 46]) # == [45+3-34, 34+17-5]
@test d(1, s) == ∂(2, s)'

# 2D embedded simplicial sets
#----------------------------

# Standard 2-simplex in ℝ³.
s = EmbeddedDeltaSet2D{Bool,Point3D}()
add_vertices!(s, 3, point=[Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)])
glue_triangle!(s, 1, 2, 3, tri_orientation=true)
@test volume(s, Tri(1)) ≈ sqrt(3)/2

# Euclidean geometry
####################

std_simplex_volume(n::Int) = sqrt(n+1) / factorial(n)

p1, p2 = Point2D(1,0), Point2D(0,1)
@test volume([p1, p2]) ≈ std_simplex_volume(1)

p1, p2, p3 = Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)
@test volume([p1, p2, p3]) ≈ std_simplex_volume(2)

p1, p2, p3, p4 = SVector(1,0,0,0), SVector(0,1,0,0), SVector(0,0,1,0), SVector(0,0,0,1)
@test volume([p1, p2, p3, p4]) ≈ std_simplex_volume(3)

end
