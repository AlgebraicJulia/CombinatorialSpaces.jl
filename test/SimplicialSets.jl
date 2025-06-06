module TestSimplicialSets
using Test

using SparseArrays, StaticArrays

using Catlab
using CombinatorialSpaces

""" Check that the semi-simplicial identities hold in dimension `n`.
"""
function is_semi_simplicial(s::HasDeltaSet, n::Int)
  all(∂(n-1, i, s, ∂(n, j, s)) == ∂(n-1, j-1, s, ∂(n, i, s))
      for i in 0:n for j in (i+1):n)
end

# 1D simplicial sets
####################

s = DeltaSet1D()
add_vertex!(s)
add_vertices!(s, 3)
add_sorted_edge!(s, 2, 1)
add_sorted_edges!(s, [2,4], [3,3])
@test (nv(s), ne(s)) == (4, 3)
@test (vertices(s), edges(s)) == (1:4, 1:3)
@test src(s) == [1,2,3]
@test tgt(s) == [2,3,4]
@test has_edge(s, 1, 2)
@test !has_edge(s, 2, 1)
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
@test is_manifold_like(s)
@test isempty(only(nonboundaries(s)))

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

s = EmbeddedDeltaSet1D{Bool,Point2d}()
add_vertices!(s, 2, point=[Point2d(-1, 0), Point2d(+1, 0)])
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
@test is_manifold_like(s)
@test isempty(nonboundaries(s)[1])
@test isempty(nonboundaries(s)[2])

# 2D embedded simplicial sets
#----------------------------

# Standard 2-simplex in ℝ³.
s = EmbeddedDeltaSet2D{Bool,Point3d}()
add_vertices!(s, 3, point=[Point3d(1,0,0), Point3d(0,1,0), Point3d(0,0,1)])
glue_triangle!(s, 1, 2, 3, tri_orientation=true)
@test volume(s, Tri(1)) ≈ sqrt(3)/2

# 3D simplicial sets
####################

s = DeltaSet3D()
add_vertices!(s, 4)
glue_tetrahedron!(s, 1, 2, 3, 4)
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
@test ntetrahedra(s) == 1
@test map(i -> ∂(2, i, s, 1), (0,1,2)) == (2,3,1)
@test map(i -> ∂(3, i, s, 1), (0,1,2,3)) == (1,2,3,4)
@test tetrahedron_vertices(s, 1) == [1,2,3,4]
@test tetrahedron_edges(s, 1) == [2,3,1, 5,4, 6]

s′ = DeltaSet3D()
add_vertices!(s′, 4)
glue_sorted_tetrahedron!(s′, 2, 4, 3, 1)
@test s′ == s
@test tetrahedron_vertices(s, 1) == tetrahedron_vertices(s′, 1)

# Two tetrahedra forming a square pyramid.
# The shared (internal) triangle is (2,3,5).
s = DeltaSet3D()
add_vertices!(s, 5)
glue_tetrahedron!(s, 1, 2, 3, 5)
glue_tetrahedron!(s, 2, 3, 4, 5)
@test ntetrahedra(s) == 2
@test tetrahedra(s) == 1:2
@test ntriangles(s) == 7
@test triangles(s) == 1:7
@test ne(s) == 9
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
@test tetrahedron_vertices(s, 1) == [1,2,3,5]
@test tetrahedron_vertices(s, 2) == [2,3,4,5]
tetrahedron_to_edges_to_vertices(s,t) = sort(unique(reduce(vcat, edge_vertices(s, tetrahedron_edges(s,t)))))
for t in tetrahedra(s)
  @test tetrahedron_to_edges_to_vertices(s,t) == tetrahedron_vertices(s,t)
end

# Six tetrahedra forming a cube.
s = DeltaSet3D()
add_vertices!(s, 8)
glue_tetrahedron!(s, 1, 2, 4, 8)
glue_tetrahedron!(s, 2, 3, 4, 8)
glue_tetrahedron!(s, 1, 2, 5, 8)
glue_tetrahedron!(s, 2, 3, 7, 8)
glue_tetrahedron!(s, 2, 5, 6, 8)
glue_tetrahedron!(s, 2, 6, 7, 8)
@test ntetrahedra(s) == 6
@test tetrahedra(s) == 1:6
@test ntriangles(s) == 18
@test triangles(s) == 1:18 # (2 * num cube faces) + (2 * num "cuts" into cube)
@test ne(s) == 19 # (num cube edges) + (num cube faces) + (internal diagonal)
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
for t in tetrahedra(s)
  @test tetrahedron_to_edges_to_vertices(s,t) == tetrahedron_vertices(s,t)
end

# Tetrahedralized cube via helper function.
s′ = DeltaSet3D()
add_vertices!(s′, 8)
glue_sorted_tet_cube!(s′, 1:8...)
@test s == s′

add_vertices!(s′, 4)
# Glue along the face of the cube with vertices 5,6,7,8.
glue_sorted_tet_cube!(s′, 5:12...)
@test is_semi_simplicial(s′, 3)
@test ntetrahedra(s′) == 2*ntetrahedra(s)
# There are 2 triangles, 5 edges, and 4 vertices on the shared cube face.
@test ntriangles(s′) == 2*ntriangles(s) - 2
@test ne(s′) == 2*ne(s) - 5
@test nv(s′) == 2*nv(s) - 4
for t in tetrahedra(s′)
  @test tetrahedron_to_edges_to_vertices(s′,t) == tetrahedron_vertices(s′,t)
end

# 3D oriented simplicial sets
#----------------------------

# Tetrahedron with orientation.
s = OrientedDeltaSet3D{Bool}()
add_vertices!(s, 4)
glue_tetrahedron!(s, 1, 2, 3, 4)
s[:edge_orientation] = [true, false, true, false, true, false]
s[:tri_orientation] = [true, false, true, false]
@test orient_component!(s, 1, true)
@test orientation(s, Tet(1)) == true
@test ∂(3, s, 1) == [1,1,1,1]
@test ∂(2, s, 1) == [1,-1,-1,0,0,0]
@test d(1, s, [17,19,23,29,31,37]) == [-25,79,-45,-9]
@test d(2, s, [3,5,17,257]) == [282] # == [3+5+17+257]

# Two tetrahedra forming a square pyramid with orientation.
# The shared (internal) triangle is (2,3,5).
s = OrientedDeltaSet3D{Bool}()
add_vertices!(s, 5)
glue_tetrahedron!(s, 1, 2, 3, 5)
glue_tetrahedron!(s, 2, 3, 4, 5)
s[:edge_orientation] = true
s[:tri_orientation] = true
@test orient!(s)
@test orientation(s, Tet(1:2)) == [true, false]
@test ∂(2, s, 1) == sparsevec([1,2,3], [1,1,-1], 9)
@test ∂(3, s, 1) == sparsevec([1,2,3,4], [1,-1,1,-1], 7)
@test ∂(s, TetChain([1,1]))::TriChain == TriChain([0,-1,1,-1,-1,1,1])
@test d(s, TriForm([1,10,100,1000,10000,100000,1000000]))::TetForm == TetForm([-909,1089999])
@test d(0, s) == ∂(1, s)'
@test d(1, s) == ∂(2, s)'
@test d(2, s) == ∂(3, s)'
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))

# Tetrahedralized cube with orientation.
s = OrientedDeltaSet3D{Bool}()
add_vertices!(s, 8)
glue_sorted_tet_cube!(s, 1:8..., tet_orientation=true)
add_vertices!(s, 4)
glue_sorted_tet_cube!(s, 5:12..., tet_orientation=false)
# Test that the kw args passed correctly:
@test s[:tet_orientation] == [fill(true,6)..., fill(false,6)...]
s[:edge_orientation] = false
s[:tri_orientation] = false
@test orient!(s)
@test is_manifold_like(s)

# 3D embedded simplicial sets
#----------------------------

# Regular tetrahedron with edge length 2√2 in ℝ³.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 4, point=[Point3d(1,1,1), Point3d(1,-1,-1),
  Point3d(-1,1,-1), Point3d(-1,-1,1)])
glue_tetrahedron!(s, 1, 2, 3, 4)
orient!(s)
equilateral_triangle_area(len) = √3/4*len^2
@test all(volume(s, Tri(1)) .≈ equilateral_triangle_area(2√2))
regular_tetrahedron_volume(len) = len^3/(6√2)
@test volume(s, Tet(1)) ≈ regular_tetrahedron_volume(2√2)

# Six tetrahedra of equal volume forming a cube with edge length 2.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 8, point=[
  Point3d(-1,1,1), Point3d(1,1,1), Point3d(1,-1,1), Point3d(-1,-1,1),
  Point3d(-1,1,-1), Point3d(1,1,-1), Point3d(1,-1,-1), Point3d(-1,-1,-1)])
glue_sorted_tetrahedron!(s, 1, 2, 4, 8)
glue_sorted_tetrahedron!(s, 2, 3, 4, 8)
glue_sorted_tetrahedron!(s, 1, 2, 5, 8)
glue_sorted_tetrahedron!(s, 2, 3, 7, 8)
glue_sorted_tetrahedron!(s, 2, 5, 6, 8)
glue_sorted_tetrahedron!(s, 2, 6, 7, 8)
@test ntetrahedra(s) == 6
@test tetrahedra(s) == 1:6
@test ntriangles(s) == 18
@test triangles(s) == 1:18 # (2 * num cube faces) + (2 * num "cuts" into cube)
@test ne(s) == 19 # (num cube edges) + (num cube faces) + (internal diagonal)
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)
for t in tetrahedra(s)
  @test volume(s, Tet(t)) ≈ 8/6
end
@test is_manifold_like(s)
for i in 1:3
  @test isempty(nonboundaries(s)[i])
end
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))
added_edge = add_edge!(s, 1,2, edge_orientation=true)
@test nonboundaries(s)[2] == EChain([added_edge])
added_triangle = add_triangle!(s, 1,2,3, tri_orientation=true)
@test nonboundaries(s)[3] == TriChain([added_triangle])

# Six tetrahedra of equal volume forming a cube with edge length 1.
# The connectivity is that of Blessent's 2009 thesis "Integration of 3D
# Geologicial and Numerical Models Based on Tetrahedral Meshes...", Figure 3.2.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 8, point=[
  Point3d(0,1,0), Point3d(0,0,0), Point3d(1,1,0), Point3d(1,0,0),
  Point3d(0,1,1), Point3d(0,0,1), Point3d(1,1,1), Point3d(1,0,1)])
# See Table 3.1 "Mesh connectivity".
glue_sorted_tetrahedron!(s, 3, 5, 4, 2)
glue_sorted_tetrahedron!(s, 7, 6, 8, 4)
glue_sorted_tetrahedron!(s, 5, 6, 7, 4)
glue_sorted_tetrahedron!(s, 3, 5, 7, 4)
glue_sorted_tetrahedron!(s, 5, 6, 4, 2)
glue_sorted_tetrahedron!(s, 1, 5, 3, 2)
@test ntetrahedra(s) == 6
@test tetrahedra(s) == 1:6
@test ntriangles(s) == 18
@test triangles(s) == 1:18 # (2 * num cube faces) + (2 * num "cuts" into cube)
@test ne(s) == 19 # (num cube edges) + (num cube faces) + (internal diagonal)
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)
for t in tetrahedra(s)
  @test volume(s, Tet(t)) ≈ 1/6
end
@test is_manifold_like(s)
for i in 1:3
  @test isempty(nonboundaries(s)[i])
end
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))

# Five tetrahedra example from Letniowski 1992, as given by Blessent Table 3.3b.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 6, point=[
  # See Table 3.3a "Nodal coordinates"
  Point3d(-2, -2,   0.5),
  Point3d( 0, -2,   0.1),
  Point3d(-2,  0,   0.1),
  Point3d( 0,  0.1, 0),
  Point3d(-2, -2,  -0.25),
  Point3d(-2, -2,   1.5)])
# See Table 3.3b "Connectivity list for Letniowski's example"
glue_sorted_tetrahedron!(s, 1, 2, 4, 6)
glue_sorted_tetrahedron!(s, 1, 3, 4, 6)
glue_sorted_tetrahedron!(s, 1, 2, 3, 5)
glue_sorted_tetrahedron!(s, 2, 3, 4, 5)
glue_sorted_tetrahedron!(s, 1, 2, 3, 4)
@test ntetrahedra(s) == 5
@test tetrahedra(s) == 1:5
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)
@test is_manifold_like(s)
for i in 1:3
  @test isempty(nonboundaries(s)[i])
end
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))

# Stacked tetrahedralized cubes each of volume 8.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 8, point=[
  Point3d(-1,1,1), Point3d(1,1,1), Point3d(1,-1,1), Point3d(-1,-1,1),
  Point3d(-1,1,-1), Point3d(1,1,-1), Point3d(1,-1,-1), Point3d(-1,-1,-1)])
glue_sorted_tet_cube!(s, 1:8...)
add_vertices!(s, 4, point=[
  Point3d(-1,1,-3), Point3d(1,1,-3), Point3d(1,-1,-3), Point3d(-1,-1,-3)])
glue_sorted_tet_cube!(s, 5:12...)
for t in tetrahedra(s)
  @test volume(s, Tet(t)) ≈ 8/6
end
@test sum([volume(s, Tet(t)) for t in tetrahedra(s)]) == 8*2

# Six tetrahedra of equal volume forming a cube with edge length 1.
# The connectivity is that of Blessent's 2009 thesis, Figure 3.2.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 8, point=[
  Point3d(0,1,0), Point3d(0,0,0), Point3d(1,1,0), Point3d(1,0,0),
  Point3d(0,1,1), Point3d(0,0,1), Point3d(1,1,1), Point3d(1,0,1)])
# See Table 3.1 "Mesh connectivity".
glue_sorted_tetrahedron!(s, 3, 5, 4, 2)
glue_sorted_tetrahedron!(s, 7, 6, 8, 4)
glue_sorted_tetrahedron!(s, 5, 6, 7, 4)
glue_sorted_tetrahedron!(s, 3, 5, 7, 4)
glue_sorted_tetrahedron!(s, 5, 6, 4, 2)
glue_sorted_tetrahedron!(s, 1, 5, 3, 2)
@test ntetrahedra(s) == 6
@test tetrahedra(s) == 1:6
@test ntriangles(s) == 18
@test triangles(s) == 1:18 # (2 * num cube faces) + (2 * num "cuts" into cube)
@test ne(s) == 19 # (num cube edges) + (num cube faces) + (internal diagonal)
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)
for t in tetrahedra(s)
  @test volume(s, Tet(t)) ≈ 1/6
end
@test is_manifold_like(s)
for i in 1:3
  @test isempty(nonboundaries(s)[i])
end
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))

# Five tetrahedra example from Letniowski 1992, as given by Blessent Table 3.3b.
s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 6, point=[
  # See Table 3.3a "Nodal coordinates"
  Point3d(-2, -2,   0.5),
  Point3d( 0, -2,   0.1),
  Point3d(-2,  0,   0.1),
  Point3d( 0,  0.1, 0),
  Point3d(-2, -2,  -0.25),
  Point3d(-2, -2,   1.5)])
# See Table 3.3b "Connectivity list for Letniowski's example"
glue_sorted_tetrahedron!(s, 1, 2, 4, 6)
glue_sorted_tetrahedron!(s, 1, 3, 4, 6)
glue_sorted_tetrahedron!(s, 1, 2, 3, 5)
glue_sorted_tetrahedron!(s, 2, 3, 4, 5)
glue_sorted_tetrahedron!(s, 1, 2, 3, 4)
@test ntetrahedra(s) == 5
@test tetrahedra(s) == 1:5
@test is_semi_simplicial(s, 2)
@test is_semi_simplicial(s, 3)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)
@test is_manifold_like(s)
for i in 1:3
  @test isempty(nonboundaries(s)[i])
end
@test d(1, s) * d(0, s) * vertices(s) == zeros(ntriangles(s))
@test d(2, s) * d(1, s) * edges(s) == zeros(ntetrahedra(s))

# Euclidean geometry
####################

std_simplex_volume(n::Int) = sqrt(n+1) / factorial(n)

p1, p2 = Point2d(1,0), Point2d(0,1)
@test volume([p1, p2]) ≈ std_simplex_volume(1)

p1, p2, p3 = Point3d(1,0,0), Point3d(0,1,0), Point3d(0,0,1)
@test volume([p1, p2, p3]) ≈ std_simplex_volume(2)

p1, p2, p3, p4 = SVector(1,0,0,0), SVector(0,1,0,0), SVector(0,0,1,0), SVector(0,0,0,1)
@test volume([p1, p2, p3, p4]) ≈ std_simplex_volume(3)

# Topological helper functions
##############################

# §62 Example 1 Figure 62.1 from Munkres 1984:
s = DeltaSet2D()
# 6 adjacent triangles forming a hexagon, with a stand-alone edge at the center.
g, h = 7, 8
add_vertices!(s, 8)
foreach(1:5, 2:6) do x,y
  glue_sorted_triangle!(s, g,x,y)
end
glue_sorted_triangle!(s, g,6,1)
add_sorted_edge!(s, g,h)

Stg = star(s, g)
@test issetequal(Stg[1], [g])
@test issetequal(Stg[2], union(coface(1,0,s,g), union(coface(1,1,s,g))))
@test issetequal(Stg[3], triangles(s))

St̄g = closed_star(s, g)
@test issetequal(St̄g[1], vertices(s))
@test issetequal(St̄g[2], edges(s))
@test issetequal(St̄g[3], triangles(s))

# "The link of the vertex g consists of the hexagon ... and the vertex h."
Lkg = link(s, g)
@test issetequal(Lkg[1], [1,2,3,4,5,6, 8])
@test issetequal(Lkg[2], setdiff(St̄g[2], Stg[2]))
@test isempty(Lkg[3])

# "The link of the vertex h is the vertex g."
Lkh = link(s, h)
@test issetequal(Lkh[1], [g])
@test isempty(Lkh[2])
@test isempty(Lkh[3])

# §62 Example 1 Figure 62.2 from Munkres 1984:
s = DeltaSet3D()
# 5 adjacent tetrahedra forming a pentagonal bipyramid.
add_vertices!(s, 7)
foreach(1:4, 2:5) do x,y
  glue_sorted_tetrahedron!(s, 6,7, x,y)
end
glue_sorted_tetrahedron!(s, 6,7, 5,1)

# "The link of the vertex a is the union of the 2-simplices bfg and efg."
# bfg=[267], efg=[567]
@test link(s,1)[3] == [1, 11]

# "The link of the vertex f is the cone: abcdea * g."
Lkf = link(s,6)
es = union(∂(0, s, Tri(Lkf[3])), ∂(1, s, Tri(Lkf[3])), ∂(2, s, Tri(Lkf[3])))
vs = union(∂(0, s, E(es)), ∂(1, s, E(es)))
@test Set(es) == Set(Lkf[2])
@test Set(vs) == Set(Lkf[1])
@test Set(Lkf[1]) == Set([1,2,3,4,5,7])

# 1D simplicial set tests, which are not given in Munkres:
VE(vs, es) = [V(vs), es == :∅ ? E[] : E(es)]
s = path_graph(DeltaSet1D, 8)
@test St(s, 1) == VE([1],     [1])
@test St̄(s, 1) == VE([1,2],   [1])
@test Lk(s, 1) == VE([2],     :∅)
# Observe that the neighborhood around 2 is not isomorphic to that around 1.
@test St(s, 2) == VE([2],     [1,2])
@test St̄(s, 2) == VE([2,3,1], [1,2])
@test Lk(s, 2) == VE([3,1],   :∅)

s = cycle_graph(DeltaSet1D, 8)
@test St(s, 1) == VE([1],     [8,1])
@test St̄(s, 1) == VE([1,2,8], [8,1])
@test Lk(s, 1) == VE([2,8],   :∅)
# Observe that the neighborhood around 2 is isomorphic to that around 1.
@test St(s, 2) == VE([2],     [1,2])
@test St̄(s, 2) == VE([2,3,1], [1,2])
@test Lk(s, 2) == VE([3,1],   :∅)

# Test boundary indices accessors.
##################################
# Line:
s = path_graph(DeltaSet1D, 8)
bvs, bes = boundary_inds(Val{0}, s), boundary_inds(Val{1}, s)
@test issetequal(bvs, [1,8])
@test issetequal(bes, [1,7])

# Circle:
s = cycle_graph(DeltaSet1D, 8)
bvs, bes = boundary_inds(Val{0}, s), boundary_inds(Val{1}, s)
@test issetequal(bvs, [])
@test issetequal(bes, [])

# Disconnected Lines:
s = path_graph(DeltaSet1D, 8)
s = apex(coproduct(s,s))
bvs, bes = boundary_inds(Val{0}, s), boundary_inds(Val{1}, s)
@test issetequal(bvs, [1,8,9,16])
@test issetequal(bes, [1,7,8,14])

# Test REPL IO
##############

function show_to_string(s)
  buf = IOBuffer()
  show(buf, MIME(Symbol("text/plain")), s)
  String(take!(buf))
end

@test show_to_string(path_graph(DeltaSet1D, 32)) ==
  "1D Delta Set with 32 vertices and 31 edges."

@test show_to_string(triangulated_grid(64,64,8,8,Point2d)) ==
  "2D Delta Set with 81 vertices, 208 edges, and 128 triangles."

s = CombinatorialSpaces.Meshes.single_tetrahedron()[1]
s[:edge_orientation] = false; s[:tri_orientation] = false; orient!(s);

@test show_to_string(apex(coproduct([s for _ in 1:32]))) ==
  "3D Delta Set with 128 vertices, 192 edges, 128 triangles, and 32 tetrahedra."

end
