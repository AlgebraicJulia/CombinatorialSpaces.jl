module TestFastDECOperators
using Test

using LinearAlgebra: Diagonal
using SparseArrays, StaticArrays

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces

function is_semi_simplicial(s::HasDeltaSet, n::Int)
  all(∂(n-1, i, s, ∂(n, j, s)) == ∂(n-1, j-1, s, ∂(n, i, s))
      for i in 0:n for j in (i+1):n)
end

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

# 1D simplicial sets
####################


# 1D oriented simplicial sets
#----------------------------
s = OrientedDeltaSet1D{Bool}()
add_vertices!(s, 4)
add_edges!(s, [1,2,3], [2,3,4], edge_orientation=[true,false,true])

# Boundary operator, sparse vectors.
vvec = Matrix(fast_boundary(1, s))[:, 1]
@test vvec == [-1,1,0,0]
vvec = fast_boundary(1, s) * [1,-1,1]
@test vvec == [-1,0,0,1]
B = fast_boundary(1, s)
@test issparse(B)
@test B*[1,-1,1] == [-1,0,0,1]

# Exterior derivative.
@test fast_d(s, VForm([1,1,1,4]))::EForm == EForm([0,0,3])
@test fast_d(s, VForm([4,1,0,0])) == EForm([-3,1,0])
@test fast_d(0, s) == B'

# 1D embedded simplicial sets
#----------------------------

# 2D simplicial sets
####################

# 2D oriented simplicial sets
#----------------------------

# Triangle with matching edge orientations.
s = OrientedDeltaSet2D{Bool}()
add_vertices!(s, 3)
add_sorted_edges!(s, [1,2,3], [2,3,1], edge_orientation=[true,true,false])
glue_triangle!(s, 1, 2, 3)
@test orient_component!(s, 1, true)
@test orientation(s, Tri(1)) == true
@test Matrix(fast_boundary(2, s))[:, 1] == [1,1,1]
@test fast_d(1, s) * [45,3,34] == [82]

# Triangulated square with consistent orientation.
s = OrientedDeltaSet2D{Bool}()
add_vertices!(s, 4)
glue_triangle!(s, 1, 2, 3)
glue_triangle!(s, 1, 3, 4)
s[:edge_orientation] = true
@test orient!(s)
@test orientation(s, Tri(1:2)) == trues(2)
@test Matrix(fast_boundary(2, s))[:, 1] == [1,1,-1,0,0]
@test fast_d(s, EForm([45,3,34,0,0]))::TriForm == TriForm([14, 34]) # == [45+3-34, 34]
@test fast_d(s, EForm([45,3,34,17,5])) == TriForm([14, 46]) # == [45+3-34, 34+17-5]
@test fast_d(1, s) == fast_boundary(2, s)'
@test is_manifold_like(s)
@test isempty(nonboundaries(s)[1])
@test isempty(nonboundaries(s)[2])

# 1D dual complex
#################

# 1D oriented dual complex
#-------------------------

primal_s = OrientedDeltaSet1D{Bool}()
add_vertices!(primal_s, 3)
add_edges!(primal_s, [1,2], [2,3], edge_orientation=[true,false])
s = OrientedDeltaDualComplex1D{Bool}(primal_s)
@test s[only(elementary_duals(0,s,1)), :D_edge_orientation] == true
@test s[only(elementary_duals(0,s,3)), :D_edge_orientation] == true

@test dual_boundary(1,s) == fast_boundary(1,s)'
@test fast_dual_derivative(0,s) == -fast_d(0,s)'

# 1D embedded dual complex
#-------------------------

# Path graph on 3 vertices with irregular lengths.
explicit_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(explicit_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(explicit_s, [1,2], [2,3], edge_orientation=true)

# Path graph on 3 vertices without orientation set beforehand.
implicit_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(implicit_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(implicit_s, [1,2], [2,3])

for primal_s in [explicit_s, implicit_s]
  s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
  subdivide_duals!(s, Barycenter())
  @test dual_point(s, edge_center(s, [1,2])) ≈ [Point2D(0.5,0), Point2D(0,1)]
  @test volume(s, E(1:2)) ≈ [1.0, 2.0]
  @test volume(s, elementary_duals(s, V(2))) ≈ [0.5, 1.0]
  @test fast_hodge_star(0,s) ≈ Diagonal([0.5, 1.5, 1.0])
  @test fast_hodge_star(1,s) ≈ Diagonal([1, 0.5])
  @test fast_hodge_star(0,s) * VForm([0,2,0]) ≈ DualForm{1}([0,3,0])

  @test fast_wedge_product(s, VForm([1,2,3]), VForm([3,4,7])) ≈ [3,8,21]
  @test fast_wedge_product(s, VForm([1,2,3]), VForm([3,4,7]))::VForm ≈ VForm([3,8,21])
  @test fast_wedge_product(s, VForm([1,1,1]), EForm([2.5, 5.0]))::EForm ≈ EForm([2.5, 5.0])
  @test fast_wedge_product(s, VForm([1,1,0]), EForm([2.5, 5.0])) ≈ EForm([2.5, 2.5])
  vform, eform = VForm([1.5, 2, 2.5]), EForm([13, 7])
  @test fast_wedge_product(s, vform, eform) ≈ ∧(s, eform, vform)
end

# 2D dual complex
#################

# 2D oriented dual complex
#-------------------------

# Triangulated square with consistent orientation.
explicit_s = OrientedDeltaSet2D{Bool}()
add_vertices!(explicit_s, 4)
glue_triangle!(explicit_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(explicit_s, 1, 3, 4, tri_orientation=true)
explicit_s[:edge_orientation] = true

# Triangulated square without explicit orientation set beforehand.
implicit_s = OrientedDeltaSet2D{Bool}()
add_vertices!(implicit_s, 4)
glue_triangle!(implicit_s, 1, 2, 3)
glue_triangle!(implicit_s, 1, 3, 4)

for primal_s in [explicit_s, implicit_s]
  s = OrientedDeltaDualComplex2D{Bool}(primal_s)
  for k in 1:2
    # Desbrun, Kanso, Tong 2008, Equation 4.2.
    @test fast_dual_derivative(2-k,s) == (-1)^k * fast_d(k-1,s)'
  end
end

# 2D embedded dual complex
#-------------------------

# Single triangle: numerical example from Gillette's notes on DEC, §2.13.
#
# Compared with Gillette, edges #2 and #3 are swapped in the ordering, which
# changes the discrete exterior derivative and other operators. The numerical
# values remain the same, as we verify.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(0,0), Point2D(1,0), Point2D(0,1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)

subdivide_duals!(s, Barycenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/3, 1/3)
@test volume(s, Tri(1)) ≈ 1/2
@test volume(s, elementary_duals(s, V(1))) ≈ [1/12, 1/12]
@test [sum(volume(s, elementary_duals(s, V(i)))) for i in 1:3] ≈ [1/6, 1/6, 1/6]

# These values are consistent with the Gillette paper, as described above
@test fast_hodge_star(0,s) ≈ Diagonal([1/6, 1/6, 1/6])
@test fast_hodge_star(1,s; hodge=DiagonalHodge()) ≈ Diagonal([√5/6, 1/6, √5/6])

# This test is consistent with Ayoub et al 2020 page 13 (up to permutation of
# vertices)
@test fast_hodge_star(1,s) ≈ [1/3 0.0 1/6;
                             0.0 1/6 0.0;
                             1/6 0.0 1/3]

# Test consistency regardless of base triangle orientation (relevant for
# geometric hodge star)
flipped_ps = deepcopy(primal_s)
orient_component!(flipped_ps, 1, false)
flipped_s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(flipped_ps)
subdivide_duals!(flipped_s, Barycenter())
@test fast_hodge_star(1,s) ≈ fast_hodge_star(1,flipped_s)

# NOTICE:
# Tests beneath this comment are not backed up by any external source, and are
# included to determine consistency as the operators are modified.
#
# If a test beneath this comment fails due to a new implementation, it is
# possible that the values for the test itself need to be modified.
@test fast_inv_hodge_star(2, s)[1,1] ≈ 0.5
@test (fast_inv_hodge_star(2, s) * [2.0])[1,1] ≈ 1.0
@test fast_inv_hodge_star(1, s, hodge=DiagonalHodge()) ≈ Diagonal([-6/√5, -6, -6/√5])
@test fast_inv_hodge_star(1, s, hodge=DiagonalHodge()) * [0.5, 2.0, 0.5] ≈ [-3/√5, -12.0, -3/√5]
@test fast_hodge_star(0, s) * VForm([1,2,3]) ≈ DualForm{2}([1/6, 1/3, 1/2]) 

subdivide_duals!(s, Circumcenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/2, 1/2)
@test fast_hodge_star(0,s) ≈ Diagonal([1/4, 1/8, 1/8])
@test fast_hodge_star(1,s) ≈ Diagonal([0.5, 0.0, 0.5])

subdivide_duals!(s, Incenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/(2+√2), 1/(2+√2))
@test isapprox(fast_hodge_star(0,s), Diagonal([0.146, 0.177, 0.177]), atol=1e-3)
@test isapprox(fast_hodge_star(1,s), [0.293 0.000 0.207;
                                         0.000 0.207 0.000;
                                         0.207 0.000 0.293], atol=1e-3)

# Triangulated square with consistent orientation.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 4, point=[Point2D(-1,+1), Point2D(+1,+1),
                                  Point2D(+1,-1), Point2D(-1,-1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(primal_s, 1, 3, 4, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())

@test fast_wedge_product(s, VForm([2,2,2,2]), TriForm([2.5, 5]))::TriForm ≈ TriForm([5.0, 10.0])
vform, triform = VForm([1.5, 2, 2.5, 3]), TriForm([5, 7.5])
@test fast_wedge_product(s, vform, triform) ≈ fast_wedge_product(s, triform, vform)
eform1, eform2 = EForm([1.5, 2, 2.5, 3, 3.5]), EForm([3, 7, 10, 11, 15])
@test fast_wedge_product(s, eform1, eform1)::TriForm ≈ TriForm([0, 0])
@test fast_wedge_product(s, eform1, eform2) ≈ -fast_wedge_product(s, eform2, eform1)

end