module TestDualSimplicialSets
using Test

using LinearAlgebra: Diagonal
using SparseArrays, StaticArrays

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

# 1D dual complex
#################

primal_s = DeltaSet1D()
add_vertices!(primal_s, 5)
add_edges!(primal_s, 1:4, repeat([5], 4))
s = DeltaDualComplex1D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s)
@test nparts(s, :DualE) == 2 * ne(primal_s)

dual_v = elementary_duals(1,s,4)
@test dual_v == [edge_center(s, 4)]
@test elementary_duals(s, E(4)) == DualV(dual_v)

dual_es = elementary_duals(0,s,5)
@test length(dual_es) == 4
@test s[dual_es, :D_∂v0] == edge_center(s, 1:4)
@test elementary_duals(s, V(5)) == DualE(dual_es)

# 1D oriented dual complex
#-------------------------

primal_s = OrientedDeltaSet1D{Bool}()
add_vertices!(primal_s, 3)
add_edges!(primal_s, [1,2], [2,3], edge_orientation=[true,false])
s = OrientedDeltaDualComplex1D{Bool}(primal_s)
@test s[only(elementary_duals(0,s,1)), :D_edge_orientation] == true
@test s[only(elementary_duals(0,s,3)), :D_edge_orientation] == true

@test ∂(s, DualChain{1}([1,0,1])) isa DualChain{0}
@test d(s, DualForm{0}([1,1])) isa DualForm{1}
@test dual_boundary(1,s) == ∂(1,s)'
@test dual_derivative(0,s) == -d(0,s)'

# 1D embedded dual complex
#-------------------------

# Path graph on 3 vertices with irregular lengths.
primal_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(primal_s, [1,2], [2,3], edge_orientation=true)
s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
@test dual_point(s, edge_center(s, [1,2])) ≈ [Point2D(0.5,0), Point2D(0,1)]
@test volume(s, E(1:2)) ≈ [1.0, 2.0]
@test volume(s, elementary_duals(s, V(2))) ≈ [0.5, 1.0]
@test ⋆(0,s) ≈ Diagonal([0.5, 1.5, 1.0])
@test ⋆(1,s) ≈ Diagonal([1, 0.5])
@test ⋆(s, VForm([0,2,0]))::DualForm{1} ≈ DualForm{1}([0,3,0])

@test ∧(0,0,s, [1,2,3], [3,4,7]) ≈ [3,8,21]
@test ∧(s, VForm([1,2,3]), VForm([3,4,7]))::VForm ≈ VForm([3,8,21])
@test ∧(s, VForm([1,1,1]), EForm([2.5, 5.0]))::EForm ≈ EForm([2.5, 5.0])
@test ∧(s, VForm([1,1,0]), EForm([2.5, 5.0])) ≈ EForm([2.5, 2.5])
vform, eform = VForm([1.5, 2, 2.5]), EForm([13, 7])
@test ∧(s, vform, eform) ≈ ∧(s, eform, vform)

# Path graph on 5 vertices with regular lengths.
#
# Equals the graph Laplacian of the underlying graph, except at the boundary.
# Note that the DEC Laplace-Beltrami operator is *not* symmetric.
primal_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_s, 5, point=[Point2D(i,0) for i in -2:2])
add_edges!(primal_s, 1:4, 2:5, edge_orientation=true)
s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
@test Δ(s, VForm([0,0,1,0,0])) ≈ VForm([0,1,-2,1,0])
@test Δ(0,s) ≈ [-2  2  0  0  0;
                 1 -2  1  0  0;
                 0  1 -2  1  0;
                 0  0  1 -2  1;
                 0  0  0  2 -2]

# 2D dual complex
#################

# Triangulated square.
primal_s = DeltaSet2D()
add_vertices!(primal_s, 4)
glue_triangle!(primal_s, 1, 2, 3)
glue_triangle!(primal_s, 1, 3, 4)
s = DeltaDualComplex2D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s) + ntriangles(primal_s)
@test nparts(s, :DualE) == 2*ne(primal_s) + 6*ntriangles(primal_s)
@test nparts(s, :DualTri) == 6*ntriangles(primal_s)
@test primal_vertex(s, subsimplices(s, Tri(1)))::V == V([1,1,2,2,3,3])

dual_vs = elementary_duals(2,s,2)
@test dual_vs == [triangle_center(s,2)]
@test elementary_duals(s, Tri(2)) == DualV(dual_vs)
@test s[elementary_duals(1,s,2), :D_∂v1] == [edge_center(s,2)]
@test s[elementary_duals(1,s,3), :D_∂v1] == repeat([edge_center(s,3)], 2)
@test [length(elementary_duals(s, V(i))) for i in 1:4] == [4,2,4,2]
@test dual_triangle_vertices(s, 1) == [1,7,10]


# 2D oriented dual complex
#-------------------------

# Triangulated square with consistent orientation.
primal_s = OrientedDeltaSet2D{Bool}()
add_vertices!(primal_s, 4)
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(primal_s, 1, 3, 4, tri_orientation=true)
primal_s[:edge_orientation] = true
s = OrientedDeltaDualComplex2D{Bool}(primal_s)
@test sum(s[:D_tri_orientation]) == nparts(s, :DualTri) ÷ 2
@test [sum(s[elementary_duals(0,s,i), :D_tri_orientation])
       for i in 1:4] == [2,1,2,1]
@test sum(s[elementary_duals(1,s,3), :D_edge_orientation]) == 1

for k in 0:1
  @test dual_boundary(2-k,s) == (-1)^k * ∂(k+1,s)'
end
for k in 1:2
  # Desbrun, Kanso, Tong 2008, Equation 4.2.
  @test dual_derivative(2-k,s) == (-1)^k * d(k-1,s)'
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
@test ⋆(0,s) ≈ Diagonal([1/6, 1/6, 1/6])
@test ⋆(1,s) ≈ Diagonal([√5/6, 1/6, √5/6])
@test ⋆(s, VForm([1,2,3]))::DualForm{2} ≈ DualForm{2}([1/6, 1/3, 1/2])
@test isapprox(δ(1,s), [ 2.236  0  2.236;
                        -2.236  1  0;
                         0     -1 -2.236], atol=1e-3)
@test δ(s, EForm([0.5,1.5,0.5])) isa VForm

subdivide_duals!(s, Circumcenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/2, 1/2)
@test ⋆(0,s) ≈ Diagonal([1/4, 1/8, 1/8])
@test ⋆(1,s) ≈ Diagonal([0.5, 0.0, 0.5])
@test δ(1,s) ≈ [ 2  0  2;
                -4  0  0;
                 0  0 -4]

subdivide_duals!(s, Incenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/(2+√2), 1/(2+√2))
@test isapprox(⋆(0,s), Diagonal([0.146, 0.177, 0.177]), atol=1e-3)
@test isapprox(⋆(1,s), Diagonal([0.359, 0.207, 0.359]), atol=1e-3)
@test isapprox(δ(1,s), [ 2.449  0      2.449;
                        -2.029  1.172  0;
                         0     -1.172 -2.029], atol=1e-3)

# Triangulated square with consistent orientation.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 4, point=[Point2D(-1,+1), Point2D(+1,+1),
                                  Point2D(+1,-1), Point2D(-1,-1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(primal_s, 1, 3, 4, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())

x̂, ŷ, zero = @SVector([1,0]), @SVector([0,1]), @SVector([0,0])
@test ♭(s, DualVectorField([x̂, -x̂])) ≈ EForm([2,0,0,2,0])
@test ♭(s, DualVectorField([ŷ, -ŷ])) ≈ EForm([0,-2,0,0,2])
@test ♭(s, DualVectorField([(x̂-ŷ)/√2, (x̂-ŷ)/√2]))[3] ≈ 2*√2
@test ♭(s, DualVectorField([(x̂-ŷ)/√2, zero]))[3] ≈ √2

@test ∧(s, VForm([2,2,2,2]), TriForm([2.5, 5]))::TriForm ≈ TriForm([2.5, 5])
vform, triform = VForm([1.5, 2, 2.5, 3]), TriForm([5, 7.5])
@test ∧(s, vform, triform) ≈ ∧(s, triform, vform)
eform1, eform2 = EForm([1.5, 2, 2.5, 3, 3.5]), EForm([3, 7, 10, 11, 15])
@test ∧(s, eform1, eform1)::TriForm ≈ TriForm([0, 0])
@test ∧(s, eform1, eform2) ≈ -∧(s, eform2, eform1)

X♭, α = EForm([1.5, 2, 2.5, 3, 3.5]), DualForm{1}([3, 7, 10, 11, 15])
@test interior_product(s, X♭, α) isa DualForm{0}
@test length(interior_product_flat(1,s, X♭.data, α.data)) == 2
@test lie_derivative(s, X♭, α) isa DualForm{1}
@test length(lie_derivative_flat(1,s, X♭.data, α.data)) == 5

end
