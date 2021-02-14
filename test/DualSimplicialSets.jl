module TestDualSimplicialSets
using Test

using StaticArrays

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

# 1D embedded dual complex
#-------------------------

primal_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(primal_s, [1,2], [2,3], edge_orientation=true)
s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
@test dual_point(s, edge_center(s, [1,2])) ≈ [Point2D(0.5,0), Point2D(0,1)]
@test volume(s, E(1:2)) ≈ [1.0, 2.0]
@test volume(s, elementary_duals(s, V(2))) ≈ [0.5, 1.0]

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

dual_vs = elementary_duals(2,s,2)
@test dual_vs == [triangle_center(s,2)]
@test elementary_duals(s, Tri(2)) == DualV(dual_vs)
@test s[elementary_duals(1,s,2), :D_∂v1] == [edge_center(s,2)]
@test s[elementary_duals(1,s,3), :D_∂v1] == repeat([edge_center(s,3)], 2)
@test [length(elementary_duals(s, V(i))) for i in 1:4] == [4,2,4,2]

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

# 2D embedded dual complex
#-------------------------

# Single triangle: numerical example from Gillett's notes on DEC, §2.13.
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

subdivide_duals!(s, Circumcenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/2, 1/2)

end
