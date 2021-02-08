module TestDualSimplicialSets
using Test

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces

# 1D dual complex
#################

primal_s = DeltaSet1D()
add_vertices!(primal_s, 5)
add_edges!(primal_s, 1:4, repeat([5], 4))
s = DeltaDualComplex1D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s)
@test nparts(s, :DualE) == 2 * ne(primal_s)

@test elementary_duals(1,s,4) == [edge_center(s, 4)]
dual_es = elementary_duals(0,s,5)
@test length(dual_es) == 4
@test s[dual_es, :D_∂v0] == edge_center(s, 1:4)

# 1D oriented dual complex
#-------------------------

primal_s = OrientedDeltaSet1D{Bool}()
add_vertices!(primal_s, 3)
add_edges!(primal_s, [1,2], [2,3], edge_orientation=[true,false])
s = OrientedDeltaDualComplex1D{Bool}(primal_s)
@test s[only(elementary_duals(0,s,1)), :D_edge_orientation] == true
@test s[only(elementary_duals(0,s,3)), :D_edge_orientation] == true

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

@test elementary_duals(2,s,2) == [triangle_center(s,2)]
@test s[elementary_duals(1,s,2), :D_∂v1] == [edge_center(s,2)]
@test s[elementary_duals(1,s,3), :D_∂v1] == repeat([edge_center(s,3)], 2)
@test [length(elementary_duals(0,s,i)) for i in 1:4] == [4,2,4,2]

# 2D oriented dual complex
#-------------------------

# Triangulated square with consistent orientation.
primal_s = OrientedDeltaSet2D{Bool}()
add_vertices!(primal_s, 4)
glue_triangle!(primal_s, 1, 2, 3)
glue_triangle!(primal_s, 1, 3, 4)
primal_s[:edge_orientation] = true
primal_s[:tri_orientation] = true
s = OrientedDeltaDualComplex2D{Bool}(primal_s)
@test sum(s[:D_tri_orientation]) == nparts(s, :DualTri) ÷ 2
@test [sum(s[elementary_duals(0,s,i), :D_tri_orientation])
       for i in 1:4] == [2,1,2,1]

end
