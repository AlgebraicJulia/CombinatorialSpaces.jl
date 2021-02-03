module TestDualSimplicialSets
using Test

using Catlab.CategoricalAlgebra.CSets
using CombinatorialSpaces

# 1D dual simplicial sets
#########################

primal_s = DeltaSet1D()
add_vertices!(primal_s, 5)
add_edges!(primal_s, 1:4, repeat([5], 4))
s = DualDeltaSet1D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s)
@test nparts(s, :DualE) == 2 * ne(primal_s)
dual_es = incident(s, s[5, :vertex_center], :D_∂v1)
@test length(dual_es) == 4
@test s[dual_es, :D_∂v0] == s[1:4, :edge_center]

end
