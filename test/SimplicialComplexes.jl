module TestSimplicialComplexes
using Test 
using CombinatorialSpaces.SimplicialSets
using CombinatorialSpaces.SimplicialComplexes

# Triangulated commutative square.
s = DeltaSet2D()
add_vertices!(s, 4)
glue_triangle!(s, 1, 2, 3)
glue_triangle!(s, 1, 4, 3)

sc = SimplicialComplex(s)
vl = VertexList(s,Simplex{2}(1))

@test vl.vs == [1,2,3]
@test has_simplex(sc,vl)
@test !has_simplex(sc,VertexList([1,2,4]))
@test sc[vl] == Simplex{2}(1)

vl′ = VertexList([1,2])∪VertexList([2,3])
@test has_simplex(sc,vl′)
@test !has_simplex(sc,VertexList([1,2])∪VertexList([2,4]))

end