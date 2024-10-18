module TestSimplicialComplexes
using Test 
using CombinatorialSpaces
using Catlab:@acset

# Triangulated commutative square.
ss = DeltaSet2D()
add_vertices!(ss, 4)
glue_triangle!(ss, 1, 2, 3)
glue_triangle!(ss, 1, 4, 3)

sc = SimplicialComplex(ss)
@test nv(sc) == 4 && ne(sc) == 5 && ntriangles(sc) == 2
sc′ = SimplicialComplex(DeltaSet2D,sc.cache).delta_set
@test nv(sc′) == 4 && ne(sc′) == 5 && ntriangles(sc′) == 2 #identifies this up to iso
#awkward height=0 edge case, technically can think of the empty sset as -1-dimensional.
sc′′=SimplicialComplex(Trie{Int,Int}()) 
@test dimension(sc′′) == 0 && nv(sc′′) == 0

vl = VertexList(ss,Simplex{2}(1))
@test vl.vs == [1,2,3]
@test has_simplex(sc,vl)
@test !has_simplex(sc,VertexList([1,2,4]))
@test sc[vl] == Simplex{2}(1)

vl′ = VertexList([1,2])∪VertexList([2,3])
@test has_simplex(sc,vl′)
@test !has_simplex(sc,VertexList([1,2])∪VertexList([2,4]))

p = GeometricPoint([0.2,0,0.5,0.3])
q = GeometricPoint([0,0.2,0.5,0.3])
r = GeometricPoint([0.5,0,0,0.5])
s = GeometricPoint([0.5,0,0.5,0])
t = GeometricPoint([0,0.5,0,0.5])
@test has_point(sc,p) && !has_point(sc,q) 
@test has_span(sc,[r,s]) && !has_span(sc,[r,t])

Δ⁰ = SimplicialComplex(@acset DeltaSet0D begin V=1 end)
Δ¹ = SimplicialComplex(@acset DeltaSet1D begin V=2; E=1; ∂v0 = [2]; ∂v1 = [1] end)
f = GeometricMap(Δ⁰,Δ¹,[1/3,2/3])
A  = [0.2 0.4
      0   0
      0.5 0
      0.3 0.6]
g = GeometricMap(Δ¹,sc,A)
@test A == as_matrix(g)
h = compose(f,g)
@test as_matrix(h) ≈ [1/3, 0, 1/6, 1/2]
isc = id(sc)
@test as_matrix(h) == as_matrix(compose(h,isc))


end