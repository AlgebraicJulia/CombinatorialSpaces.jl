using Test

include("../src/CubicalComplexes.jl")

s = EmbeddedCubicalComplex2D()

add_vertex!(s, Point3d(0,0,0))
add_vertex!(s, Point3d(1,0,0))
add_vertex!(s, Point3d(1,1,0))
add_vertex!(s, Point3d(0,1,0))

glue_quad!(s, 3, 4, 1, 2)

@test nv(s) == 4
@test ne(s) == 4
@test nquads(s) == 1

@test point(s, 1) == Point3d(0,0,0)
@test edge_length(s, 1) == 1.0

@test quad_area(s, 1) == 1.0

@test s.∂v0 == [1,1,2,3]
@test s.∂v1 == [2,4,3,4]

@test edge_vertices(s, 1) == [1,2]
@test quad_vertices(s, 1) == [1,2,3,4]
