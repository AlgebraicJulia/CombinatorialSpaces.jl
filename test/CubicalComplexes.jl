using Test
using CairoMakie

include("../src/CubicalComplexes.jl")
s = EmbeddedCubicalComplex2D()

add_vertex!(s, Point3d(0,0,0))
add_vertex!(s, Point3d(1,0,0))
add_vertex!(s, Point3d(1,1,0))
add_vertex!(s, Point3d(0,1,0))

glue_quad!(s, 3, 4, 1, 2)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

@test nv(s) == 4
@test ne(s) == 4
@test nquads(s) == 1

@test point(s, 1) == Point3d(0,0,0)
@test edge_length(s, 1) == 1.0

@test quad_area(s, 1) == 1.0

@test s.∂v0 == [3,1,1,2]
@test s.∂v1 == [4,4,2,3]

@test edge_vertices(s, 1) == [3,4]
@test quad_edges(s, 1) == [1,2,3,4]
@test quad_vertices(s, 1) == [3,4,1,2]
@test quad_edge_orients(s, 1) == [1,-1,1,1]

d0 = exterior_derivative(Val(0), s)
d1 = exterior_derivative(Val(1), s)

@test all(0 .== d1 * d0)

grid = construct_grid(10, 10, 11, 11)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, grid)
fig

d0 = exterior_derivative(Val(0), grid)
d1 = exterior_derivative(Val(1), grid)

@test all(0 .== d1 * d0)
