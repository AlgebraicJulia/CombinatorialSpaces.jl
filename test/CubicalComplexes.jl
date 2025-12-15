using Test
using CairoMakie

include("../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 2, 2)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

@test nv(s) == 4
@test ne(s) == 4
@test nquads(s) == 1

@test nhe(s) == 2
@test nve(s) == 2

@test point(s, 1) == Point3d(0,0,0)
@test coord_to_vert(s, 1, 1) == 1
@test coord_to_vert(s, 2, 1) == 2
@test coord_to_vert(s, 1, 2) == 3
@test coord_to_vert(s, 2, 2) == 4

@test edge_length(s, 1) == 10.0

@test quad_area(s, 1) == 100.0

@test edge_vertices(s, 1) == [1,2]
@test quad_vertices(s, 1) == [1,2,4,3]
@test quad_edges(s, 1) == [1,2,3,4]

for e in edges(s)
  @test all(1 .<= edge_vertices(s, e) .<= nv(s))
end

for q in quadrilaterals(s)
  @test all(1 .<= quad_vertices(s, q) .<= nv(s))
end

for q in quadrilaterals(s)
  @test all(1 .<= quad_edges(s, q) .<= ne(s))
end

d0 = exterior_derivative(Val(0), s)
d1 = exterior_derivative(Val(1), s)

@test all(0 .== d1 * d0)

ps = Point3d[]
for y in [0, 1, 3, 6, 10]
  for x in [0, 1, 3, 6, 10]
    push!(ps, Point3d(x, y, 0))
  end
end

s = EmbeddedCubicalComplex2D(5, 5, ps)
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

@test nv(s) == 25
@test ne(s) == 40
@test nquads(s) == 16

@test nhe(s) == 20
@test nve(s) == 20

for e in edges(s)
  @test all(1 .<= edge_vertices(s, e) .<= nv(s))
end

for q in quadrilaterals(s)
  @test all(1 .<= quad_vertices(s, q) .<= nv(s))
end

for q in quadrilaterals(s)
  @test all(1 .<= quad_edges(s, q) .<= ne(s))
end

@test edge_length(s, 1) == 1.0
@test edge_length(s, 2) == 2.0
@test edge_length(s, 3) == 3.0
@test edge_length(s, 4) == 4.0

@test quad_area(s, 1) == 1.0
@test quad_area(s, 6) == 4.0
@test quad_area(s, 11) == 9.0
@test quad_area(s, 16) == 16.0

@test quad_vertices(s, 1) == [1,2,7,6]
@test quad_vertices(s, 2) == [2,3,8,7]
@test quad_vertices(s, 5) == [6,7,12,11]
@test quad_vertices(s, 16) == [19,20,25,24]

@test quad_edges(s, 1) == [1,5,21,22]
@test 21 - nhe(s) == 1

@test quad_edges(s, 16) == [16,20,39,40]

d0 = exterior_derivative(Val(0), s)
d1 = exterior_derivative(Val(1), s)

@test all(0 .== d1 * d0)
