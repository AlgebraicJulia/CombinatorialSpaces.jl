using Test
using CairoMakie
using Distributions

include("../../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 2, 2)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/SquareGrid.png", fig)

# Counts
@test nx(s) == 2
@test ny(s) == 2

@test nxquads(s) == 1
@test nyquads(s) == 1

@test length(points(s)) == 4

@test nv(s) == 4
@test ne(s) == 4
@test nquads(s) == 1

@test nhe(s) == 2
@test nve(s) == 2

@test ndv(s) == 1
@test nde(s) == 4
@test ndquads(s) == 4

# Working with one form wonkiness
@test is_hedge(CartesianIndex(1, 1, 1))
@test !is_hedge(CartesianIndex(2, 1, 1))

@test is_vedge(CartesianIndex(2, 1, 1))
@test !is_vedge(CartesianIndex(1, 1, 1))

@test is_d_vedge(CartesianIndex(1, 1, 1))
@test !is_d_vedge(CartesianIndex(2, 1, 1))

@test is_d_hedge(CartesianIndex(2, 1, 1))
@test !is_d_hedge(CartesianIndex(1, 1, 1))

# Tensorfication
@test size(tensorfy(Val(0), s, ones(nv(s)))) == (2, 2)

@test length(tensorfy(Val(1), s, ones(nv(s)))) == 2
@test size(hdeges(tensorfy(Val(1), s, ones(ne(s))))) == (1, 2)
@test size(vedges(tensorfy(Val(1), s, ones(ne(s))))) == (2, 1)

@test size(tensorfy(Val(2), s, ones(nquads(s)))) == (1, 1)

# Retrieving topological information
@test point(s, 1) == Point3d(0,0,0)
@test coord_to_vert(s, CartesianIndex(1, 1)) == 1
@test coord_to_vert(s, CartesianIndex(2, 1)) == 2
@test coord_to_vert(s, CartesianIndex(1, 2)) == 3
@test coord_to_vert(s, CartesianIndex(2, 2)) == 4

# Horizontal
@test coord_to_edge(s, CartesianIndex(1, 1, 1)) == 1
@test coord_to_edge(s, CartesianIndex(1, 1, 2)) == 2

# Vertical
@test coord_to_edge(s, CartesianIndex(2, 1, 1)) == 3
@test coord_to_edge(s, CartesianIndex(2, 2, 1)) == 4

@test coord_to_quad(s, CartesianIndex(1, 1)) == 1

@test src(CartesianIndex(1,1,1)) == CartesianIndex(1,1)
@test tgt(CartesianIndex(1,1,1)) == CartesianIndex(2,1)

@test src(CartesianIndex(2,1,1)) == CartesianIndex(1,1)
@test tgt(CartesianIndex(2,1,1)) == CartesianIndex(1,2)

# Bottom left, bottom right, top right, top left
@test map(v -> coord_to_vert(s, v), quad_vertices(CartesianIndex(1,1))) == [1, 2, 4, 3]

# Bottom, top, left, right
@test map(e -> coord_to_edge(s, e), quad_edges(CartesianIndex(1,1))) == [1, 2, 3, 4]

@test hedge_len(s, CartesianIndex(1,1,1)) == 10.0
@test vedge_len(s, CartesianIndex(2,1,1)) == 10.0

@test quad_width(s, CartesianIndex(1,1)) == 10.0
@test quad_height(s, CartesianIndex(1,1)) == 10.0

@test quad_area(s, CartesianIndex(1,1)) == 100.0
@test safe_quad_area(s, CartesianIndex(1,1)) == 100.0
@test safe_quad_area(s, CartesianIndex(2,2)) == 0.0

@test coord_to_quad(s, he_t_q(CartesianIndex(1,1,1))) == 1
@test coord_to_quad(s, he_b_q(CartesianIndex(1,1,2))) == 1

@test coord_to_quad(s, ve_l_q(CartesianIndex(2,2,1))) == 1
@test coord_to_quad(s, ve_r_q(CartesianIndex(2,1,1))) == 1

@test coord_to_edge(s, v_l_he(CartesianIndex(2,1))) == 1
@test coord_to_edge(s, v_r_he(CartesianIndex(1,1))) == 1

@test coord_to_edge(s, v_b_ve(CartesianIndex(1,2))) == 3
@test coord_to_edge(s, v_t_ve(CartesianIndex(1,1))) == 3

@test coord_to_quad(s, v_bl_q(CartesianIndex(2,2))) == 1
@test coord_to_quad(s, v_br_q(CartesianIndex(1,2))) == 1
@test coord_to_quad(s, v_tl_q(CartesianIndex(2,1))) == 1
@test coord_to_quad(s, v_tr_q(CartesianIndex(1,1))) == 1

@test dual_point(s, CartesianIndex(1,1)) == Point3d(5,5,0)

for e in edges(s)
  @test d_edge_len(s, CartesianIndex(e)) == 5.0
end

for dq in vertices(s)
  @test d_quad_area(s, CartesianIndex(dq)) == 25.0
end

### NON-UNIFORM MESH TESTS

ps = Point3d[]
for y in [0, 1, 3, 6, 10]
  for x in [0, 1, 3, 6, 10]
    push!(ps, Point3d(x, y, 0))
  end
end

s = EmbeddedCubicalComplex2D(5, 5, ps);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/IrregularGrid.png", fig)

@test nv(s) == 25
@test ne(s) == 40
@test nquads(s) == 16

@test nhe(s) == 20
@test nve(s) == 20

@test coords_to_quads(s, edge_quads(hedge(4, 1))) == [4, 0]
@test coords_to_quads(s, edge_quads(hedge(4, 2))) == [8, 4]
@test coords_to_quads(s, edge_quads(hedge(4, 5))) == [20, 16]

@test coords_to_quads(s, edge_quads(vedge(1, 1))) == [0, 1]
@test coords_to_quads(s, edge_quads(vedge(2, 1))) == [1, 2]
@test coords_to_quads(s, edge_quads(vedge(5, 1))) == [4, 5]

@test edge_len(s, hedge(1, 1)) == 1.0
@test edge_len(s, hedge(2, 1)) == 2.0
@test edge_len(s, hedge(3, 1)) == 3.0
@test edge_len(s, hedge(4, 1)) == 4.0

@test edge_len(s, vedge(1, 1)) == 1.0
@test edge_len(s, vedge(1, 2)) == 2.0
@test edge_len(s, vedge(1, 3)) == 3.0
@test edge_len(s, vedge(1, 4)) == 4.0

@test quad_area(s, CartesianIndex(1,1)) == 1.0
@test quad_area(s, CartesianIndex(2,2)) == 4.0
@test quad_area(s, CartesianIndex(3,3)) == 9.0
@test quad_area(s, CartesianIndex(4,4)) == 16.0

@test coords_to_verts(s, quad_vertices(CartesianIndex(1,1))) == [1,2,7,6]
@test coords_to_verts(s, quad_vertices(CartesianIndex(2,1))) == [2,3,8,7]
@test coords_to_verts(s, quad_vertices(CartesianIndex(1,2))) == [6,7,12,11]
@test coords_to_verts(s, quad_vertices(CartesianIndex(4,4))) == [19,20,25,24]

@test coords_to_edges(s, quad_edges(CartesianIndex(1,1))) == [1,5,21,22]
@test coords_to_edges(s, quad_edges(CartesianIndex(4,4))) == [16,20,39,40]