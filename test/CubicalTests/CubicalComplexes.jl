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
@test is_xedge(1)
@test !is_xedge(2)

@test is_yedge(2)
@test !is_yedge(1)

@test is_d_yedge(1)
@test !is_d_yedge(2)

@test is_d_xedge(2)
@test !is_d_xedge(1)

# Tensorfication
@test size(tensorfy(Val(0), s, ones(nv(s)))) == (2, 2)

@test length(tensorfy(Val(1), s, ones(nv(s)))) == 2
@test size(xedges(tensorfy(Val(1), s, ones(ne(s))))) == (1, 2)
@test size(yedges(tensorfy(Val(1), s, ones(ne(s))))) == (2, 1)

@test size(tensorfy(Val(2), s, ones(nquads(s)))) == (1, 1)

# Retrieving topological information
@test point(s, 1) == Point3d(0,0,0)
@test coord_to_vert(s, 1, 1) == 1
@test coord_to_vert(s, 2, 1) == 2
@test coord_to_vert(s, 1, 2) == 3
@test coord_to_vert(s, 2, 2) == 4

# Horizontal
@test coord_to_edge(s, 1, 1, 1) == 1
@test coord_to_edge(s, 1, 1, 2) == 2

# Vertical
@test coord_to_edge(s, 2, 1, 1) == 3
@test coord_to_edge(s, 2, 2, 1) == 4

@test coord_to_quad(s, 1, 1) == 1

@test src(1, 1, 1) == (1,1)
@test tgt(1, 1, 1) == (2,1)

@test src(2, 1, 1) == (1,1)
@test tgt(2, 1, 1) == (1,2)

# Bottom left, bottom right, top right, top left
@test coords_to_verts(s, quad_vertices(1,1)) == [1, 2, 4, 3]

# Bottom, top, left, right
@test coords_to_xedges(s, quad_edges(1,1)[1:2]) == [1, 2]
@test coords_to_yedges(s, quad_edges(1,1)[3:4]) == [3, 4]

@test xedge_len(s, 1, 1) == 10.0
@test yedge_len(s, 1, 1) == 10.0

@test quad_width(s, 1,1) == 10.0
@test quad_height(s, 1,1) == 10.0

@test quad_area(s, 1,1) == 100.0
@test safe_quad_area(s, 1,1) == 100.0
@test safe_quad_area(s, 2,2) == 0.0

@test coord_to_quad(s, xe_t_q(1, 1)...) == 1
@test coord_to_quad(s, xe_b_q(1, 2)...) == 1

@test coord_to_quad(s, ye_l_q(2, 1)...) == 1
@test coord_to_quad(s, ye_r_q(1, 1)...) == 1

@test coord_to_xedge(s, v_l_xe(2,1)...) == 1
@test coord_to_xedge(s, v_r_xe(1,1)...) == 1

@test coord_to_yedge(s, v_b_ye(1,2)...) == 3
@test coord_to_yedge(s, v_t_ye(1,1)...) == 3

@test coord_to_quad(s, v_bl_q(2,2)...) == 1
@test coord_to_quad(s, v_br_q(1,2)...) == 1
@test coord_to_quad(s, v_tl_q(2,1)...) == 1
@test coord_to_quad(s, v_tr_q(1,1)...) == 1

@test dual_point(s, CartesianIndex(1,1)) == Point3d(5,5,0)

for e in edges(s)
  @test d_edge_len(s, e...) == 5.0
end

for dq in vertices(s)
  @test d_quad_area(s, dq...) == 25.0
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

@test coords_to_quads(s, edge_quads(xedge(4, 1)...)) == [4, 0]
@test coords_to_quads(s, edge_quads(xedge(4, 2)...)) == [8, 4]
@test coords_to_quads(s, edge_quads(xedge(4, 5)...)) == [20, 16]

@test coords_to_quads(s, edge_quads(yedge(1, 1)...)) == [0, 1]
@test coords_to_quads(s, edge_quads(yedge(2, 1)...)) == [1, 2]
@test coords_to_quads(s, edge_quads(yedge(5, 1)...)) == [4, 5]

@test edge_len(s, xedge(1, 1)...) == 1.0
@test edge_len(s, xedge(2, 1)...) == 2.0
@test edge_len(s, xedge(3, 1)...) == 3.0
@test edge_len(s, xedge(4, 1)...) == 4.0

@test edge_len(s, yedge(1, 1)...) == 1.0
@test edge_len(s, yedge(1, 2)...) == 2.0
@test edge_len(s, yedge(1, 3)...) == 3.0
@test edge_len(s, yedge(1, 4)...) == 4.0

@test quad_area(s, 1,1) == 1.0
@test quad_area(s, 2,2) == 4.0
@test quad_area(s, 3,3) == 9.0
@test quad_area(s, 4,4) == 16.0

@test coords_to_verts(s, quad_vertices(1,1)) == [1,2,7,6]
@test coords_to_verts(s, quad_vertices(2,1)) == [2,3,8,7]
@test coords_to_verts(s, quad_vertices(1,2)) == [6,7,12,11]
@test coords_to_verts(s, quad_vertices(4,4)) == [19,20,25,24]

@test coords_to_xedges(s, quad_edges(1,1)[1:2]) == [1,5]
@test coords_to_yedges(s, quad_edges(1,1)[3:4]) == [21,22]

@test coords_to_xedges(s, quad_edges(4,4)[1:2]) == [16,20]
@test coords_to_yedges(s, quad_edges(4,4)[3:4]) == [39,40]
