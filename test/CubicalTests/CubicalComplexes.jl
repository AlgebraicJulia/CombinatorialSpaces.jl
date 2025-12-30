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

@test hedge_len(s, CartesianIndex(1,1)) == 10.0
@test vedge_len(s, CartesianIndex(1,1)) == 10.0

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

# # Exterior derivative
# d0 = exterior_derivative(Val(0), s)
# d1 = exterior_derivative(Val(1), s)

# @test all(0 .== d1 * d0)

# dual_d0 = dual_derivative(Val(0), s)
# dual_d1 = dual_derivative(Val(1), s)

# @test all(0 .== dual_d1 * dual_d0)

# # Wedge product
# v1 = ones(nv(s))
# e1 = ones(ne(s))
# q1 = ones(nquads(s))

# all(1.0 .== wedge_product(Val((0,1)), s, v1, e1))
# all(1.0 .== wedge_product(Val((0,2)), s, v1, q1))

# all(0.0 .== wedge_product(Val((1,1)), s, e1, e1))
# all(0.0 .== wedge_product(Val((1,1)), s, e1, 10 * e1))

# eX = zeros(ne(s))
# eX[1:nhe(s)] .= 1

# eY = zeros(ne(s))
# eY[nhe(s)+1:ne(s)] .= 1
# all(1.0 .== wedge_product(Val((1,1)), s, eX, eY))
# all(-1.0 .== wedge_product(Val((1,1)), s, eY, eX))

# # Hodge star
# hdg_0 = hodge_star(Val(0), s)
# @test all(25 .== diag(hdg_0))
# @test (100 == sum(diag(hdg_0)))

# hdg_1 = hodge_star(Val(1), s)
# @test all(0.5 .== diag(hdg_1))

# hdg_2 = hodge_star(Val(2), s)
# @test all(1/100 .== diag(hdg_2))

# # Sharp PD
# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = 10

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 1.0 && Y[1] == 0.0

# alpha = zeros(ne(s))
# alpha[3] = alpha[4] = -20

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 0.0 && Y[1] == -2.0

# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = 10
# alpha[3] = alpha[4] = -20

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 1.0 && Y[1] == -2.0

# # Sharp DD
# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = -10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 0.0 && Y[1] == 2.0

# alpha = zeros(ne(s))
# alpha[3] = alpha[4] = 10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 2.0 && Y[1] == 0.0

# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = -10
# alpha[3] = alpha[4] = 10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 2.0 && Y[1] == 2.0

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

@test edge_quads(s, 4) == [0, 4]
@test edge_quads(s, 8) == [4, 8]
@test edge_quads(s, 20) == [16, 0]

@test edge_quads(s, 21) == [0, 1]
@test edge_quads(s, 22) == [1, 2]
@test edge_quads(s, 25) == [4, 0]

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

# Exterior derivative
d0 = exterior_derivative(Val(0), s)
d1 = exterior_derivative(Val(1), s)

@test all(0 .== d1 * d0)

dual_d0 = dual_derivative(Val(0), s)
dual_d1 = dual_derivative(Val(1), s)

@test all(0 .== dual_d1 * dual_d0)

# Wedge product
v1 = ones(nv(s))
e1 = ones(ne(s))
q1 = ones(nquads(s))

all(1.0 .== wedge_product(Val((0,1)), s, v1, e1))
all(1.0 .== wedge_product(Val((0,2)), s, v1, q1))

all(0.0 .== wedge_product(Val((1,1)), s, e1, e1))
all(0.0 .== wedge_product(Val((1,1)), s, e1, 10 * e1))

eX = zeros(ne(s))
eX[1:nhe(s)] .= 1

eY = zeros(ne(s))
eY[nhe(s)+1:ne(s)] .= 1
all(1.0 .== wedge_product(Val((1,1)), s, eX, eY))
all(-1.0 .== wedge_product(Val((1,1)), s, eY, eX))

# Hodge star
hdg_0 = hodge_star(Val(0), s)
@test (100 == sum(diag(hdg_0)))

hdg_1 = hodge_star(Val(1), s)
for i in 1:4 # Bottom horizontal edges
  @test 1/(2*i) == diag(hdg_1)[i]
end

for i in 2:4 # Bottom interior vertical edges
  @test 0.5*(2i-1) == diag(hdg_1)[i+nhe(s)]
end

hdg_2 = hodge_star(Val(2), s)
@test 1 == diag(hdg_2)[1]
@test 1/4 == diag(hdg_2)[6]
@test 1/9 == diag(hdg_2)[11]
@test 1/16 == diag(hdg_2)[16]