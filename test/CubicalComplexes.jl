using Test
using CairoMakie

include("../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 2, 2)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/SquareGrid.png", fig)

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

for dq in vertices(s)
  @test 25.0 == dual_quad_area(s, dq)
end

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
@test all(25 .== diag(hdg_0))
@test (100 == sum(diag(hdg_0)))

hdg_1 = hodge_star(Val(1), s)
@test all(0.5 .== diag(hdg_1))

hdg_2 = hodge_star(Val(2), s)
@test all(1/100 .== diag(hdg_2))

# Sharp PD
alpha = zeros(ne(s))
alpha[1] = alpha[2] = 10

X, Y = sharp_pd(s, alpha)

@test X[1] == 1.0 && Y[1] == 0.0

alpha = zeros(ne(s))
alpha[3] = alpha[4] = -20

X, Y = sharp_pd(s, alpha)

@test X[1] == 0.0 && Y[1] == -2.0

alpha = zeros(ne(s))
alpha[1] = alpha[2] = 10
alpha[3] = alpha[4] = -20

X, Y = sharp_pd(s, alpha)

@test X[1] == 1.0 && Y[1] == -2.0

# Sharp DD
alpha = zeros(ne(s))
alpha[1] = alpha[2] = -10

X, Y = sharp_dd(s, alpha)
@test X[1] == 0.0 && Y[1] == 2.0

alpha = zeros(ne(s))
alpha[3] = alpha[4] = 10

X, Y = sharp_dd(s, alpha)
@test X[1] == 2.0 && Y[1] == 0.0

alpha = zeros(ne(s))
alpha[1] = alpha[2] = -10
alpha[3] = alpha[4] = 10

X, Y = sharp_dd(s, alpha)
@test X[1] == 2.0 && Y[1] == 2.0

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

### HEAT EQUATION ###

function create_gif(solution, file_name)
  frames = length(solution)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1,1])
  msh = CairoMakie.mesh!(ax, s, color=first(solution), colormap=:jet, colorrange=extrema(first(solution)))
  Colorbar(fig[1,2], msh)
  CairoMakie.record(fig, file_name, 1:10:frames; framerate = 15) do t
    msh.color = solution[t]
  end
end

s = uniform_grid(10, 10, 101, 101);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/HeatGrid.png", fig)

u_0 = map(points(s)) do p
  if (3 <= p[1] <= 7) && (3 <= p[2] <= 7)
    return 1.0
  else
    return 0.0
  end
end

# u_0 = map(p -> p[1], points(s))

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = u_0)
Colorbar(fig[1,2])
save("imgs/InitialSquare.png", fig)

Δ0 = laplacian(Val(0), s) 

Δt = 0.001
u = deepcopy(u_0)

diff_us = []
push!(diff_us, u_0)

for _ in 0:Δt:0.5
  u .= u .- Δt * Δ0 * u
  push!(diff_us, deepcopy(u))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(diff_us))
Colorbar(fig[1,2])
save("imgs/DiffusionEnd.png", fig)

create_gif(diff_us, "imgs/SquareDiffusion.mp4")

### CONSTANT ADVECTION ###

V = zeros(ne(s))
V[1:nhe(s)] .= 1

u = deepcopy(u_0)
adv_us = []
push!(adv_us, u_0)

δ1 = codifferential(Val(1), s)

depth = 1
topb = top_boundary_quads(s, depth);
botb = bottom_boundary_quads(s, depth);

leftb = left_boundary_quads(s, depth);
rightb = right_boundary_quads(s, depth);

tb_bounds = vcat(topb, botb);
lr_bounds = vcat(leftb, rightb);

vtb_bounds = VertexMapping(s, tb_bounds);
vlr_bounds = VertexMapping(s, lr_bounds);

Δt = 0.001
for _ in 0:Δt:0.25
  apply_periodic!(u, vlr_bounds)
  u .= u .+ Δt * δ1 * (wedge_product(Val((0,1)), s, u, V))
  push!(adv_us, deepcopy(u))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(adv_us))
Colorbar(fig[1,2])
save("imgs/AdvectionEnd.png", fig)

create_gif(adv_us, "imgs/SquareAdvection.mp4")