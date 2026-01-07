using Test
using CairoMakie
using Distributions

include("../../src/CubicalComplexes.jl")

### LEFT-RIGHT PERIODIC ADVECTION ###

s = uniform_grid(10, 10, 101, 101);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/AdvGrid.png", fig)

dist = MvNormal([5, 5], [1, 1])
u_0 = [pdf(dist, [p[1], p[2]]) for p in points(s)]

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = u_0)
Colorbar(fig[1,2])
save("imgs/InitialAdv.png", fig)

# Rightwards motion
V = zeros(ne(s))
V[1:nxe(s)] .= 1

δ1 = codifferential(Val(1), s)

depth = 1

u = deepcopy(u_0)
adv_us = []
push!(adv_us, u_0)

Δt = 0.001
for _ in 0:Δt:1
  lr_boundary_verts_map!(u, s, depth)
  u .= u .+ Δt * δ1 * (wedge_product(Val((0,1)), s, u, V))
  push!(adv_us, deepcopy(u))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(adv_us))
Colorbar(fig[1,2])
save("imgs/LeftRightAdvectionEnd.png", fig)

create_gif(adv_us, "imgs/LeftRightAdvection.mp4")

### TOP-BOTTOM PERIODIC ADVECTION ###

s = uniform_grid(10, 10, 101, 101);

dist = MvNormal([5, 5], [1, 1])
u_0 = [pdf(dist, [p[1], p[2]]) for p in points(s)]

# Upwards motion
V = 1 * ones(ne(s))
V[1:nxe(s)] .= 0

δ1 = codifferential(Val(1), s)

depth = 1

u = deepcopy(u_0)
adv_us = []
push!(adv_us, u_0)

Δt = 0.001
for _ in 0:Δt:1
  tb_boundary_verts_map!(u, s, depth)
  u .= u .+ Δt * δ1 * (wedge_product(Val((0,1)), s, u, V))
  push!(adv_us, deepcopy(u))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(adv_us))
Colorbar(fig[1,2])
save("imgs/TopBottomAdvectionEnd.png", fig)

create_gif(adv_us, "imgs/TopBottomAdvection.mp4")

### KERNEL ADVECTION ###

s = uniform_grid(10, 10, 101, 101);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/AdvGrid.png", fig)

dist = MvNormal([5, 5], [1, 1])
u_0 = [pdf(dist, [p[1], p[2]]) for p in points(s)]

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = u_0)
Colorbar(fig[1,2])
save("imgs/InitialAdv.png", fig)

# Rightwards motion
V = init_tensor(Val(1), s)
xedges(V) .= 1

bound_res = init_tensor(Val(0), s);
wdg01_res = init_tensor(Val(1), s);
hdg1_res = init_tensor(Val(1), s);
dd1_res = init_tensor(Val(0), s); # Init tensor dual form
final_res = init_tensor(Val(0), s); # Init tensor dual form

v = tensorfy(Val(0), s, deepcopy(u_0))
adv_us = []
push!(adv_us, detensorfy(Val(0), s, deepcopy(v)))

Δt = 0.001
for _ in 0:Δt:1.0
  boundary_v_map!(bound_res, s, v, (1, 0))
  wedge_product!(wdg01_res, Val((0,1)), s, bound_res, V)
  hodge_star!(hdg1_res, Val(1), s, wdg01_res)
  dual_derivative!(dd1_res, Val(1), s, hdg1_res)
  hodge_star!(final_res, Val(0), s, dd1_res; inv = true)

  v .= v .+ Δt .* final_res
  push!(adv_us, detensorfy(Val(0), s, deepcopy(v)))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(adv_us))
Colorbar(fig[1,2])
save("imgs/KernelAdvectionEnd.png", fig)

create_gif(adv_us, "imgs/KernelAdvection.mp4")

init = detensorfy(Val(0), s, first(adv_us))
fin = detensorfy(Val(0), s, last(adv_us))

extrema(init .- fin)

fig = Figure()
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = first(adv_us))
Colorbar(fig[1,2])
save("imgs/KernelAdvectionStart.png", fig)

### KERNEL LIE ADVECTION ###

s = uniform_grid(10, 10, 101, 101);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/AdvGrid.png", fig)

dist = MvNormal([5, 5], [1, 1])
u_0 = map(q -> begin p = dual_point(s, q...); pdf(dist, [p[1], p[2]]); end, quadrilaterals(s))

xpos = map(q -> dual_point(s, q...)[1], quadrilaterals(s))
ypos = map(q -> dual_point(s, q...)[2], quadrilaterals(s))

# TODO: Better way to plot dual values
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = u_0)
save("imgs/DualAdv.png", fig)

# Rightwards motion
V = init_tensor(Val(1), s)
xedges(V) .= 1

bound_res = init_tensor(Val(2), s);

# L = ⋆(V ∧ ⋆du )
hdg2_res = init_tensor_d(Val(0), s)
dd0_res = init_tensor_d(Val(1), s);
invhdg1_res = init_tensor(Val(1), s);

wdg11_res = init_tensor(Val(2), s);

final_res = init_tensor_d(Val(0), s);

v = init_tensor(Val(2), s)
hodge_star!(v, Val(2), s, tensorfy_d(Val(0), s, u_0); inv = true)

adv_us = []
save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

push!(adv_us, save_res(v))

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = first(adv_us))
save("imgs/DualAdvInit.png", fig)

Δt = 0.001
for _ in 0:Δt:1.0
  boundary_quad_map!(bound_res, s, v, (1, 0))
  
  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)
  hodge_star!(invhdg1_res, Val(1), s, dd0_res; inv = true)

  wedge_product!(wdg11_res, Val((1,1)), s, invhdg1_res, V)

  v .= v .+ Δt .* wdg11_res
  push!(adv_us, save_res(boundary_quad_map!(bound_res, s, v, (1, 0))))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = last(adv_us))
save("imgs/DualAdvEnd.png", fig)

# create_gif(adv_us, "imgs/KernelAdvection.mp4")
