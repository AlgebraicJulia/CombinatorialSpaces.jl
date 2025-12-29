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
V[1:nhe(s)] .= 1

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
V[1:nhe(s)] .= 0

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
V = init_tensor_form(Val(1), s)
get_hedge_form(V) .= 1

wdg01_res = init_tensor_form(Val(1), s)
hdg1_res = init_tensor_form(Val(1), s)
dd1_res = init_tensor_form(Val(0), s) # Init tensor dual form
final_res = init_tensor_form(Val(0), s) # Init tensor dual form

v = tensorfy_form(Val(0), s, deepcopy(u_0))
adv_us = []
push!(adv_us, detensorfy_form(Val(0), s, deepcopy(v)))

Δt = 0.001
for _ in 0:Δt:1.0
  v = lr_boundary_coord_verts_map!(v, s, 1)
  wedge_product!(wdg01_res, Val((0,1)), s, v, V)
  hodge_star!(hdg1_res, Val(1), s, wdg01_res)
  dual_derivative!(dd1_res, Val(1), s, hdg1_res)
  hodge_star!(final_res, Val(0), s, dd1_res; inv = true)

  v .= v .+ Δt * final_res
  push!(adv_us, detensorfy_form(Val(0), s, deepcopy(v)))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(adv_us))
Colorbar(fig[1,2])
save("imgs/KernelAdvectionEnd.png", fig)

create_gif(adv_us, "imgs/KernelAdvection.mp4")

flat_V = detensorfy_form(Val(1), s, V)
test = wedge_product(Val((0,1)), s, u_0, flat_V)

wdg01_res = init_tensor_form(Val(1), s)
wedge_product!(wdg01_res, Val((0,1)), s, v_0, V)
comp = detensorfy_form(Val(1), s, wdg01_res)

extrema(test .- comp)