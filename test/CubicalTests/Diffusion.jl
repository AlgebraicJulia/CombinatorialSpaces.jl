using Test
using CairoMakie

include("../../src/CubicalComplexes.jl")

### HEAT EQUATION ###

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

### KERNEL HEAT EQUATION ###

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

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = u_0)
Colorbar(fig[1,2])
save("imgs/InitialSquare.png", fig)

v_0 = tensorfy_form(s, u_0)
d0_res = init_tensor_form(Val(1), s)
hdg1_res = init_tensor_form(Val(1), s)
dd1_res = init_tensor_form(Val(0), s) # Init tensor dual form
final_res = init_tensor_form(Val(0), s) # Init tensor dual form

# final = detensorfy_form(Val(0), s, final_res)

Δt = 0.001
v = deepcopy(v_0)

diff_us = []
push!(diff_us, detensorfy_form(Val(0), s, v_0))

for _ in 0:Δt:0.5
  exterior_derivative!(d0_res, Val(0), v)
  hodge_star!(hdg1_res, Val(1), s, d0_res)
  dual_derivative!(dd1_res, Val(1), s, hdg1_res)
  hodge_star!(final_res, Val(0), s, dd1_res; inv = true)

  v .= v .- Δt * final_res
  push!(diff_us, detensorfy_form(Val(0), s, deepcopy(v)))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, s, color = last(diff_us))
Colorbar(fig[1,2])
save("imgs/KernelDiffusionEnd.png", fig)

create_gif(diff_us, "imgs/KernelSquareDiffusion.mp4")