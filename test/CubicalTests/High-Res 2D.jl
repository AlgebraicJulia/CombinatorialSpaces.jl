using Test
using CairoMakie
using Distributions

include("../../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 101, 101);
fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/AdvGrid.png", fig)

# dist = MvNormal([5, 5], [1, 1])
# u_0 = map(q -> begin p = dual_point(s, q...); pdf(dist, [p[1], p[2]]); end, quadrilaterals(s))

u_0 = map(q -> begin p = dual_point(s, q...); (4 <= p[1] <= 6 && 4 <= p[2] <= 6) ? 1 : 0; end, quadrilaterals(s))

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
hdg2_res = init_tensor_d(Val(0), s);
dd0_res = init_tensor_d(Val(1), s);
invhdg1_res = init_tensor(Val(1), s);

invhdg2_res = init_tensor(Val(2), s);

final_res = init_tensor_d(Val(0), s);

v_0 = init_tensor(Val(2), s)
hodge_star!(v_0, Val(2), s, tensorfy_d(Val(0), s, u_0); inv = true)
v = deepcopy(v_0)

# TODO: This is only for uniform, rightward motion without boundary conditions
function interior_product_quick(::Val{1}, s::EmbeddedCubicalComplex2D, X, f)
  Xx = xedges(X)
  Xy = yedges(X)

  fx = d_xedges(f)
  fy = d_yedges(f)

  res = init_tensor_d(Val(0), s)

  h = edge_len(s, 1, 1, 1)

  for i in 1:nxquads(s) - 1
    for j in 1:nyquads(s)
      res[i, j] = 0.5 * (Xx[i, j+1] + Xx[i, j]) * (3/8 * fx[i, j] + 6/8 * fx[i+1,j] - 1/8 * fx[i+2,j]) / h^2
    end
  end

  return res
end

function interior_product_upwind(::Val{1}, s::EmbeddedCubicalComplex2D, X, f)
  Xx = xedges(X)
  Xy = yedges(X)

  fx = d_xedges(f)
  fy = d_yedges(f)

  res = init_tensor_d(Val(0), s)

  h = edge_len(s, 1, 1, 1)

  for i in 1:nxquads(s) - 1
    for j in 1:nyquads(s)
      res[i, j] = 0.5 * (Xx[i, j+1] + Xx[i, j]) * fx[i+1, j] / h^2
    end
  end

  return res
end

Δt = 0.001
t_e = 0.05

adv_us = []
save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

push!(adv_us, save_res(v_0))

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = first(adv_us))
save("imgs/UpwindAdvInit.png", fig)

for _ in 0:Δt:t_e
  boundary_quad_map!(bound_res, s, v, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_upwind(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  v .= v .+ Δt .* invhdg2_res
  push!(adv_us, save_res(boundary_quad_map!(bound_res, s, v, (1, 0))))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = last(adv_us))
save("imgs/UpwindAdvEnd.png", fig)

adv_us = []
save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

push!(adv_us, save_res(v_0))
v = deepcopy(v_0)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = first(adv_us))
save("imgs/QUICKAdvInit.png", fig)

Δt = 0.001
for _ in 0:Δt:t_e
  boundary_quad_map!(bound_res, s, v, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_quick(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  v .= v .+ Δt .* invhdg2_res
  push!(adv_us, save_res(boundary_quad_map!(bound_res, s, v, (1, 0))))
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
scatter!(ax, xpos, ypos; color = last(adv_us))
save("imgs/QUICKDualAdvEnd.png", fig)
