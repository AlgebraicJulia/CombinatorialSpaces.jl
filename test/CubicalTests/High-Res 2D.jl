using Test
using CairoMakie
using Distributions
using OrdinaryDiffEq

include("../../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 201, 201);
sd = dual_mesh(s); # Mainly for plotting

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
save("imgs/AdvGrid.png", fig)

u_0 = map(q -> begin p = dual_point(s, q...); (4 <= p[1] <= 6 && 4 <= p[2] <= 6) ? 1 : 0; end, quadrilaterals(s))

# dist = MvNormal([5, 5], [1, 1])
# u_0 = [pdf(dist, [p[1], p[2]]) for p in dual_points(s)]

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = u_0)
save("imgs/DualAdv.png", fig)

# Rightwards motion
V = init_tensor(Val(1), s)
xedges(V) .= edge_len(s, 1, 1, 1)

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

  for i in 2:nxquads(s)
    for j in 1:nyquads(s)
      res[i, j] = 0.5 * (Xx[i, j+1] + Xx[i, j]) * (3/8 * fx[i + 1, j] + 6/8 * fx[i,j] - 1/8 * fx[i-1,j]) / h^2
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

  for i in 1:nxquads(s)
    for j in 1:nyquads(s)
      res[i, j] = 0.5 * (Xx[i, j+1] + Xx[i, j]) * fx[i, j] / h^2
    end
  end

  return res
end

# TODO: Implement improved WENO-Z
function WENO5_left_upwind(s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  # Optimal coefficients for combined stencil, 5th order
  # Cr_k, r is order, k is stencil shift
  C3_0 = 1/10
  C3_1 = 6/10
  C3_2 = 3/10

  ϵ = 1e-40

  # Coefficients for individual stencils, 3rd order
  # ar_kl, r is order, j is stencil shift, l is term
  a3_01 = 1/3
  a3_02 = -7/6
  a3_03 = 11/6

  a3_11 = -1/6
  a3_12 = 5/6
  a3_13 = 1/3

  a3_21 = 1/3
  a3_22 = 5/6
  a3_23 = -1/6

  IS0(f, i::Int, j::Int) = 13/12*(f[i-2, j]-2f[i-1, j]+f[i, j])^2 + 1/4*(f[i-2, j]-4f[i-1, j]+3f[i, j])^2
  IS1(f, i::Int, j::Int) = 13/12*(f[i-1, j]-2f[i, j]+f[i+1, j])^2 + 1/4*(f[i-1, j]-f[i+1, j])^2
  IS2(f, i::Int, j::Int) = 13/12*(f[i, j]-2f[i+1, j]+f[i+2, j])^2 + 1/4*(3f[i, j]-4f[i+1, j]+f[i+2, j])^2


  IS_0 = IS0(f, i, j); IS_1 = IS1(f, i, j); IS_2 = IS2(f, i, j);

  tau_5 = abs(IS_0 - IS_2)

  a0 = C3_0 * (1 + tau_5 / (IS_0 + ϵ)); a1 = C3_1 * (1 + tau_5 / (IS_1 + ϵ)); a2 = C3_2 * (1 + tau_5 / (IS_2 + ϵ));

  a = a0 + a1 + a2

  w0 = a0 / a; w1 = a1 / a; w2 = a2 / a;

  flux_0 = w0 * (a3_01 * f[i - 2, j] + a3_02 * f[i - 1, j] + a3_03 * f[i, j])
  flux_1 = w1 * (a3_11 * f[i - 1, j] + a3_12 * f[i, j] + a3_13 * f[i + 1, j])
  flux_2 = w2 * (a3_21 * f[i, j] + a3_22 * f[i + 1, j] + a3_23 * f[i + 2, j])

  return flux_0 + flux_1 + flux_2
end

function WENO5_right_upwind(s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  # Optimal coefficients for combined stencil, 5th order
  # Cr_k, r is order, k is stencil shift
  C3_0 = 1/10
  C3_1 = 6/10
  C3_2 = 3/10

  ϵ = 1e-6

  # Coefficients for individual stencils, 3rd order
  # ar_kl, r is order, j is stencil shift, l is term
  a3_01 = 1/3
  a3_02 = -7/6
  a3_03 = 11/6

  a3_11 = -1/6
  a3_12 = 5/6
  a3_13 = 1/3

  a3_21 = 1/3
  a3_22 = 5/6
  a3_23 = -1/6

  IS0(f, i::Int, j::Int) = 13/12*(f[i+3, j]-2f[i+2, j]+f[i+1, j])^2 + 1/4*(f[i+3, j]-4f[i+2, j]+3f[i+1, j])^2
  IS1(f, i::Int, j::Int) = 13/12*(f[i+2, j]-2f[i+1, j]+f[i, j])^2 + 1/4*(f[i+2, j]-f[i, j])^2
  IS2(f, i::Int, j::Int) = 13/12*(f[i+1, j]-2f[i, j]+f[i-1, j])^2 + 1/4*(3f[i+1, j]-4f[i, j]+f[i-1, j])^2

  IS_0 = IS0(f, i, j); IS_1 = IS1(f, i, j); IS_2 = IS2(f, i, j);

  tau_5 = abs(IS_0 - IS_2)

  a0 = C3_0 * (1 + tau_5 / (IS_0 + ϵ)); a1 = C3_1 * (1 + tau_5 / (IS_1 + ϵ)); a2 = C3_2 * (1 + tau_5 / (IS_2 + ϵ));

  a = a0 + a1 + a2

  w0 = a0 / a; w1 = a1 / a; w2 = a2 / a;

  flux_0 = w0 * (a3_01 * f[i + 3, j] + a3_02 * f[i + 2, j] + a3_03 * f[i + 1, j])
  flux_1 = w1 * (a3_11 * f[i + 2, j] + a3_12 * f[i + 1, j] + a3_13 * f[i, j])
  flux_2 = w2 * (a3_21 * f[i + 1, j] + a3_22 * f[i, j] + a3_23 * f[i - 1, j])

  return flux_0 + flux_1 + flux_2
end

function interior_product_WENO5(::Val{1}, s::EmbeddedCubicalComplex2D, X, f)
  Xx = xedges(X)
  Xy = yedges(X)

  fx = d_xedges(f)
  fy = d_yedges(f)

  res = init_tensor_d(Val(0), s)

  h = edge_len(s, 1, 1, 1)

  for i in 3:nxquads(s) - 2
    for j in 1:nyquads(s)
      interp_Xx = 0.5 * (Xx[i, j+1] + Xx[i, j])
      if interp_Xx >= 0
        res[i, j] = interp_Xx * WENO5_left_upwind(s, fx, i, j) / h^2
      else
        res[i, j] = interp_Xx * WENO5_right_upwind(s, fx, i, j) / h^2
      end
    end
  end

  return res
end

t_e = 3.0

save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

function upwind_adv(du, u, p, t)
  boundary_quad_map!(bound_res, s, u, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_upwind(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  du .= -invhdg2_res
end

function QUICK_adv(du, u, p, t)
  boundary_quad_map!(bound_res, s, u, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_quick(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  du .= -invhdg2_res
end

function WENO5_adv(du, u, p, t)

  c = 1
  if t >= 1.5
    c = -1
  end

  boundary_quad_map!(bound_res, s, u, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_WENO5(Val(1), s, c .* V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  du .= -invhdg2_res
end

prob = ODEProblem(upwind_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[2]))
Colorbar(fig[1,2])
save("imgs/UpwindAdvEnd.png", fig)

prob = ODEProblem(QUICK_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[end]))
Colorbar(fig[1,2])
save("imgs/QUICKDualAdvEnd.png", fig)

prob = ODEProblem(WENO5_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[end]))
Colorbar(fig[1,2])
save("imgs/WENO5DualAdvEnd.png", fig)

function create_gif(solution, s::HasCubicalComplex, file_name, length)
  frames = range(0, t_e, length = length)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1,1])
  init = detensorfy(Val(0), s, solution(0.0))

  msh = CairoMakie.mesh!(ax, s, color=init, colormap=:jet, colorrange=extrema(init))
  Colorbar(fig[1,2], msh)
  CairoMakie.record(fig, file_name, frames; framerate = 15) do t
    msh.color = detensorfy(Val(0), s, solution(t))
  end
end

create_gif(soln, sd, "imgs/WENO5_DualAdv.mp4", 100)