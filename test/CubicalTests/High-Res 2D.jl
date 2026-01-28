using Test
using CairoMakie
using Distributions
using OrdinaryDiffEq

include("../../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 101, 101);
sd = dual_mesh(s); # Mainly for plotting

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
mesh!(ax, sd; color = u_0)
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

IS0(f, j::Int) = 13/12*(f[j-2]-2f[j-1]+f[j])^2 + 1/4*(f[j-2]-4f[j-1]+3f[j])^2
IS1(f, j::Int) = 13/12*(f[j-1]-2f[j]+f[j+1])^2 + 1/4*(f[j-1]-f[j+1])^2
IS2(f, j::Int) = 13/12*(f[j]-2f[j+1]+f[j+2])^2 + 1/4*(3f[j]-4f[j+1]+f[j+2])^2

function interior_product_WENO5(::Val{1}, s::EmbeddedCubicalComplex2D, X, f)
  f_hat = zeros(nx)

  IS_0 = (map(j -> IS0(f, j), 3:nx+2) .+ ϵ).^2
  IS_1 = (map(j -> IS1(f, j), 3:nx+2) .+ ϵ).^2
  IS_2 = (map(j -> IS2(f, j), 3:nx+2) .+ ϵ).^2

  a0 = C3_0 ./ IS_0; a1 = C3_1 ./ IS_1; a2 = C3_2 ./ IS_2

  a = a0 .+ a1 .+ a2

  w0 = a0 ./ a; w1 = a1 ./ a; w2 = a2 ./ a;

  f_hat .+= w0 .* map(i -> a3_01 * f[i - 2] + a3_02 * f[i - 1] + a3_03 * f[i], 3:nx+2)
  f_hat .+= w1 .* map(i -> a3_11 * f[i - 1] + a3_12 * f[i] + a3_13 * f[i + 1], 3:nx+2)
  f_hat .+= w2 .* map(i -> a3_21 * f[i] + a3_22 * f[i + 1] + a3_23 * f[i + 2], 3:nx+2)


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

t_e = 0.05

save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

function upwind_adv(u, p, t)
  boundary_quad_map!(bound_res, s, v, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_upwind(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  return invhdg2_res
end

function QUICK_adv(u, p, t)
  boundary_quad_map!(bound_res, s, v, (1, 0))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product_quick(Val(1), s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  return invhdg2_res
end

prob = ODEProblem(upwind_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[end]))
save("imgs/UpwindAdvEnd.png", fig)

prob = ODEProblem(QUICK_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[end]))
save("imgs/QUICKDualAdvEnd.png", fig)
