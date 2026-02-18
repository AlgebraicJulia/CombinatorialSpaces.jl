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
yedges(V) .= -edge_len(s, 1, 1, 1)

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

function WENO_left_upwind(k::Int, s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  weno = UniformWENO{k, Float64}()

  r = polyorder(weno)
  1 + r <= i <= nx(s) - r || return 0

  return WENO(weno, RectStencil{k, Float64}(view(f, i-r:i+r, j)...))
end

function WENO_right_upwind(k::Int, s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  weno = UniformWENO{k, Float64}()

  r = polyorder(weno)
  r <= i <= nx(s) - r - 1 || return 0

  return WENO(weno, RectStencil{k, Float64}(view(f, i+r+1:-1:i-r+1, j)...))
end

function WENO_up_upwind(k::Int, s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  weno = UniformWENO{k, Float64}()

  r = polyorder(weno)
  1 + r <= j <= ny(s) - r || return 0

  return WENO(weno, RectStencil{k, Float64}(view(f, i, j-r:j+r)...))
end

function WENO_down_upwind(k::Int, s::EmbeddedCubicalComplex2D, f, i::Int, j::Int)
  weno = UniformWENO{k, Float64}()

  r = polyorder(weno)
  r <= j <= ny(s) - r - 1 || return 0

  return WENO(weno, RectStencil{k, Float64}(view(f, i, j+r+1:-1:j-r+1)...))
end

function interior_product(::Val{1}, k::Int, s::EmbeddedCubicalComplex2D, X, f)
  Xx = xedges(X)
  Xy = yedges(X)

  fx = d_xedges(f)
  fy = d_yedges(f)

  res = init_tensor_d(Val(0), s)

  h = edge_len(s, 1, 1, 1)

  for i in 1:nxquads(s)
    for j in 1:nyquads(s)
      interp_Xx = 0.5 * (Xx[i, j+1] + Xx[i, j])
      if interp_Xx >= 0
        res[i, j] += interp_Xx * WENO_left_upwind(k, s, fx, i, j) / h^2
      else
        res[i, j] += interp_Xx * WENO_right_upwind(k, s, fx, i, j) / h^2
      end

      interp_Xy = 0.5 * (Xy[i+1, j] + Xy[i, j])
      if interp_Xy <= 0
        res[i, j] += interp_Xy * WENO_up_upwind(k, s, fy, i, j) / h^2
      else
        res[i, j] += interp_Xy * WENO_down_upwind(k, s, fy, i, j) / h^2
      end
    end
  end

  return res
end

t_e = 6.0

save_res(arr) = detensorfy_d(Val(0), s, deepcopy(hodge_star!(final_res, Val(2), s, arr)))

n = 9

function WENO5_adv(du, u, p, t)
  boundary_quad_map!(bound_res, s, u, (n-1, n-1))

  hodge_star!(hdg2_res, Val(2), s, bound_res)
  dual_derivative!(dd0_res, Val(0), s, hdg2_res)

  iXv_res = interior_product(Val(1), n, s, V, dd0_res)

  hodge_star!(invhdg2_res, Val(2), s, iXv_res, inv = true)

  du .= -invhdg2_res
end

prob = ODEProblem(WENO5_adv, v_0, (0.0, t_e))
soln = solve(prob, Tsit5())

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
mesh!(ax, sd; color = save_res(soln.u[end]))
Colorbar(fig[1,2])
save("imgs/WENO$(n)DualAdvEnd.png", fig)

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

create_gif(soln, sd, "imgs/WENO$(n)_DualAdv.mp4", 100)
