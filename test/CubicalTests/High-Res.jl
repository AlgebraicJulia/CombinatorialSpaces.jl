using CairoMakie
using Distributions
using SparseArrays
using StaticArrays
using OrdinaryDiffEqTsit5

include("../../src/CubicalCode/WENO.jl")

nx = 251
lx = 10.0

h = lx / (nx - 1)

points = range(0, lx; length = nx)
dual_points = range(h / 2, lx - h / 2; length = nx - 1)

c = h

# ̇v = c∂v/∂x

# Periodic conditions
function dual_derivative()
  I, J = zeros(Int64, 2 * (nx - 1)), zeros(Int64, 2 * (nx - 1))
  V = zeros(2 * (nx - 1))

  # Left boundary
  I[1] = 1
  I[2] = 1

  J[1] = nx - 1
  J[2] = 1

  V[1] = -1
  V[2] = 1

  for i in 2:nx-1
    idx = 2 * i - 1
    I[idx] = i
    I[idx + 1] = i

    J[idx] = i - 1
    J[idx + 1] = i

    V[idx] = -1
    V[idx + 1] = 1
  end

  return sparse(I, J, V)
end

d0 = dual_derivative()

function WENO_left_upwind(k::Int, f, i::Int)
  weno = UniformWENO{k, Float64}()

  r = polyorder(weno)
  1 + r <= i <= nx - r - 1 || return 0

  return WENO(weno, RectStencil{k, Float64}(view(f, i-r:i+r)...))
end

function interior_product(::Val{1}, k::Int, f)
  res = zeros(nx - 1)

  for i in 1:nx-1
    res[i] = WENO_left_upwind(k, f, i) / h^2
  end

  return res
end

# # This doesn't properly deal with boundary conditions
# # Between a one form and a dual one form
# function upwinding_interior(X, dω)
#   res = zeros(nx - 1)
#   for i in 1:nx-2
#     res[i] = X[i] * dω[i+1] / h^2
#   end
#   return res
# end

dist = Normal(lx / 2, 0.5)
u_0 = [pdf(dist, x) for x in dual_points];

t_e = 0.2

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
lines!(ax, dual_points, u_0)
save("imgs/BurgerInit.png", fig)

weno_order = 9

function burger(du, u, p, t)
  du .= -interior_product(Val(1), weno_order, d0 * (u .^2 / 2))
end

prob = ODEProblem(burger, u_0, (0.0, t_e))
soln = solve(prob, Tsit5())
soln.retcode

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
lines!(ax, dual_points, soln.u[50])
save("imgs/BurgerFinal.png", fig)

function create_gif(solution, file_name, length = 100)
  time = Observable(0.0)

  y = @lift(soln($time))

  fig = lines(dual_points, y, color = :blue, linewidth = 4,
      axis = (title = @lift("t = $(round($time, digits = 3))"),))

  timestamps = range(0, t_e, length = length)

  record(fig, file_name, timestamps) do t
      time[] = t
  end
end

create_gif(soln, "imgs/WENO$(weno_order)_Burger.mp4")
