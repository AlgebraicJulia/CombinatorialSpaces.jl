using CairoMakie
using Distributions
using SparseArrays

nx = 101
lx = 1.0

h = lx / (nx - 1)

points = range(0, lx; length = nx)
dual_points = range(h / 2, lx - h / 2; length = nx - 1)

c = 0.1

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

# This doesn't properly deal with boundary conditions
# Between a one form and a dual one form
function central_interior(X, dω)
  res = zeros(nx - 1)
  res[end] = X[end] * (dω[1] + dω[end - 1]) / (2 * h^2)
  for i in 1:nx-2
    res[i] = X[i] * (dω[i+1] + dω[i+1]) / (2 * h^2)
  end
  return res
end

X = [c for i in 1:nx-1];

dist = Normal(lx / 2, 0.1)
ω_0 = [pdf(dist, x) for x in dual_points];

Δt = 0.001
t_e = 100 * Δt

ω = deepcopy(ω_0)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
plot!(ax, dual_points, ω_0)
fig

for _ in 0:Δt:t_e
  ω .= ω .+ Δt * central_interior(X, d0 * ω)
end

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
plot!(ax, dual_points, ω)
fig
