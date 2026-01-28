using Plots
using Distributions
using OrdinaryDiffEq

nx = 301
lx = 2
hx = lx / (nx - 1)

xs = range(0, lx; length = nx)

dist = Normal(0.3, 0.01)
f_0 = map(x -> pdf(dist, x), xs)

plot(xs, f_0)

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

f = [f_0[end-1], f_0[end], f_0..., f_0[1], f_0[2]]

t_e = 4.0

CFL = 0.1
Δt = CFL * hx

function WENO5(f, p, t)
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

  f[1] = f[end-3]
  f[2] = f[end-2]

  f[end-1] = f[3]
  f[end] = f[4]

  return -[0, 0, (f_hat[1] - f_hat[end]) / hx, diff(f_hat) / hx..., 0, 0]
end

function basic_upwind(f, p, t)

  f_hat = map(i -> 1/2 * (f[i] + f[i - 1]) , 3:nx+3)

  f[1] = f[end-3]
  f[2] = f[end-2]

  f[end-1] = f[3]
  f[end] = f[4]

  return -[0, 0, diff(f_hat) / hx..., 0, 0]
end

prob = ODEProblem(WENO5, f, (0.0, t_e))
soln = solve(prob, Tsit5())

for t in range(0, t_e, length = 100)
  display(plot(xs, soln(t)[3:end-2]))
end
