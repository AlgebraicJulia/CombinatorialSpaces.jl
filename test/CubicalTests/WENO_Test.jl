using Plots
using Distributions

nx = 201
lx = 2
hx = lx / (nx - 1)

xs = range(0, lx; length = nx)

dist = Normal(0.3, 0.05)
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

IS0(f, j::Int) = 13/12 *(f[j-2]-2f[j-1]+f[j])^2 + 1/4*(f[j-2]-4f[j-1]+3f[j])^2
IS1(f, j::Int) = 13/12 *(f[j-1]-2f[j]+f[j+1])^2 + 1/4*(f[j-1]-f[j+1])^2
IS2(f, j::Int) = 13/12 *(f[j]-2f[j+1]+f[j+2])^2 + 1/4*(3f[j]-4f[j+1]+f[j+2])^2

f = [f_0[end-1], f_0[end], f_0..., f_0[1], f_0[2]]

t_e = 1

CFL = 0.1
Δt = CFL * hx

for (i,_) in enumerate(0:Δt:t_e)
  f_hat = zeros(nx)
  f_hat += C3_0 * map(i -> a3_01 * f[i - 2] + a3_02 * f[i - 1] + a3_03 * f[i], 3:nx+2)
  f_hat += C3_1 * map(i -> a3_11 * f[i - 1] + a3_12 * f[i] + a3_13 * f[i + 1], 3:nx+2)
  f_hat += C3_2 * map(i -> a3_21 * f[i] + a3_22 * f[i + 1] + a3_23 * f[i + 2], 3:nx+2)

  
  f[4:end-2] -= Δt * diff(f_hat) / hx
  f[3] -= Δt * (f_hat[1] - f_hat[end]) / hx

  f[1] = f[end-3]
  f[2] = f[end-2]

  f[end-1] = f[3]
  f[end] = f[4]

  if i % 100 == 0
    display(plot(xs, f[3:end-2]))
  end
end

for (i,_) in enumerate(0:Δt:t_e)
  f_hat = map(i -> 1/2 * (f[i] + f[i - 1]) , 3:nx+3)
  
  f[3:end-2] -= Δt * diff(f_hat) / hx

  f[1] = f[end-3]
  f[2] = f[end-2]

  f[end-1] = f[3]
  f[end] = f[4]

  if i % 100 == 0
    display(plot(xs, f[3:end-2]))
  end
end
