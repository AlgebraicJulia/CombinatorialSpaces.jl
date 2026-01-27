using Plots

half_nx = 50
nx = half_nx * 2

xs = range(0, 1; length = nx)

f = zeros(nx)
f[1:half_nx] = map(x -> sin(2pi*x), xs[1:half_nx])
f[half_nx+1:end] = map(x -> 1 - sin(2pi*x), xs[half_nx+1:end])

scatter(xs, f)

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

f_hat = zeros(nx - 1)
f_hat[3:end] = map(i -> a3_01 * f[i - 2] + a3_02 * f[i - 1] + a3_03 * f[i], 3:nx-1)
f_hat[2:end] = map(i -> a3_11 * f[i - 1] + a3_12 * f[i] + a3_13 * f[i + 1], 2:nx-1)
f_hat[1:end-1] = map(i -> a3_21 * f[i] + a3_22 * f[i + 1] + a3_23 * f[i + 2], 1:nx-2)

scatter(xs, f_hat)
