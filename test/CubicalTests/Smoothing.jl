using CairoMakie
using LinearAlgebra
using SparseArrays
using KernelAbstractions
using Adapt
using Test

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

##############################
##### GRID SETUP         #####
##############################

lx_ = ly_ = 2π
nx_ = ny_ = 501
s = UniformCubicalComplex2D(nx_, ny_, lx_, ly_)

##############################
##### INITIAL CONDITION  #####
##############################

# Sinusoidal initial condition on dual 0-forms (one per quad)
dps = dual_points(s)
freq = 25
f0 = [sin(freq * 2π * p[1] / lx_) * sin(freq * 2π * p[2] / ly_) for p in dps]

##############################
##### MATRIX SMOOTHING   #####
##############################

c_smooth = 1.0
forward_smooth  = smoothing_dual0(s,  c_smooth)
backward_smooth = smoothing_dual0(s, -c_smooth)
smoothing_mat   = backward_smooth * forward_smooth

# Each row sums to 1 → constant functions are preserved.
@test all(abs.(sum(forward_smooth,  dims=2) .- 1) .< 1e-10)
@test all(abs.(sum(backward_smooth, dims=2) .- 1) .< 1e-10)
@test all(abs.(sum(smoothing_mat,   dims=2) .- 1) .< 1e-10)

# After one application the high-frequency sinusoidal signal must be attenuated.
f_mat = smoothing_mat * f0
@test maximum(abs, f_mat) < maximum(abs, f0)

##############################
##### KERNEL SMOOTHING   #####
##############################

# SmoothingCache encodes the same two-pass operation as the matrix pair.
sm_cache = SmoothingCache(s, c_smooth)

# Diagonal weights should match the analytic formula.
@test sm_cache.diag_fwd ≈ 1.0 - c_smooth / 2
@test sm_cache.diag_bwd ≈ 1.0 + c_smooth / 2

# Apply one fused two-pass step and verify attenuation.
f_kernel = smooth_dual0_fused!(similar(f0), sm_cache, f0)
@test maximum(abs, f_kernel) < maximum(abs, f0)

# Single-step agreement between matrix and kernel (exact arithmetic path differs
# slightly, so allow a tight but not zero tolerance).
@test maximum(abs, f_kernel - f_mat) < 1e-10

##############################
##### MULTI-ITER AGREEMENT ###
##############################

# Run both methods in lockstep for N iterations and verify they stay close at
# every step.  Floating-point differences between sparse mat-vec and the kernel
# accumulate slowly, so we allow a per-step tolerance of 1e-10.
n_iters  = 50
f_mat_i    = copy(f0)
f_kernel_i = copy(f0)
tmp_buf    = similar(f0)

for i in 1:n_iters
  f_mat_i .= smoothing_mat * f_mat_i
  smooth_dual0_fused!(tmp_buf, sm_cache, f_kernel_i)
  f_kernel_i .= tmp_buf

  @test maximum(abs, f_kernel_i - f_mat_i) < 1e-10
end

# After 50 applications both paths should have noticeably attenuated the signal.
@test maximum(abs, f_mat_i)    < 0.99 * maximum(abs, f0)
@test maximum(abs, f_kernel_i) < 0.99 * maximum(abs, f0)

##############################
##### VISUAL OUTPUT      #####
##############################

mkpath("imgs/Smoothing")
save("imgs/Smoothing/Initial.png", plot_twoform(s, f0))
save("imgs/Smoothing/Smoothed_matrix_50_iters.png",  plot_twoform(s, f_mat_i))
save("imgs/Smoothing/Smoothed_kernel_50_iters.png",  plot_twoform(s, f_kernel_i))
save("imgs/Smoothing/Kernel_vs_matrix_diff.png",     plot_twoform(s, f_kernel - f_mat))

println("Max initial:       ", maximum(abs, f0))
println("Max matrix(1):     ", maximum(abs, f_mat))
println("Max kernel(1):     ", maximum(abs, f_kernel))
println("Max matrix(50):    ", maximum(abs, f_mat_i))
println("Max kernel(50):    ", maximum(abs, f_kernel_i))
println("Max kernel-matrix step 1: ", maximum(abs, f_kernel - f_mat))
println("Max kernel-matrix step 50: ", maximum(abs, f_kernel_i - f_mat_i))
println("Saved plots to imgs/Smoothing/")
