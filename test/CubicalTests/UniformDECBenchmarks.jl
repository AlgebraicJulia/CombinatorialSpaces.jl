# UniformDECBenchmarks.jl
#
# Compares computation speed of matrix-based DEC operators (UniformMatrixDEC)
# with the two kernel-based implementations (UniformKernelDEC) on
# UniformCubicalComplex2D:
#
#   matrix apply     – sparse matrix-vector multiply via mul!
#                      (CPU-only; skipped for non-CPU backends)
#   kernel apply     – KernelAbstractions kernel, coordinate work per thread
#   cached apply     – cached-index kernel (UniformDECCache), no per-thread
#                      integer division or branching
#
# Backend selection
# -----------------
# Set the environment variable BENCH_BACKEND before launching Julia:
#
#   BENCH_BACKEND=cpu  julia UniformDECBenchmarks.jl   # default
#   BENCH_BACKEND=cuda julia UniformDECBenchmarks.jl
#
# The variable is case-insensitive.  Only one backend runs per invocation.

using BenchmarkTools
using Dates
using LinearAlgebra
using SparseArrays
using Printf
using Adapt

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")
include("../../src/CubicalCode/WENO.jl")
include("../../src/CubicalCode/UniformUpwinding.jl")

# ── Backend selection ─────────────────────────────────────────────────────────
# const _BENCH_BACKEND = lowercase(get(ENV, "BENCH_BACKEND", "cpu"))
const _BENCH_BACKEND = "cuda"

if _BENCH_BACKEND == "cuda"
  using CUDA
  using CUDA.CUSPARSE
  using KernelAbstractions
  CUDA.allowscalar(false)
  const _BACKEND        = CUDABackend()
  _dev_array(x)         = CuArray(x)
  _dev_matrix(A)        = CuSparseMatrixCSC{Float64}(A)
  _sync()               = CUDA.synchronize()
  println("Backend: CUDA  (", CUDA.name(CUDA.device()), ")")
elseif _BENCH_BACKEND == "cpu"
  using KernelAbstractions
  const _BACKEND        = CPU()
  _dev_array(x)         = x          # already on CPU
  _dev_matrix(A)        = A          # already on CPU
  _sync()               = nothing
  println("Backend: CPU")
else
  error("Unknown BENCH_BACKEND=$(repr(_BENCH_BACKEND)).  Use \"cpu\" or \"cuda\".")
end

const GRID_SIZES = [50, 200, 500]

suite = BenchmarkGroup()

# BenchmarkTools teardown hook – synchronizes the device after each sample so
# that GPU timings measure actual kernel completion rather than queue dispatch.
_teardown = quote $(_sync()) end

for n in GRID_SIZES
  local s   = UniformCubicalComplex2D(n, n, 1.0 / n, 1.0 / n)
  local key = "$(n)x$(n)"
  suite[key] = BenchmarkGroup()

  # Build cache on CPU then adapt to the target device.
  local cache_cpu = UniformDECCache(s)
  local cache     = Adapt.adapt(_BACKEND, cache_cpu)

  local f0 = _dev_array(rand(nv(s)))
  local f1 = _dev_array(rand(ne(s)))
  local f2 = _dev_array(rand(nquads(s)))

  local res_nv = _dev_array(zeros(nv(s)))
  local res_ne = _dev_array(zeros(ne(s)))
  local res_nq = _dev_array(zeros(nquads(s)))

  # ── d0: exterior derivative 0-forms → 1-forms ───────────────────────────
  local d0 = _dev_matrix(exterior_derivative(Val(0), s))
  suite[key]["d0"] = BenchmarkGroup()
  suite[key]["d0"]["matrix apply"] = @benchmarkable(mul!($res_ne, $d0, $f0), teardown=$_teardown)
  suite[key]["d0"]["kernel apply"] = @benchmarkable(exterior_derivative!($res_ne, Val(0), $s, $f0),    teardown=$_teardown)
  suite[key]["d0"]["cached apply"] = @benchmarkable(exterior_derivative!($res_ne, Val(0), $cache, $f0), teardown=$_teardown)

  # ── d1: exterior derivative 1-forms → 2-forms ───────────────────────────
  local d1 = _dev_matrix(exterior_derivative(Val(1), s))
  suite[key]["d1"] = BenchmarkGroup()
  suite[key]["d1"]["matrix apply"] = @benchmarkable(mul!($res_nq, $d1, $f1), teardown=$_teardown)
  suite[key]["d1"]["kernel apply"] = @benchmarkable(exterior_derivative!($res_nq, Val(1), $s, $f1),    teardown=$_teardown)
  suite[key]["d1"]["cached apply"] = @benchmarkable(exterior_derivative!($res_nq, Val(1), $cache, $f1), teardown=$_teardown)

  # ── Dual derivatives ──────────────────────────────────────────────────────
  local dd0 = _dev_matrix(dual_derivative(Val(0), s))
  suite[key]["dd0"] = BenchmarkGroup()
  suite[key]["dd0"]["matrix apply"] = @benchmarkable(mul!($res_ne, $dd0, $f2), teardown=$_teardown)
  suite[key]["dd0"]["cached apply"] = @benchmarkable(dual_derivative!($res_ne, Val(0), $cache, $f2), teardown=$_teardown)

  local dd1 = _dev_matrix(dual_derivative(Val(1), s))
  suite[key]["dd1"] = BenchmarkGroup()
  suite[key]["dd1"]["matrix apply"] = @benchmarkable(mul!($res_nv, $dd1, $f1), teardown=$_teardown)
  suite[key]["dd1"]["cached apply"] = @benchmarkable(dual_derivative!($res_nv, Val(1), $cache, $f1), teardown=$_teardown)

  local d_beta_mat = _dev_matrix(dual_derivative_beta(Val(1), s))
  suite[key]["d_beta_mul"] = BenchmarkGroup()
  suite[key]["d_beta_mul"]["matrix apply"] = @benchmarkable(mul!($res_nv, $d_beta_mat, $f1), teardown=$_teardown)
  suite[key]["d_beta_mul"]["cached apply"] = @benchmarkable(d_beta_mul!($res_nv, $cache, $f1), teardown=$_teardown)

  # ── Hodge stars ───────────────────────────────────────────────────────────
  local hs0 = _dev_matrix(hodge_star(Val(0), s))
  suite[key]["hodge_star_0"] = BenchmarkGroup()
  suite[key]["hodge_star_0"]["matrix apply"] = @benchmarkable(mul!($res_nv, $hs0, $f0), teardown=$_teardown)
  suite[key]["hodge_star_0"]["cached apply"] = @benchmarkable(hodge_star!($res_nv, Val(0), $cache, $f0), teardown=$_teardown)

  local hs1 = _dev_matrix(hodge_star(Val(1), s))
  suite[key]["hodge_star_1"] = BenchmarkGroup()
  suite[key]["hodge_star_1"]["matrix apply"] = @benchmarkable(mul!($res_ne, $hs1, $f1), teardown=$_teardown)
  suite[key]["hodge_star_1"]["cached apply"] = @benchmarkable(hodge_star!($res_ne, Val(1), $cache, $f1), teardown=$_teardown)

  local hs2 = _dev_matrix(hodge_star(Val(2), s))
  suite[key]["hodge_star_2"] = BenchmarkGroup()
  suite[key]["hodge_star_2"]["matrix apply"] = @benchmarkable(mul!($res_nq, $hs2, $f2), teardown=$_teardown)
  suite[key]["hodge_star_2"]["cached apply"] = @benchmarkable(hodge_star!($res_nq, Val(2), $cache, $f2), teardown=$_teardown)

  # ── Inverse Hodge stars ───────────────────────────────────────────────────
  local ihs0 = _dev_matrix(inv_hodge_star(Val(0), s))
  suite[key]["inv_hodge_star_0"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_0"]["matrix apply"] = @benchmarkable(mul!($res_nv, $ihs0, $f0), teardown=$_teardown)
  suite[key]["inv_hodge_star_0"]["cached apply"] = @benchmarkable(inv_hodge_star!($res_nv, Val(0), $cache, $f0), teardown=$_teardown)

  local ihs1 = _dev_matrix(inv_hodge_star(Val(1), s))
  suite[key]["inv_hodge_star_1"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_1"]["matrix apply"] = @benchmarkable(mul!($res_ne, $ihs1, $f1), teardown=$_teardown)
  suite[key]["inv_hodge_star_1"]["cached apply"] = @benchmarkable(inv_hodge_star!($res_ne, Val(1), $cache, $f1), teardown=$_teardown)

  local ihs2 = _dev_matrix(inv_hodge_star(Val(2), s))
  suite[key]["inv_hodge_star_2"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_2"]["matrix apply"] = @benchmarkable(mul!($res_nq, $ihs2, $f2), teardown=$_teardown)
  suite[key]["inv_hodge_star_2"]["cached apply"] = @benchmarkable(inv_hodge_star!($res_nq, Val(2), $cache, $f2), teardown=$_teardown)

  # ── Laplacians ────────────────────────────────────────────────────────────
  local L0 = _dev_matrix(laplacian(Val(0), s))
  suite[key]["laplacian_0"] = BenchmarkGroup()
  suite[key]["laplacian_0"]["matrix apply"] = @benchmarkable(mul!($res_nv, $L0, $f0), teardown=$_teardown)
  suite[key]["laplacian_0"]["cached apply"] = @benchmarkable(laplacian!($res_nv, Val(0), $cache, $f0), teardown=$_teardown)

  local L1 = _dev_matrix(laplacian(Val(1), s))
  local L1_tmp1 = _dev_array(zeros(nv(s)))
  local L1_tmp2 = _dev_array(zeros(nquads(s)))
  suite[key]["laplacian_1"] = BenchmarkGroup()
  suite[key]["laplacian_1"]["matrix apply"] = @benchmarkable(mul!($res_ne, $L1, $f1), teardown=$_teardown)
  suite[key]["laplacian_1"]["cached apply"] = @benchmarkable(laplacian!($res_ne, $L1_tmp1, $L1_tmp2, Val(1), $cache, $f1), teardown=$_teardown)

  local L2 = _dev_matrix(laplacian(Val(2), s))
  suite[key]["laplacian_2"] = BenchmarkGroup()
  suite[key]["laplacian_2"]["matrix apply"] = @benchmarkable(mul!($res_nq, $L2, $f2), teardown=$_teardown)
  suite[key]["laplacian_2"]["cached apply"] = @benchmarkable(laplacian!($res_nq, Val(2), $cache, $f2), teardown=$_teardown)

  # ── Dual Laplacians ───────────────────────────────────────────────────────
  local DL0 = _dev_matrix(dual_laplacian(Val(0), s))
  suite[key]["dual_laplacian_0"] = BenchmarkGroup()
  suite[key]["dual_laplacian_0"]["matrix apply"] = @benchmarkable(mul!($res_nq, $DL0, $f2), teardown=$_teardown)
  suite[key]["dual_laplacian_0"]["cached apply"] = @benchmarkable(dual_laplacian!($res_nq, Val(0), $cache, $f2), teardown=$_teardown)

  local DL1 = _dev_matrix(dual_laplacian(Val(1), s))
  local DL1_tmp1 = _dev_array(zeros(nv(s)))
  local DL1_tmp2 = _dev_array(zeros(nquads(s)))
  
  suite[key]["dual_laplacian_1"] = BenchmarkGroup()
  suite[key]["dual_laplacian_1"]["matrix apply"] = @benchmarkable(mul!($res_ne, $DL1, $f1), teardown=$_teardown)
  suite[key]["dual_laplacian_1"]["cached apply"] = @benchmarkable(dual_laplacian!($res_ne, $DL1_tmp1, $DL1_tmp2, Val(1), $cache, $f1), teardown=$_teardown)

  local DL2 = _dev_matrix(dual_laplacian(Val(2), s))
  suite[key]["dual_laplacian_2"] = BenchmarkGroup()
  suite[key]["dual_laplacian_2"]["matrix apply"] = @benchmarkable(mul!($res_nv, $DL2, $f0), teardown=$_teardown)
  suite[key]["dual_laplacian_2"]["cached apply"] = @benchmarkable(dual_laplacian!($res_nv, Val(2), $cache, $f0), teardown=$_teardown)

  # ── Codifferentials ───────────────────────────────────────────────────────
  local cd1_mat = _dev_matrix(codifferential(Val(1), s))
  suite[key]["codifferential_1"] = BenchmarkGroup()
  suite[key]["codifferential_1"]["matrix apply"] = @benchmarkable(mul!($res_nv, $cd1_mat, $f1), teardown=$_teardown)
  suite[key]["codifferential_1"]["cached apply"] = @benchmarkable(codifferential!($res_nv, Val(1), $cache, $f1), teardown=$_teardown)

  local cd2_mat = _dev_matrix(codifferential(Val(2), s))
  suite[key]["codifferential_2"] = BenchmarkGroup()
  suite[key]["codifferential_2"]["matrix apply"] = @benchmarkable(mul!($res_ne, $cd2_mat, $f2), teardown=$_teardown)
  suite[key]["codifferential_2"]["cached apply"] = @benchmarkable(codifferential!($res_ne, Val(2), $cache, $f2), teardown=$_teardown)

  local dcd1_mat = _dev_matrix(dual_codifferential(Val(1), s))
  suite[key]["dual_codifferential_1"] = BenchmarkGroup()
  suite[key]["dual_codifferential_1"]["matrix apply"] = @benchmarkable(mul!($res_nq, $dcd1_mat, $f1), teardown=$_teardown)
  suite[key]["dual_codifferential_1"]["cached apply"] = @benchmarkable(dual_codifferential!($res_nq, Val(1), $cache, $f1), teardown=$_teardown)

  local dcd2_mat = _dev_matrix(dual_codifferential(Val(2), s))
  suite[key]["dual_codifferential_2"] = BenchmarkGroup()
  suite[key]["dual_codifferential_2"]["matrix apply"] = @benchmarkable(mul!($res_ne, $dcd2_mat, $f0), teardown=$_teardown)
  suite[key]["dual_codifferential_2"]["cached apply"] = @benchmarkable(dual_codifferential!($res_ne, Val(2), $cache, $f0), teardown=$_teardown)

  # ── Wedge products ────────────────────────────────────────────────────────
  suite[key]["wedge_01"] = BenchmarkGroup()
  suite[key]["wedge_01"]["kernel apply"] = @benchmarkable(wedge_product(Val(0), Val(1), $s, $f0, $f1),    teardown=$_teardown)
  suite[key]["wedge_01"]["cached apply"] = @benchmarkable(wedge_product(Val(0), Val(1), $cache, $f0, $f1), teardown=$_teardown)

  suite[key]["wedge_11"] = BenchmarkGroup()
  suite[key]["wedge_11"]["kernel apply"] = @benchmarkable(wedge_product(Val(1), Val(1), $s, $f1, $f1),    teardown=$_teardown)
  suite[key]["wedge_11"]["cached apply"] = @benchmarkable(wedge_product(Val(1), Val(1), $cache, $f1, $f1), teardown=$_teardown)

  # ── Advection: upwind and WENO5 wedge_product_11 ─────────────────────────
  # "kernel apply" = uncached kernel (coord arithmetic per thread)
  # "cached apply" = cached kernel (precomputed edge indices, no integer div)
  local res_nq_adv  = _dev_array(zeros(nquads(s)))
  local acache_up   = Adapt.adapt(_BACKEND, AdvectionCache(Upwind(), s))
  local acache_w5   = Adapt.adapt(_BACKEND, AdvectionCache(WENO5(),  s))

  suite[key]["wedge_11_upwind"] = BenchmarkGroup()
  suite[key]["wedge_11_upwind"]["kernel apply"] = @benchmarkable(wedge_product_11!($res_nq_adv, Upwind(), $s,          $f1, $f1), teardown=$_teardown)
  suite[key]["wedge_11_upwind"]["cached apply"] = @benchmarkable(wedge_product_11!($res_nq_adv, Upwind(), $acache_up,  $f1, $f1), teardown=$_teardown)

  suite[key]["wedge_11_weno5"] = BenchmarkGroup()
  local w5_tmp_x = _dev_array(zeros(nquads(s)))
  local w5_tmp_y = _dev_array(zeros(nquads(s)))
  suite[key]["wedge_11_weno5"]["kernel apply"] = @benchmarkable(wedge_product_11!($res_nq_adv, WENO5(), $s,         $f1, $f1), teardown=$_teardown)
  suite[key]["wedge_11_weno5"]["cached apply"] = @benchmarkable(wedge_product_11!($res_nq_adv, $w5_tmp_x, $w5_tmp_y, WENO5(), $acache_w5, $f1, $f1), teardown=$_teardown)

  suite[key]["wedge_dd_01"] = BenchmarkGroup()
  suite[key]["wedge_dd_01"]["kernel apply"] = @benchmarkable(wedge_product_dd(Val(0), Val(1), $s, $f2, $f1),    teardown=$_teardown)
  suite[key]["wedge_dd_01"]["cached apply"] = @benchmarkable(wedge_product_dd(Val(0), Val(1), $cache, $f2, $f1), teardown=$_teardown)

  # ── Smoothing: dual 0-form (quad-based) ───────────────────────────────────
  # "matrix apply"  = pre-multiplied backward∘forward sparse matrix
  # "cached apply"  = fused kernel pass with single pre-allocated cache
  local sm_mat_comp = _dev_matrix(smoothing_dual0(s, -0.1) * smoothing_dual0(s, 0.1))
  local scache      = Adapt.adapt(_BACKEND, SmoothingCache(s, 0.1))
  local sm_tmp      = _dev_array(zeros(nquads(s)))

  suite[key]["smooth_dual0"] = BenchmarkGroup()
  suite[key]["smooth_dual0"]["matrix apply"] = @benchmarkable(
    mul!($res_nq, $sm_mat_comp, $f2), teardown=$_teardown)
  suite[key]["smooth_dual0"]["cached apply"] = @benchmarkable(
    smooth_dual0_fused!($res_nq, $sm_tmp, $scache, $f2),
    teardown=$_teardown)

  # ── Sharp and Flat ────────────────────────────────────────────────────────
  local X, Y = sharp_dd(s, f1)
  _sync()

  suite[key]["sharp_dd"] = BenchmarkGroup()
  suite[key]["sharp_dd"]["kernel apply"] = @benchmarkable(sharp_dd($s, $f1),    teardown=$_teardown)
  suite[key]["sharp_dd"]["cached apply"] = @benchmarkable(sharp_dd($cache, $f1), teardown=$_teardown)

  suite[key]["flat_dp"] = BenchmarkGroup()
  suite[key]["flat_dp"]["kernel apply"] = @benchmarkable(flat_dp($s, $X, $Y),    teardown=$_teardown)
  suite[key]["flat_dp"]["cached apply"] = @benchmarkable(flat_dp($cache, $X, $Y), teardown=$_teardown)

  suite[key]["flat_dd"] = BenchmarkGroup()
  suite[key]["flat_dd"]["kernel apply"] = @benchmarkable(flat_dd($s, $X, $Y),    teardown=$_teardown)
  suite[key]["flat_dd"]["cached apply"] = @benchmarkable(flat_dd($cache, $X, $Y), teardown=$_teardown)

  # ── interpolate_dp (fused flat_dp ∘ sharp_dd) ─────────────────────────────
  # "kernel apply" = unfused two-step path (sharp_dd + synchronize + flat_dp)
  # "cached apply" = fused single-pass, no intermediate X/Y, no synchronize
  local res_ne_interp = _dev_array(zeros(ne(s)))
  suite[key]["interpolate_dp"] = BenchmarkGroup()
  suite[key]["interpolate_dp"]["kernel apply"] = @benchmarkable(interpolate_dp(Val(1), $s, $f1),               teardown=$_teardown)
  suite[key]["interpolate_dp"]["cached apply"] = @benchmarkable(interpolate_dp!($res_ne_interp, Val(1), $cache, $f1), teardown=$_teardown)

  # ── set_periodic! ─────────────────────────────────────────────────────────
  # Only meaningful on a mesh with a halo; build a matching halo mesh/cache
  # once and reuse across all three form-degree benchmarks.
  local sh      = UniformCubicalComplex2D(n, n, 1.0 / n, 1.0 / n; halo_x = 1, halo_y = 1)
  local cacheh  = Adapt.adapt(_BACKEND, UniformDECCache(sh))
  local fh0     = _dev_array(rand(nv(sh)))
  local fh1     = _dev_array(rand(ne(sh)))
  local fh2     = _dev_array(rand(nquads(sh)))

  # Val{0} – vertices
  suite[key]["set_periodic_0"] = BenchmarkGroup()
  suite[key]["set_periodic_0"]["kernel apply"] = @benchmarkable(set_periodic!($(copy(fh0)), Val(0), $sh,     ALL), teardown=$_teardown)
  suite[key]["set_periodic_0"]["cached apply"] = @benchmarkable(set_periodic!($(copy(fh0)), Val(0), $cacheh, ALL), teardown=$_teardown)

  # Val{1} – edges
  suite[key]["set_periodic_1"] = BenchmarkGroup()
  suite[key]["set_periodic_1"]["kernel apply"] = @benchmarkable(set_periodic!($(copy(fh1)), Val(1), $sh,     ALL), teardown=$_teardown)
  suite[key]["set_periodic_1"]["cached apply"] = @benchmarkable(set_periodic!($(copy(fh1)), Val(1), $cacheh, ALL), teardown=$_teardown)

  # Val{2} – quads
  suite[key]["set_periodic_2"] = BenchmarkGroup()
  suite[key]["set_periodic_2"]["kernel apply"] = @benchmarkable(set_periodic!($(copy(fh2)), Val(2), $sh,     ALL), teardown=$_teardown)
  suite[key]["set_periodic_2"]["cached apply"] = @benchmarkable(set_periodic!($(copy(fh2)), Val(2), $cacheh, ALL), teardown=$_teardown)
end

# ── Run ───────────────────────────────────────────────────────────────────────
println("Running benchmarks on backend=", _BENCH_BACKEND, "  (this may take a few minutes)...")
results = run(suite; verbose = false, seconds = 5)

# ── Format and write results ──────────────────────────────────────────────────
const COL_OP  = 24
const COL_VAL = 16

function fmt_time(variants, vname)
  haskey(variants, vname) ? BenchmarkTools.prettytime(median(variants[vname]).time) : "-"
end

function write_results(io::IO)
  println(io, "run_timestamp = ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
  println(io, "backend       = ", _BENCH_BACKEND)
  println(io)
  println(io, "=" ^ 88)
  println(io, "Uniform DEC Operator Benchmarks: Matrix vs Kernel vs Cached  [", uppercase(_BENCH_BACKEND), "]")
  println(io, "=" ^ 88)

  for n in GRID_SIZES
    key = "$(n)x$(n)"
    s   = UniformCubicalComplex2D(n, n, 1.0 / n, 1.0 / n)
    println(io)
    println(io, "Grid $(n)×$(n)  (nv=$(nv(s))  ne=$(ne(s))  nquads=$(nquads(s)))")
    println(io, "  " * "-" ^ 84)
    @printf(io, "  %-*s  %-*s  %-*s  %s\n",
      COL_OP, "Operator",
      COL_VAL, "matrix apply",
      COL_VAL, "kernel apply",
      "cached apply")
    println(io, "  " * "-" ^ 84)

    for op in sort(collect(keys(results[key])))
      v = results[key][op]
      @printf(io, "  %-*s  %-*s  %-*s  %s\n",
        COL_OP, op,
        COL_VAL, fmt_time(v, "matrix apply"),
        COL_VAL, fmt_time(v, "kernel apply"),
        fmt_time(v, "cached apply"))
    end
  end
end

timestamp = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
outfile   = joinpath(@__DIR__, "benchmark_results_$(timestamp)_$(_BENCH_BACKEND).txt")

write_results(stdout)

open(outfile, "w") do io
  write_results(io)
end

println("\nResults written to: ", outfile)
