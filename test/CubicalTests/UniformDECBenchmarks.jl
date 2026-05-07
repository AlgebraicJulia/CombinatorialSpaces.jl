# UniformDECBenchmarks.jl
#
# Compares computation speed of matrix-based DEC operators (UniformMatrixDEC)
# with the two kernel-based implementations (UniformKernelDEC) on
# UniformCubicalComplex2D:
#
#   matrix apply     – sparse matrix-vector multiply via mul!
#   kernel apply     – KernelAbstractions kernel, coordinate work per thread
#   cached apply     – cached-index kernel (UniformDECCache), no per-thread
#                      integer division or branching

using BenchmarkTools
using Dates
using LinearAlgebra
using SparseArrays
using Printf

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")

const GRID_SIZES = [50, 200, 500]

suite = BenchmarkGroup()

for n in GRID_SIZES
  local s   = UniformCubicalComplex2D(n, n, 1.0 / n, 1.0 / n)
  local key = "$(n)x$(n)"
  suite[key] = BenchmarkGroup()

  local cache = UniformDECCache(s)

  local f0 = rand(nv(s))
  local f1 = rand(ne(s))
  local f2 = rand(nquads(s))

  local res_nv = zeros(nv(s))
  local res_ne = zeros(ne(s))
  local res_nq = zeros(nquads(s))

  # ── d0: exterior derivative 0-forms → 1-forms ───────────────────────────
  local d0 = exterior_derivative(Val(0), s)
  suite[key]["d0"] = BenchmarkGroup()
  suite[key]["d0"]["matrix apply"] = @benchmarkable mul!($res_ne, $d0, $f0)
  suite[key]["d0"]["kernel apply"] = @benchmarkable exterior_derivative!($res_ne, Val(0), $s, $f0)
  suite[key]["d0"]["cached apply"] = @benchmarkable exterior_derivative!($res_ne, Val(0), $cache, $f0)

  # ── d1: exterior derivative 1-forms → 2-forms ───────────────────────────
  local d1 = exterior_derivative(Val(1), s)
  suite[key]["d1"] = BenchmarkGroup()
  suite[key]["d1"]["matrix apply"] = @benchmarkable mul!($res_nq, $d1, $f1)
  suite[key]["d1"]["kernel apply"] = @benchmarkable exterior_derivative!($res_nq, Val(1), $s, $f1)
  suite[key]["d1"]["cached apply"] = @benchmarkable exterior_derivative!($res_nq, Val(1), $cache, $f1)

  # ── Dual derivatives ──────────────────────────────────────────────────────
  local dd0 = dual_derivative(Val(0), s)
  local dd1 = dual_derivative(Val(1), s)
  suite[key]["dd0"] = BenchmarkGroup()
  suite[key]["dd0"]["matrix apply"] = @benchmarkable mul!($res_ne, $dd0, $f2)
  suite[key]["dd0"]["cached apply"] = @benchmarkable dual_derivative!($res_ne, Val(0), $cache, $f2)

  suite[key]["dd1"] = BenchmarkGroup()
  suite[key]["dd1"]["matrix apply"] = @benchmarkable mul!($res_nv, $dd1, $f1)
  suite[key]["dd1"]["cached apply"] = @benchmarkable dual_derivative!($res_nv, Val(1), $cache, $f1)

  # ── Hodge stars ───────────────────────────────────────────────────────────
  local hs0 = hodge_star(Val(0), s)
  local hs1 = hodge_star(Val(1), s)
  local hs2 = hodge_star(Val(2), s)

  suite[key]["hodge_star_0"] = BenchmarkGroup()
  suite[key]["hodge_star_0"]["matrix apply"] = @benchmarkable mul!($res_nv, $hs0, $f0)
  suite[key]["hodge_star_0"]["cached apply"] = @benchmarkable hodge_star!($res_nv, Val(0), $cache, $f0)

  suite[key]["hodge_star_1"] = BenchmarkGroup()
  suite[key]["hodge_star_1"]["matrix apply"] = @benchmarkable mul!($res_ne, $hs1, $f1)
  suite[key]["hodge_star_1"]["cached apply"] = @benchmarkable hodge_star!($res_ne, Val(1), $cache, $f1)

  suite[key]["hodge_star_2"] = BenchmarkGroup()
  suite[key]["hodge_star_2"]["matrix apply"] = @benchmarkable mul!($res_nq, $hs2, $f2)
  suite[key]["hodge_star_2"]["cached apply"] = @benchmarkable hodge_star!($res_nq, Val(2), $cache, $f2)

  # ── Inverse Hodge stars ───────────────────────────────────────────────────
  local ihs0 = inv_hodge_star(Val(0), s)
  local ihs1 = inv_hodge_star(Val(1), s)
  local ihs2 = inv_hodge_star(Val(2), s)

  suite[key]["inv_hodge_star_0"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_0"]["matrix apply"] = @benchmarkable mul!($res_nv, $ihs0, $f0)
  suite[key]["inv_hodge_star_0"]["cached apply"] = @benchmarkable inv_hodge_star!($res_nv, Val(0), $cache, $f0)

  suite[key]["inv_hodge_star_1"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_1"]["matrix apply"] = @benchmarkable mul!($res_ne, $ihs1, $f1)
  suite[key]["inv_hodge_star_1"]["cached apply"] = @benchmarkable inv_hodge_star!($res_ne, Val(1), $cache, $f1)

  suite[key]["inv_hodge_star_2"] = BenchmarkGroup()
  suite[key]["inv_hodge_star_2"]["matrix apply"] = @benchmarkable mul!($res_nq, $ihs2, $f2)
  suite[key]["inv_hodge_star_2"]["cached apply"] = @benchmarkable inv_hodge_star!($res_nq, Val(2), $cache, $f2)

  # ── Laplacians ────────────────────────────────────────────────────────────
  local L0 = laplacian(Val(0), s)
  local L1 = laplacian(Val(1), s)
  local L2 = laplacian(Val(2), s)

  suite[key]["laplacian_0"] = BenchmarkGroup()
  suite[key]["laplacian_0"]["matrix apply"] = @benchmarkable mul!($res_nv, $L0, $f0)
  suite[key]["laplacian_0"]["cached apply"] = @benchmarkable laplacian!($res_nv, Val(0), $cache, $f0)

  suite[key]["laplacian_1"] = BenchmarkGroup()
  suite[key]["laplacian_1"]["matrix apply"] = @benchmarkable mul!($res_ne, $L1, $f1)
  suite[key]["laplacian_1"]["cached apply"] = @benchmarkable laplacian!($res_ne, Val(1), $cache, $f1)

  suite[key]["laplacian_2"] = BenchmarkGroup()
  suite[key]["laplacian_2"]["matrix apply"] = @benchmarkable mul!($res_nq, $L2, $f2)
  suite[key]["laplacian_2"]["cached apply"] = @benchmarkable laplacian!($res_nq, Val(2), $cache, $f2)

  # ── Dual Laplacians ───────────────────────────────────────────────────────
  local DL0 = dual_laplacian(Val(0), s)
  local DL1 = dual_laplacian(Val(1), s)
  local DL2 = dual_laplacian(Val(2), s)

  suite[key]["dual_laplacian_0"] = BenchmarkGroup()
  suite[key]["dual_laplacian_0"]["matrix apply"] = @benchmarkable mul!($res_nq, $DL0, $f2)
  suite[key]["dual_laplacian_0"]["cached apply"] = @benchmarkable dual_laplacian!($res_nq, Val(0), $cache, $f2)

  suite[key]["dual_laplacian_1"] = BenchmarkGroup()
  suite[key]["dual_laplacian_1"]["matrix apply"] = @benchmarkable mul!($res_ne, $DL1, $f1)
  suite[key]["dual_laplacian_1"]["cached apply"] = @benchmarkable dual_laplacian!($res_ne, Val(1), $cache, $f1)

  suite[key]["dual_laplacian_2"] = BenchmarkGroup()
  suite[key]["dual_laplacian_2"]["matrix apply"] = @benchmarkable mul!($res_nv, $DL2, $f0)
  suite[key]["dual_laplacian_2"]["cached apply"] = @benchmarkable dual_laplacian!($res_nv, Val(2), $cache, $f0)

  # ── Codifferentials ───────────────────────────────────────────────────────
  local cd1_mat = codifferential(Val(1), s)
  local cd2_mat = codifferential(Val(2), s)
  local dcd1_mat = dual_codifferential(Val(1), s)
  local dcd2_mat = dual_codifferential(Val(2), s)

  suite[key]["codifferential_1"] = BenchmarkGroup()
  suite[key]["codifferential_1"]["matrix apply"] = @benchmarkable mul!($res_nv, $cd1_mat, $f1)
  suite[key]["codifferential_1"]["cached apply"] = @benchmarkable codifferential!($res_nv, Val(1), $cache, $f1)

  suite[key]["codifferential_2"] = BenchmarkGroup()
  suite[key]["codifferential_2"]["matrix apply"] = @benchmarkable mul!($res_ne, $cd2_mat, $f2)
  suite[key]["codifferential_2"]["cached apply"] = @benchmarkable codifferential!($res_ne, Val(2), $cache, $f2)

  suite[key]["dual_codifferential_1"] = BenchmarkGroup()
  suite[key]["dual_codifferential_1"]["matrix apply"] = @benchmarkable mul!($res_nq, $dcd1_mat, $f1)
  suite[key]["dual_codifferential_1"]["cached apply"] = @benchmarkable dual_codifferential!($res_nq, Val(1), $cache, $f1)

  suite[key]["dual_codifferential_2"] = BenchmarkGroup()
  suite[key]["dual_codifferential_2"]["matrix apply"] = @benchmarkable mul!($res_ne, $dcd2_mat, $f0)
  suite[key]["dual_codifferential_2"]["cached apply"] = @benchmarkable dual_codifferential!($res_ne, Val(2), $cache, $f0)

  # ── Wedge products ────────────────────────────────────────────────────────
  suite[key]["wedge_01"] = BenchmarkGroup()
  suite[key]["wedge_01"]["kernel apply"] = @benchmarkable wedge_product(Val(0), Val(1), $s, $f0, $f1)
  suite[key]["wedge_01"]["cached apply"] = @benchmarkable wedge_product(Val(0), Val(1), $cache, $f0, $f1)

  suite[key]["wedge_11"] = BenchmarkGroup()
  suite[key]["wedge_11"]["kernel apply"] = @benchmarkable wedge_product(Val(1), Val(1), $s, $f1, $f1)
  suite[key]["wedge_11"]["cached apply"] = @benchmarkable wedge_product(Val(1), Val(1), $cache, $f1, $f1)

  suite[key]["wedge_dd_01"] = BenchmarkGroup()
  suite[key]["wedge_dd_01"]["kernel apply"] = @benchmarkable wedge_product_dd(Val(0), Val(1), $s, $f2, $f1)
  suite[key]["wedge_dd_01"]["cached apply"] = @benchmarkable wedge_product_dd(Val(0), Val(1), $cache, $f2, $f1)

  # ── Sharp and Flat ────────────────────────────────────────────────────────
  local X, Y = sharp_dd(s, f1)

  suite[key]["sharp_dd"] = BenchmarkGroup()
  suite[key]["sharp_dd"]["kernel apply"] = @benchmarkable sharp_dd($s, $f1)
  suite[key]["sharp_dd"]["cached apply"] = @benchmarkable sharp_dd($cache, $f1)

  suite[key]["flat_dp"] = BenchmarkGroup()
  suite[key]["flat_dp"]["kernel apply"] = @benchmarkable flat_dp($s, $X, $Y)
  suite[key]["flat_dp"]["cached apply"] = @benchmarkable flat_dp($cache, $X, $Y)

  suite[key]["flat_dd"] = BenchmarkGroup()
  suite[key]["flat_dd"]["kernel apply"] = @benchmarkable flat_dd($s, $X, $Y)
  suite[key]["flat_dd"]["cached apply"] = @benchmarkable flat_dd($cache, $X, $Y)

  # ── interpolate_dp (fused flat_dp ∘ sharp_dd) ─────────────────────────────
  # "kernel apply" = unfused two-step path (sharp_dd + synchronize + flat_dp)
  # "cached apply" = fused single-pass, no intermediate X/Y, no synchronize
  local res_ne_interp = zeros(ne(s))
  suite[key]["interpolate_dp"] = BenchmarkGroup()
  suite[key]["interpolate_dp"]["kernel apply"] = @benchmarkable interpolate_dp(Val(1), $s, $f1)
  suite[key]["interpolate_dp"]["cached apply"] = @benchmarkable interpolate_dp!($res_ne_interp, Val(1), $cache, $f1)
end

# ── Run ───────────────────────────────────────────────────────────────────────
println("Running benchmarks (this may take a few minutes)...")
results = run(suite; verbose = false, seconds = 5)

# ── Format and write results ──────────────────────────────────────────────────
const COL_OP  = 24
const COL_VAL = 16

function fmt_time(variants, vname)
  haskey(variants, vname) ? BenchmarkTools.prettytime(median(variants[vname]).time) : "-"
end

function write_results(io::IO)
  println(io, "run_timestamp = ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
  println(io)
  println(io, "=" ^ 88)
  println(io, "Uniform DEC Operator Benchmarks: Matrix vs Kernel vs Cached")
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
outfile   = joinpath(@__DIR__, "benchmark_results_$(timestamp).txt")

write_results(stdout)

open(outfile, "w") do io
  write_results(io)
end

println("\nResults written to: ", outfile)
