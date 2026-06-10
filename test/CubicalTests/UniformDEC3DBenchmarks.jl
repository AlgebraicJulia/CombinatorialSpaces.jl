# UniformDEC3DBenchmarks.jl
#
# Benchmarks for 3D Cubical DEC operators using KernelAbstractions kernels.
#
# Backend selection
# -----------------
# Set the environment variable BENCH_BACKEND before launching Julia:
#
#   BENCH_BACKEND=cpu  julia UniformDEC3DBenchmarks.jl   # default
#   BENCH_BACKEND=cuda julia UniformDEC3DBenchmarks.jl
#
# Operator group selection
# ------------------------
# Set the environment variable BENCH_OPS to a comma-separated list of groups:
#
#   BENCH_OPS=all                              # default
#   BENCH_OPS=exterior_derivative
#   BENCH_OPS=dual_derivative
#   BENCH_OPS=hodge_star
#   BENCH_OPS=inv_hodge_star
#   BENCH_OPS=wedge
#   BENCH_OPS=sharp_flat
#   BENCH_OPS=exterior_derivative,hodge_star   # multiple groups
#
# The variable is case-insensitive. Only one backend runs per invocation.

using BenchmarkTools
using Dates
using Printf
using KernelAbstractions

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")

# ── Backend selection ─────────────────────────────────────────────────────────
const _BENCH_BACKEND = "cuda"  # default

if _BENCH_BACKEND == "cuda"
    using CUDA
    using KernelAbstractions
    CUDA.allowscalar(false)
    const _BACKEND = CUDABackend()
    _dev_array(x)  = CuArray(x)
    _sync()        = CUDA.synchronize()
    println("Backend: CUDA  (", CUDA.name(CUDA.device()), ")")
elseif _BENCH_BACKEND == "cpu"
    const _BACKEND = CPU()
    _dev_array(x)  = x
    _sync()        = nothing
    println("Backend: CPU")
else
    error("Unknown BENCH_BACKEND=$(repr(_BENCH_BACKEND)).  Use \"cpu\" or \"cuda\".")
end

# ── Operator group selection ──────────────────────────────────────────────────
const _BENCH_OPS = Set([
    "exterior_derivative",
    "dual_derivative",
    "hodge_star",
    "inv_hodge_star",
    "wedge",
    "sharp_flat",
])

println("Operator groups: ", join(sort(collect(_BENCH_OPS)), ", "))

bench_op(group::String) = group in _BENCH_OPS

const GRID_SIZES = [51, 91, 131]

suite    = BenchmarkGroup()
_teardown = quote $(_sync()) end

for n in GRID_SIZES
    local s   = UniformCubicalComplex3D(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n)
    local key = "$(n)x$(n)x$(n)"
    suite[key] = BenchmarkGroup()

    local FT = Float64

    # Primal forms
    local f0 = _dev_array(rand(FT, nv(s)))
    local f1 = _dev_array(rand(FT, ne(s)))
    local f2 = _dev_array(rand(FT, nquads(s)))
    local f3 = _dev_array(rand(FT, nboids(s)))

    # Dual forms
    local d0 = _dev_array(rand(FT, nboids(s)))
    local d1 = _dev_array(rand(FT, nquads(s)))
    local d2 = _dev_array(rand(FT, ne(s)))
    local d3 = _dev_array(rand(FT, nv(s)))

    # Vector field components
    local X = _dev_array(rand(FT, nboids(s)))
    local Y = _dev_array(rand(FT, nboids(s)))
    local Z = _dev_array(rand(FT, nboids(s)))

    # ── 1. Exterior Derivatives ───────────────────────────────────────────
    if bench_op("exterior_derivative")
        suite[key]["d0"] = BenchmarkGroup()
        suite[key]["d0"]["kernel apply"] = @benchmarkable(exterior_derivative(Val(0), $s, $f0), teardown=$_teardown)

        suite[key]["d1"] = BenchmarkGroup()
        suite[key]["d1"]["kernel apply"] = @benchmarkable(exterior_derivative(Val(1), $s, $f1), teardown=$_teardown)

        suite[key]["d2"] = BenchmarkGroup()
        suite[key]["d2"]["kernel apply"] = @benchmarkable(exterior_derivative(Val(2), $s, $f2), teardown=$_teardown)
    end

    # ── 2. Dual Derivatives ───────────────────────────────────────────────
    if bench_op("dual_derivative")
        suite[key]["dd0"] = BenchmarkGroup()
        suite[key]["dd0"]["kernel apply"] = @benchmarkable(dual_derivative(Val(0), $s, $d0), teardown=$_teardown)

        suite[key]["dd1"] = BenchmarkGroup()
        suite[key]["dd1"]["kernel apply"] = @benchmarkable(dual_derivative(Val(1), $s, $d1), teardown=$_teardown)

        suite[key]["dd2"] = BenchmarkGroup()
        suite[key]["dd2"]["kernel apply"] = @benchmarkable(dual_derivative(Val(2), $s, $d2), teardown=$_teardown)
    end

    # ── 3. Hodge Stars ────────────────────────────────────────────────────
    if bench_op("hodge_star")
        suite[key]["hodge_star_0"] = BenchmarkGroup()
        suite[key]["hodge_star_0"]["kernel apply"] = @benchmarkable(hodge_star(Val(0), $s, $f0), teardown=$_teardown)

        suite[key]["hodge_star_1"] = BenchmarkGroup()
        suite[key]["hodge_star_1"]["kernel apply"] = @benchmarkable(hodge_star(Val(1), $s, $f1), teardown=$_teardown)

        suite[key]["hodge_star_2"] = BenchmarkGroup()
        suite[key]["hodge_star_2"]["kernel apply"] = @benchmarkable(hodge_star(Val(2), $s, $f2), teardown=$_teardown)

        suite[key]["hodge_star_3"] = BenchmarkGroup()
        suite[key]["hodge_star_3"]["kernel apply"] = @benchmarkable(hodge_star(Val(3), $s, $f3), teardown=$_teardown)
    end

    # ── 4. Inverse Hodge Stars ────────────────────────────────────────────
    if bench_op("inv_hodge_star")
        suite[key]["inv_hodge_star_0"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_0"]["kernel apply"] = @benchmarkable(inv_hodge_star(Val(0), $s, $d0), teardown=$_teardown)

        suite[key]["inv_hodge_star_1"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_1"]["kernel apply"] = @benchmarkable(inv_hodge_star(Val(1), $s, $d1), teardown=$_teardown)

        suite[key]["inv_hodge_star_2"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_2"]["kernel apply"] = @benchmarkable(inv_hodge_star(Val(2), $s, $d2), teardown=$_teardown)

        suite[key]["inv_hodge_star_3"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_3"]["kernel apply"] = @benchmarkable(inv_hodge_star(Val(3), $s, $d3), teardown=$_teardown)
    end

    # ── 5. Wedge Products ─────────────────────────────────────────────────
    if bench_op("wedge")
        suite[key]["wedge_11"] = BenchmarkGroup()
        suite[key]["wedge_11"]["kernel apply"] = @benchmarkable(wedge_product(Val(1), Val(1), $s, $f1, $f1), teardown=$_teardown)

        suite[key]["wedge_12"] = BenchmarkGroup()
        suite[key]["wedge_12"]["kernel apply"] = @benchmarkable(wedge_product(Val(1), Val(2), $s, $f1, $f2), teardown=$_teardown)

        suite[key]["wedge_dd_01"] = BenchmarkGroup()
        suite[key]["wedge_dd_01"]["kernel apply"] = @benchmarkable(wedge_product_dd(Val(0), Val(1), $s, $d0, $d1), teardown=$_teardown)
    end

    # ── 6. Sharp and Flat ─────────────────────────────────────────────────
    if bench_op("sharp_flat")
        suite[key]["sharp_dd"] = BenchmarkGroup()
        suite[key]["sharp_dd"]["kernel apply"] = @benchmarkable(sharp_dd($s, $d1), teardown=$_teardown)

        suite[key]["flat_dp"] = BenchmarkGroup()
        suite[key]["flat_dp"]["kernel apply"] = @benchmarkable(flat_dp($s, $X, $Y, $Z), teardown=$_teardown)
    end
end

# ── Format and write results ──────────────────────────────────────────────────
const COL_OP  = 24
const COL_VAL = 16

function fmt_time(variants, vname)
    haskey(variants, vname) ? BenchmarkTools.prettytime(median(variants[vname]).time) : "-"
end

function write_results(io::IO, results)
    println(io, "run_timestamp  = ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "backend        = ", _BENCH_BACKEND)
    println(io, "operator_groups = ", join(sort(collect(_BENCH_OPS)), ", "))
    println(io)
    println(io, "=" ^ 88)
    println(io, "Uniform 3D DEC Operator Benchmarks: Kernel  [", uppercase(_BENCH_BACKEND), "]")
    println(io, "=" ^ 88)

    for n in GRID_SIZES
        local key = "$(n)x$(n)x$(n)"
        local s   = UniformCubicalComplex3D(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n)
        println(io)
        println(io, "Grid $(n)×$(n)×$(n)  (nv=$(nv(s))  ne=$(ne(s))  nquads=$(nquads(s))  nboids=$(nboids(s)))")
        println(io, "  " * "-" ^ 84)
        @printf(io, "  %-*s  %s\n", COL_OP, "Operator", "kernel apply")
        println(io, "  " * "-" ^ 84)

        isempty(results[key]) && println(io, "  (no operators selected)")

        for op in sort(collect(keys(results[key])))
            v = results[key][op]
            @printf(io, "  %-*s  %s\n",
                COL_OP, op,
                fmt_time(v, "kernel apply"))
        end
    end
end

println("Tuning benchmarks (this may take a few minutes)...")
tune!(suite)

println("Running benchmarks...")
results = run(suite, verbose=true)

write_results(stdout, results)

timestamp = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
outdir    = joinpath(@__DIR__, "benchmarks", "3D")
mkpath(outdir)
ops_tag   = join(sort(collect(_BENCH_OPS)), "-")
outfile   = joinpath(outdir, "benchmark_results_$(timestamp)_$(_BENCH_BACKEND).txt")

open(outfile, "w") do io
    write_results(io, results)
end

println("\nResults written to: ", outfile)