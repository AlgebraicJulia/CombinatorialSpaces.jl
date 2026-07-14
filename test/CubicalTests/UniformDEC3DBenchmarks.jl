using BenchmarkTools
using Dates
using Printf
using KernelAbstractions
using Adapt

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")
# If your cached implementation lives separately, include it here instead.
# include("../../src/CubicalCode/UniformKernelDEC3DCache.jl")

# ── Backend selection ─────────────────────────────────────────────────────────
const _BENCH_BACKEND = if "--cuda" in ARGS
    "cuda"
elseif "--amd" in ARGS
    "amd"
else
    "cpu"
end

if _BENCH_BACKEND == "cuda"
    using CUDA
    CUDA.allowscalar(false)
    const _BACKEND = CUDABackend()
    _dev_array(x) = CuArray(x)
    _sync() = CUDA.synchronize()
    println("Backend: CUDA  (", CUDA.name(CUDA.device()), ")")
elseif _BENCH_BACKEND == "cpu"
    const _BACKEND = CPU()
    _dev_array(x) = x
    _sync() = nothing
    println("Backend: CPU")
elseif _BENCH_BACKEND == "amd"
    using AMDGPU, SparseArrays
    AMDGPU.allowscalar(false)
    const _BACKEND = ROCBackend()
    _dev_array(x) = ROCArray(x)
    _dev_matrix(A) = ROCSparseMatrixCSR{Float64}(A)
    _sync() = AMDGPU.synchronize()
    println("Backend: AMD  (", AMDGPU.HIP.name(AMDGPU.device()), ")")
else
    error("Unknown BENCH_BACKEND=$(repr(_BENCH_BACKEND)). Use \"cpu\" or \"cuda\".")
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

const GRID_SIZES = [201]
suite     = BenchmarkGroup()
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

    # Cached 3D DEC data
    local cache_host = UniformDECCache3D(s)
    local cache = Adapt.adapt(_BACKEND, cache_host)

    # ── Preallocated outputs ────────────────────────────────────────────────
    # Exterior derivatives
    local out_d0 = _dev_array(zeros(FT, ne(s)))
    local out_d1 = _dev_array(zeros(FT, nquads(s)))
    local out_d2 = _dev_array(zeros(FT, nboids(s)))

    # Dual derivatives
    local out_dd0 = _dev_array(zeros(FT, nquads(s)))
    local out_dd1 = _dev_array(zeros(FT, ne(s)))
    local out_dd2 = _dev_array(zeros(FT, nv(s)))

    # Hodge stars
    local out_hs0 = _dev_array(zeros(FT, nv(s)))
    local out_hs1 = _dev_array(zeros(FT, ne(s)))
    local out_hs2 = _dev_array(zeros(FT, nquads(s)))
    local out_hs3 = _dev_array(zeros(FT, nboids(s)))

    # Inverse Hodge stars
    local out_ihs0 = _dev_array(zeros(FT, nv(s)))
    local out_ihs1 = _dev_array(zeros(FT, ne(s)))
    local out_ihs2 = _dev_array(zeros(FT, nquads(s)))
    local out_ihs3 = _dev_array(zeros(FT, nboids(s)))

    # Wedge products
    local out_w11 = _dev_array(zeros(FT, nquads(s)))
    local out_w12 = _dev_array(zeros(FT, nboids(s)))
    local out_wdd = _dev_array(zeros(FT, nquads(s)))

    # Sharp / flat
    local out_X   = _dev_array(zeros(FT, nboids(s)))
    local out_Y   = _dev_array(zeros(FT, nboids(s)))
    local out_Z   = _dev_array(zeros(FT, nboids(s)))
    local out_fdp = _dev_array(zeros(FT, ne(s)))

    # ── 1. Exterior Derivatives ───────────────────────────────────────────
    if bench_op("exterior_derivative")
        suite[key]["d0"] = BenchmarkGroup()
        suite[key]["d0"]["kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d0, Val(0), $s, $f0),
            teardown=$_teardown
        )
        suite[key]["d0"]["cached kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d0, Val(0), $cache, $f0),
            teardown=$_teardown
        )

        suite[key]["d1"] = BenchmarkGroup()
        suite[key]["d1"]["kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d1, Val(1), $s, $f1),
            teardown=$_teardown
        )
        suite[key]["d1"]["cached kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d1, Val(1), $cache, $f1),
            teardown=$_teardown
        )

        suite[key]["d2"] = BenchmarkGroup()
        suite[key]["d2"]["kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d2, Val(2), $s, $f2),
            teardown=$_teardown
        )
        suite[key]["d2"]["cached kernel apply!"] = @benchmarkable(
            exterior_derivative!($out_d2, Val(2), $cache, $f2),
            teardown=$_teardown
        )
    end

    # ── 2. Dual Derivatives ───────────────────────────────────────────────
    if bench_op("dual_derivative")
        suite[key]["dd0"] = BenchmarkGroup()
        suite[key]["dd0"]["kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd0, Val(0), $s, $d0),
            teardown=$_teardown
        )
        suite[key]["dd0"]["cached kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd0, Val(0), $cache, $d0),
            teardown=$_teardown
        )

        suite[key]["dd1"] = BenchmarkGroup()
        suite[key]["dd1"]["kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd1, Val(1), $s, $d1),
            teardown=$_teardown
        )
        suite[key]["dd1"]["cached kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd1, Val(1), $cache, $d1),
            teardown=$_teardown
        )

        suite[key]["dd2"] = BenchmarkGroup()
        suite[key]["dd2"]["kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd2, Val(2), $s, $d2),
            teardown=$_teardown
        )
        suite[key]["dd2"]["cached kernel apply!"] = @benchmarkable(
            dual_derivative!($out_dd2, Val(2), $cache, $d2),
            teardown=$_teardown
        )
    end

    # ── 3. Hodge Stars ────────────────────────────────────────────────────
    if bench_op("hodge_star")
        suite[key]["hodge_star_0"] = BenchmarkGroup()
        suite[key]["hodge_star_0"]["kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs0, Val(0), $s, $f0),
            teardown=$_teardown
        )
        suite[key]["hodge_star_0"]["cached kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs0, Val(0), $cache, $f0),
            teardown=$_teardown
        )

        suite[key]["hodge_star_1"] = BenchmarkGroup()
        suite[key]["hodge_star_1"]["kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs1, Val(1), $s, $f1),
            teardown=$_teardown
        )
        suite[key]["hodge_star_1"]["cached kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs1, Val(1), $cache, $f1),
            teardown=$_teardown
        )

        suite[key]["hodge_star_2"] = BenchmarkGroup()
        suite[key]["hodge_star_2"]["kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs2, Val(2), $s, $f2),
            teardown=$_teardown
        )
        suite[key]["hodge_star_2"]["cached kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs2, Val(2), $cache, $f2),
            teardown=$_teardown
        )

        suite[key]["hodge_star_3"] = BenchmarkGroup()
        suite[key]["hodge_star_3"]["kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs3, Val(3), $s, $f3),
            teardown=$_teardown
        )
        suite[key]["hodge_star_3"]["cached kernel apply!"] = @benchmarkable(
            hodge_star!($out_hs3, Val(3), $cache, $f3),
            teardown=$_teardown
        )
    end

    # ── 4. Inverse Hodge Stars ────────────────────────────────────────────
    if bench_op("inv_hodge_star")
        suite[key]["inv_hodge_star_0"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_0"]["kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs0, Val(0), $s, $f0),
            teardown=$_teardown
        )
        suite[key]["inv_hodge_star_0"]["cached kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs0, Val(0), $cache, $f0),
            teardown=$_teardown
        )

        suite[key]["inv_hodge_star_1"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_1"]["kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs1, Val(1), $s, $f1),
            teardown=$_teardown
        )
        suite[key]["inv_hodge_star_1"]["cached kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs1, Val(1), $cache, $f1),
            teardown=$_teardown
        )

        suite[key]["inv_hodge_star_2"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_2"]["kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs2, Val(2), $s, $f2),
            teardown=$_teardown
        )
        suite[key]["inv_hodge_star_2"]["cached kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs2, Val(2), $cache, $f2),
            teardown=$_teardown
        )

        suite[key]["inv_hodge_star_3"] = BenchmarkGroup()
        suite[key]["inv_hodge_star_3"]["kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs3, Val(3), $s, $f3),
            teardown=$_teardown
        )
        suite[key]["inv_hodge_star_3"]["cached kernel apply!"] = @benchmarkable(
            inv_hodge_star!($out_ihs3, Val(3), $cache, $f3),
            teardown=$_teardown
        )
    end

    # ── 5. Wedge Products ─────────────────────────────────────────────────
    if bench_op("wedge")
        suite[key]["wedge_11"] = BenchmarkGroup()
        suite[key]["wedge_11"]["kernel apply!"] = @benchmarkable(
            wedge_product!($out_w11, Val(1), Val(1), $s, $f1, $f1),
            teardown=$_teardown
        )
        suite[key]["wedge_11"]["cached kernel apply!"] = @benchmarkable(
            wedge_product!($out_w11, Val(1), Val(1), $cache, $f1, $f1),
            teardown=$_teardown
        )

        suite[key]["wedge_12"] = BenchmarkGroup()
        suite[key]["wedge_12"]["kernel apply!"] = @benchmarkable(
            wedge_product!($out_w12, Val(1), Val(2), $s, $f1, $f2),
            teardown=$_teardown
        )
        suite[key]["wedge_12"]["cached kernel apply!"] = @benchmarkable(
            wedge_product!($out_w12, Val(1), Val(2), $cache, $f1, $f2),
            teardown=$_teardown
        )

        suite[key]["wedge_dd_01"] = BenchmarkGroup()
        suite[key]["wedge_dd_01"]["kernel apply!"] = @benchmarkable(
            wedge_product_dd!($out_wdd, Val(0), Val(1), $s, $d0, $d1),
            teardown=$_teardown
        )
        suite[key]["wedge_dd_01"]["cached kernel apply!"] = @benchmarkable(
            wedge_product_dd!($out_wdd, Val(0), Val(1), $cache, $d0, $d1),
            teardown=$_teardown
        )
    end

    # ── 6. Sharp and Flat ─────────────────────────────────────────────────
    if bench_op("sharp_flat")
        suite[key]["sharp_dd"] = BenchmarkGroup()
        suite[key]["sharp_dd"]["kernel apply!"] = @benchmarkable(
            sharp_dd!($out_X, $out_Y, $out_Z, $s, $d1),
            teardown=$_teardown
        )
        # suite[key]["sharp_dd"]["cached kernel apply!"] = @benchmarkable(
        #     sharp_dd!($out_X, $out_Y, $out_Z, $cache, $d1),
        #     teardown=$_teardown
        # )

        suite[key]["flat_dp"] = BenchmarkGroup()
        suite[key]["flat_dp"]["kernel apply!"] = @benchmarkable(
            flat_dp!($out_fdp, $s, $X, $Y, $Z),
            teardown=$_teardown
        )
        # suite[key]["flat_dp"]["cached kernel apply!"] = @benchmarkable(
        #     flat_dp!($out_fdp, $cache, $X, $Y, $Z),
        #     teardown=$_teardown
        # )
    end
end

# ── Format and write results ──────────────────────────────────────────────────
const COL_OP   = 24
const COL_VAL  = 16
const COL_VAL2 = 20

function fmt_time(variants, vname)
    haskey(variants, vname) ? BenchmarkTools.prettytime(median(variants[vname]).time) : "-"
end

function write_results(io::IO, results)
    println(io, "run_timestamp   = ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "backend         = ", _BENCH_BACKEND)
    println(io, "operator_groups = ", join(sort(collect(_BENCH_OPS)), ", "))
    println(io)
    println(io, "=" ^ 108)
    println(io, "Uniform 3D DEC Operator Benchmarks: Kernel vs Cached Kernel  [", uppercase(_BENCH_BACKEND), "]")
    println(io, "=" ^ 108)

    for n in GRID_SIZES
        local key = "$(n)x$(n)x$(n)"
        local s   = UniformCubicalComplex3D(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n)

        println(io)
        println(io, "Grid $(n)×$(n)×$(n)  (nv=$(nv(s))  ne=$(ne(s))  nquads=$(nquads(s))  nboids=$(nboids(s)))")
        println(io, "  " * "-" ^ 104)
        @printf(io, "  %-*s  %-*s  %s\n", COL_OP, "Operator", COL_VAL, "kernel apply!", "cached kernel apply!")
        println(io, "  " * "-" ^ 104)

        isempty(results[key]) && println(io, "  (no operators selected)")

        for op in sort(collect(keys(results[key])))
            v = results[key][op]
            @printf(
                io,
                "  %-*s  %-*s  %s\n",
                COL_OP, op,
                COL_VAL, fmt_time(v, "kernel apply!"),
                fmt_time(v, "cached kernel apply!"),
            )
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