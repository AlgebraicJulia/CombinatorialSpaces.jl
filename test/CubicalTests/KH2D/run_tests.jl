# kelvin_helmholtz_mpi/run_KH_MPI.jl
using Dates

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
const SCRIPT      = joinpath(@__DIR__, "main.jl")

# (wx, wy, ox, oy) — keep nprocs = wx*wy + ox*oy within your core count
const CONFIGS = [
    (1, 1, 1, 1),
    (1, 2, 1, 1),
    (2, 2, 1, 1),
    (2, 4, 1, 1),
    (4, 4, 1, 1),
    (4, 8, 1, 1),
    (8, 8, 1, 1),
    # (11, 11, 1, 1),
    # (2, 2, 2, 1),
    # (2, 2, 1, 2),
    # (2, 2, 2, 2),
    # (11, 11, 2, 2),
]

println("# KH-MPI Benchmark — $(Sys.cpu_info()[1].model)")
println("# $(length(CONFIGS)) configurations, CPU-only")
flush(stdout)

for (wx, wy, ox, oy) in CONFIGS
    nprocs  = wx * wy + ox * oy
    tag     = "w$(wx)x$(wy)_o$(ox)x$(oy)"

    savedir = joinpath(@__DIR__, tag, "logs")
    mkpath(savedir)

    timestamp = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")

    logfile = joinpath(savedir, "$(timestamp).log")
    errfile = joinpath(savedir, "err_$timestamp.log")

    cmd = `mpiexecjl -n $nprocs julia --project=$PROJECT_DIR $SCRIPT \
        $wx $wy $ox $oy $tag`

    print("[$tag] ($nprocs ranks) running... ")
    flush(stdout)

    t       = time()
    run(pipeline(cmd; stdout = logfile, stderr = errfile); wait = true)
    elapsed = round(time() - t; digits = 1)

    println("FINISHED ($(elapsed)s)  →  $logfile")
end