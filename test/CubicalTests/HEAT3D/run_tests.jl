const PROJECT_DIR = joinpath(@__DIR__, "..", "..", "..")
const SCRIPT = joinpath(@__DIR__, "New_Heat_3D_MPI.jl")
const LOGDIR = joinpath(@__DIR__, "logs")

const CONFIGS = [
    ((1, 1, 1), (1, 1, 1)),
    ((2, 1, 1), (1, 1, 1)),
    ((1, 2, 1), (1, 1, 1)),
    ((1, 1, 2), (1, 1, 1)),
    ((2, 2, 1), (1, 1, 1)),
    ((2, 1, 2), (1, 1, 1)),
    ((1, 2, 2), (1, 1, 1)),
    ((2, 2, 2), (1, 1, 1)),
]

mkpath(LOGDIR)

for (w_dims, o_dims) in CONFIGS
    nprocs = prod(w_dims) + prod(o_dims)
    tag = "w$(w_dims[1])x$(w_dims[2])x$(w_dims[3])_o$(o_dims[1])x$(o_dims[2])x$(o_dims[3])"
    logfile = joinpath(LOGDIR, "$tag.log")

    cmd = `mpiexecjl -n $nprocs julia --project=$PROJECT_DIR $SCRIPT \
               $(w_dims[1]) $(w_dims[2]) $(w_dims[3]) \
               $(o_dims[1]) $(o_dims[2]) $(o_dims[3]) \
               $tag`

    print("[$tag] ($nprocs ranks) running... ")
    flush(stdout)

    t = time()
    result = run(pipeline(cmd; stdout = logfile, stderr = logfile); wait = true)
    elapsed = round(time() - t; digits = 1)

    status = success(result) ? "PASS" : "FAIL"
    println("$status ($(elapsed)s)  →  $logfile")
end