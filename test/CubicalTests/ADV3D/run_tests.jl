# run_tests.jl  (lives in ADV3D/)

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
const SCRIPT = joinpath(@__DIR__, "New_Adv_3D_MPI.jl")
const LOGDIR = joinpath(@__DIR__, "logs")

const CONFIGS = [
    (4, 4, 4),
    # (2, 1, 1), (1, 2, 1), (1, 1, 2)
]

mkpath(LOGDIR)

for (wx, wy, wz) in CONFIGS
    nprocs = wx * wy * wz + 1
    tag = "w$(wx)x$(wy)x$(wz)_o1x1x1"
    logfile = joinpath(LOGDIR, "$tag.log")

    cmd = `mpiexecjl -n $nprocs julia --project=$PROJECT_DIR $SCRIPT \
               $wx $wy $wz $tag`

    print("[$tag] ($nprocs ranks) running... ")
    flush(stdout)

    t = time()
    result = run(pipeline(cmd; stdout = logfile, stderr = logfile); wait = true)
    elapsed = round(time() - t; digits = 1)

    status = success(result) ? "PASS" : "FAIL"
    println("$status ($(elapsed)s)  →  $logfile")
end