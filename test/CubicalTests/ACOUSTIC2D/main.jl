# kelvin_helmholtz_mpi/KH_MPI.jl
using MPI
using Printf
using TOML
MPI.Init()

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformDEC.jl"))

const CONFIG = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

# ── ARGS ──────────────────────────────────────────────────────────────────────
const w_dims  = (parse(Int, ARGS[1]), parse(Int, ARGS[2]))
const o_dims  = (parse(Int, ARGS[3]), parse(Int, ARGS[4]))
const LOG_DIR = ARGS[5]

# ── Dirs ──────────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, LOG_DIR, "output")
const IMGDIR = joinpath(@__DIR__, LOG_DIR, "imgs")

let comm_world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm_world)
    if rank == 0
        rm(OUTPUT_DIR; recursive = true, force = true)
        rm(IMGDIR; recursive = true, force = true)

        mkpath(OUTPUT_DIR)
        mkpath(IMGDIR)
    end
    MPI.Barrier(comm_world)
end


# ── Config ────────────────────────────────────────────────────────────────────
const FT = Float64
const NX = CONFIG["Mesh"]["nx"]
const NY = CONFIG["Mesh"]["ny"]
const LX = FT(CONFIG["Mesh"]["lx"])
const LY = FT(CONFIG["Mesh"]["ly"])
const HALO = CONFIG["Mesh"]["halo"]
const RE = FT(CONFIG["Physics"]["Re"])
const PR = FT(CONFIG["Physics"]["Pr"])
const TE = FT(CONFIG["Simulation"]["te"])
# const DT = FT(CONFIG["Simulation"]["dt"])
const DT = floor(min(LX / NX, LY / NY) / 360 / 2, sigdigits=1)
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])
const PERIODIC = CONFIG["Simulation"]["periodic"]

const m_dims = (NX + 1, NY + 1)   # vertex counts
const OUTFILE = joinpath(OUTPUT_DIR, "acoustic.h5")

let comm_world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm_world)
    if rank == 0
        println("Running on a mesh size of $NX x $NY, halo size of $HALO")
        println("Re=$RE, Pr=$PR")
        println("Chosen time-step is $DT based on CFL=dx/u")
        println("Running until time $TE and saving every $SAVETIME for about $(TE/SAVETIME) saves")
    end
    MPI.Barrier(comm_world)
end

# ── Topology ──────────────────────────────────────────────────────────────────
const PERIODS = PERIODIC == "ALL"        ? (true, true)  :
                PERIODIC == "EASTWEST"   ? (true, false) :
                PERIODIC == "NORTHSOUTH" ? (false, true) :
                (false, false)

const topo = MPITopology(m_dims, w_dims, o_dims; periods = PERIODS)

# const HAS_CUDA   = Base.find_package("CUDA")   !== nothing
const HAS_AMDGPU = Base.find_package("AMDGPU") !== nothing

if HAS_AMDGPU
    using AMDGPU
    AMDGPU.allowscalar(false)
    const USE_AMDGPU = AMDGPU.functional()
    # println("AMDGPU is functional: $USE_AMDGPU")
else
    const USE_AMDGPU = false
    # println("No GPU package found. Running on CPU.")
end

if output(topo)
    include(joinpath(@__DIR__, "output.jl"))
else
    include(joinpath(@__DIR__, "worker.jl"))
end

MPI.Barrier(topo.world_comm)
MPI.Finalize()