using MPI
using Printf
using TOML
MPI.Init()

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformDEC.jl"))

const CONFIG = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

# ── ARGS ──────────────────────────────────────────────────────────────────────
# Order: wx wy wz ox oy oz LOG_DIR
const w_dims  = (parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
const o_dims  = (parse(Int, ARGS[4]), parse(Int, ARGS[5]), parse(Int, ARGS[6]))
const LOG_DIR = ARGS[7]

# ── Dirs ──────────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, LOG_DIR, "output")
const IMGDIR     = joinpath(@__DIR__, LOG_DIR, "imgs")

let comm_world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm_world)
    if rank == 0
        rm(OUTPUT_DIR; recursive = true, force = true)
        rm(IMGDIR;     recursive = true, force = true)
        mkpath(OUTPUT_DIR)
        mkpath(IMGDIR)
    end
    MPI.Barrier(comm_world)
end

# ── Config ────────────────────────────────────────────────────────────────────
const FT      = Float64
const NX      = CONFIG["Mesh"]["nx"]
const NY      = CONFIG["Mesh"]["ny"]
const NZ      = CONFIG["Mesh"]["nz"]
const LX      = FT(CONFIG["Mesh"]["lx"])
const LY      = FT(CONFIG["Mesh"]["ly"])
const LZ      = FT(CONFIG["Mesh"]["lz"])
const HALO    = CONFIG["Mesh"]["halo"]
const RE      = FT(CONFIG["Physics"]["Re"])
const PR      = FT(CONFIG["Physics"]["Pr"])
const TE      = FT(CONFIG["Simulation"]["te"])
const DT      = floor(min(LX / NX, LY / NY, LZ / NZ) / 360 / 2, sigdigits = 1)
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])
const PERIODIC = CONFIG["Simulation"]["periodic"]

const m_dims  = (NX, NY, NZ)   # vertex counts
const OUTFILE = joinpath(OUTPUT_DIR, "acoustic.h5")

let comm_world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm_world)
    if rank == 0
        println("Running on a mesh size of $NX x $NY x $NZ, halo size of $HALO")
        println("Re=$RE, Pr=$PR")
        println("Chosen time-step is $DT based on CFL=dx/u")
        println("Running until time $TE and saving every $SAVETIME for about $(TE/SAVETIME) saves")
    end
    MPI.Barrier(comm_world)
end

# ── GPU detection ─────────────────────────────────────────────────────────────
const HAS_AMDGPU = Base.find_package("AMDGPU") !== nothing

if HAS_AMDGPU
    using AMDGPU
    AMDGPU.allowscalar(false)
    const USE_AMDGPU = AMDGPU.functional()
else
    const USE_AMDGPU = false
end

# TODO: Remove this to allow for use of GPU
# TODO: Actually, add a config option for GPU or no
const USE_AMDGPU = false

# ── Topology ──────────────────────────────────────────────────────────────────
# TODO: This should be able to support combinations of options
const PERIODS = if PERIODIC == "ALL"
    (true, true, true)
elseif PERIODIC == "EASTWEST"
    (true, false, false)
elseif PERIODIC == "NORTHSOUTH"
    (false, true, false)
elseif PERIODIC == "UPDOWN"
    (false, false, true)
else
    (false, false, false)
end

const topo = MPITopology(m_dims, w_dims, o_dims; periods = PERIODS)

if output(topo)
    include(joinpath(@__DIR__, "output.jl"))
else
    include(joinpath(@__DIR__, "worker.jl"))
end

MPI.Barrier(topo.world_comm)
MPI.Finalize()