using MPI
using Printf
using TOML
MPI.Init()

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformDEC.jl"))

# ── ARGS ──────────────────────────────────────────────────────────────────────
const w_dims  = (parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
const o_dims  = (parse(Int, ARGS[4]), parse(Int, ARGS[5]), parse(Int, ARGS[6]))
const SIM_NAME = ARGS[7]

const CONFIG = TOML.parsefile(joinpath(@__DIR__, "Examples", "$SIM_NAME.toml"))

const LOG_DIR = CONFIG["Metadata"]["logpath"]

# ── Dirs ──────────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(LOG_DIR, "output")
const IMGDIR     = joinpath(LOG_DIR, "imgs")

comm_world = MPI.COMM_WORLD
world_rank = MPI.Comm_rank(comm_world)
if world_rank == 0
    rm(OUTPUT_DIR; recursive = true, force = true)
    rm(IMGDIR;     recursive = true, force = true)
    mkpath(LOG_DIR)
    mkpath(OUTPUT_DIR)
    mkpath(IMGDIR)
end
MPI.Barrier(comm_world)

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
const GRAVITY = get(CONFIG["Physics"], "Gravity", false)

const TE      = FT(CONFIG["Simulation"]["te"])
const DT      = FT(get(CONFIG["Simulation"], "dt", floor(min(LX / NX, LY / NY, LZ / NZ) / 360 / 2, sigdigits = 1)))
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])
const PERIODIC = CONFIG["Simulation"]["periodic"]

const m_dims  = (NX, NY, NZ)   # vertex counts
const OUTFILE = joinpath(OUTPUT_DIR, "acoustic.h5")

if world_rank == 0
    println("Running on a mesh size of $NX x $NY x $NZ, halo size of $HALO")
    println("Re=$RE, Pr=$PR")
    println("Chosen time-step is $DT based on CFL=dx/u")
    println("Running until time $TE and saving every $SAVETIME for about $(TE/SAVETIME) saves")
end
MPI.Barrier(comm_world)

# ── Topology ──────────────────────────────────────────────────────────────────
const EWPERIODIC = false
const NSPERIODIC = false
const UDPERIODIC = false
if "ALL" in PERIODIC
    const EWPERIODIC = true
    const NSPERIODIC = true
    const UDPERIODIC = true
elseif "EASTWEST" in PERIODIC
    const EWPERIODIC = true
elseif "NORTHSOUTH" in PERIODIC
    const NSPERIODIC = true
elseif "UPDOWN" in PERIODIC
    const UDPERIODIC = true
end

const PERIODS = (EWPERIODIC, NSPERIODIC, UDPERIODIC)

const topo = MPITopology(m_dims, w_dims, o_dims; periods = PERIODS)

if "--amd" in ARGS
    const XPU = "AMD"
else
    const XPU = "CPU"
end

use_amdgpu(use_xpu::String) = use_xpu == "AMD"
use_cpu(use_xpu::String) = use_xpu == "CPU"

if XPU == "AMD"
    using AMDGPU
    AMDGPU.allowscalar(false)
    const USE_AMDGPU = AMDGPU.functional()
    const USE_XPU = USE_AMDGPU ? "AMD" : "CPU"
    world_rank == 0 && println("AMDGPU is functional: $USE_AMDGPU")
else
    const USE_XPU = "CPU"
    world_rank == 0 && println("Defaulting to CPU")
end

if output(topo)
    include(joinpath(@__DIR__, "output.jl"))
else
    include(joinpath(@__DIR__, "worker.jl"))
end

world_rank == 0 & println("Closing out...")
MPI.Barrier(topo.world_comm)
MPI.Finalize()