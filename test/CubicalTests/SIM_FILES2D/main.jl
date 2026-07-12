using MPI
using Printf
using TOML
MPI.Init()

const PROJECT_DIR = dirname(Base.active_project())
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformDEC.jl"))

# ── ARGS ──────────────────────────────────────────────────────────────────────
const w_dims  = (parse(Int, ARGS[1]), parse(Int, ARGS[2]))
const o_dims  = (parse(Int, ARGS[3]), parse(Int, ARGS[4]))
const LOG_DIR = ARGS[5]
const SIM_NAME = ARGS[6]

# TODO: Abstract this to a generic path
const CONFIG = TOML.parsefile(joinpath(@__DIR__, "Examples", "$SIM_NAME.toml"))

# ── Dirs ──────────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, LOG_DIR, "output")
const IMGDIR = joinpath(@__DIR__, LOG_DIR, "imgs")

let comm_world = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(comm_world)
    if world_rank == 0
        rm(OUTPUT_DIR; recursive = true, force = true)
        rm(IMGDIR; recursive = true, force = true)

        mkpath(OUTPUT_DIR)
        mkpath(IMGDIR)
    end
    MPI.Barrier(comm_world)
end


# ── Config ────────────────────────────────────────────────────────────────────
const CONFIG_FT = get(CONFIG["Metadata"], "float", "Float64")

const FT = Float64

# TODO: Fix this 
# if CONFIG_FT == "Float64"
#     Float64
# elseif CONFIG_FT == "Float32"
#     Float32
# else
#     world_rank == 0 && @warn "Invalid float set, defaulting to Float64"
#     Float64
# end

const NX = CONFIG["Mesh"]["nx"]
const NY = CONFIG["Mesh"]["ny"]
const LX = FT(CONFIG["Mesh"]["lx"])
const LY = FT(CONFIG["Mesh"]["ly"])
const HALO = CONFIG["Mesh"]["halo"]

const RE = FT(CONFIG["Physics"]["Re"])
const PR = FT(CONFIG["Physics"]["Pr"])
const TE = FT(CONFIG["Simulation"]["te"])

const DT = FT(get(CONFIG["Simulation"], "dt", floor(min(LX / NX, LY / NY) / 360 / 2, sigdigits=1)))
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])
const PERIODIC = CONFIG["Simulation"]["periodic"]

const m_dims = (NX, NY)   # vertex counts
const OUTFILE = joinpath(OUTPUT_DIR, "savedata.h5")

comm_world = MPI.COMM_WORLD
world_rank = MPI.Comm_rank(comm_world)
if world_rank == 0
    println("Running on a mesh size of $NX x $NY, halo size of $HALO")
    println("Re=$RE, Pr=$PR")
    println("Chosen time-step is $DT based on CFL=dx/u")
    println("Running until time $TE and saving every $SAVETIME for about $(TE/SAVETIME) saves")
end
MPI.Barrier(comm_world)


# ── Topology ──────────────────────────────────────────────────────────────────
const PERIODS = PERIODIC == "ALL"        ? (true, true)  :
                PERIODIC == "EASTWEST"   ? (true, false) :
                PERIODIC == "NORTHSOUTH" ? (false, true) :
                (false, false)

const topo = MPITopology(m_dims, w_dims, o_dims; periods = PERIODS)

const XPU = get(CONFIG["Metadata"], "xpu", "CPU")

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

MPI.Barrier(topo.world_comm)
MPI.Finalize()