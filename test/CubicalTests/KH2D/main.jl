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
const RUN_TAG = ARGS[5]

# ── Dirs ──────────────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, RUN_TAG, "output")
const IMGDIR = joinpath(@__DIR__, RUN_TAG, "imgs")

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
const DT = FT(CONFIG["Simulation"]["dt"])
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])
const PERIODIC = CONFIG["Simulation"]["periodic"]

const m_dims = (NX + 1, NY + 1)   # vertex counts
const OUTFILE = joinpath(OUTPUT_DIR, "kh.h5")

# ── Topology ──────────────────────────────────────────────────────────────────
const PERIODS = PERIODIC == "ALL"        ? (true, true)  :
                PERIODIC == "EASTWEST"   ? (true, false) :
                PERIODIC == "NORTHSOUTH" ? (false, true) :
                (false, false)

const topo = MPITopology(m_dims, w_dims, o_dims; periods = PERIODS)

if output(topo)
    include(joinpath(@__DIR__, "output.jl"))
else
    include(joinpath(@__DIR__, "worker.jl"))
end

MPI.Barrier(topo.world_comm)
MPI.Finalize()