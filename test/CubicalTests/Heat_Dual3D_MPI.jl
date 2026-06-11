using MPI
using HDF5
using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using DiffEqCallbacks
using Printf

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")
include("../../src/CubicalCode/UniformMPI3D.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

# TODO: This should be updated with all the changes in the 2D sim

# ── MPI Setup ─────────────────────────────────────────────────────────────────
MPI.Init()

comm  = MPI.COMM_WORLD
rank  = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

# ── Cartesian topology ────────────────────────────────────────────────────────
topo = MPITopology{3}(comm; periodic=(true, true, true), dims=(0, 0, 1))
coords = MPI.Cart_coords(topo.comm)
dims, _   = MPI.Cart_get(topo.comm)

rank == 0 && println("Cartesian grid: $(dims[1])×$(dims[2])×$(dims[3])")

# ── Global problem parameters ─────────────────────────────────────────────────
const NX_GLOBAL = 80
const NY_GLOBAL = 80
const NZ_GLOBAL = 80

const NXB_GLOBAL = NX_GLOBAL - 1
const NYB_GLOBAL = NY_GLOBAL - 1
const NZB_GLOBAL = NZ_GLOBAL - 1

const LX = 5.0
const LY = 5.0
const LZ = 5.0
const K_DIFFUSION = 0.5
const T_START    = 0.0
const T_END      = 0.1
const DT         = 0.001
const SAVEAT = 0.025
const HALO = 1
const PRINT_EVERY_N_STEPS = 50

FT = Float64

# ── Domain decomposition ──────────────────────────────────────────────────────
function decompose(n_global::Int, dim::Int32, coord::Int)
    base  = n_global ÷ dim
    rem   = n_global % dim
    nlocal = base + (coord < rem ? 1 : 0)
    return nlocal
end

nxr_local = decompose(NX_GLOBAL, dims[1], coords[1])
nyr_local = decompose(NY_GLOBAL, dims[2], coords[2])
nzr_local = decompose(NZ_GLOBAL, dims[3], coords[3])

# Global offset of this rank's real cells (1-indexed)
function global_offset(n_global::Int, dim::Int32, coord::Int)
    base = n_global ÷ dim
    rem  = n_global % dim
    return coord * base + min(coord, rem)
end

x_offset = global_offset(NX_GLOBAL, dims[1], coords[1])
y_offset = global_offset(NY_GLOBAL, dims[2], coords[2])
z_offset = global_offset(NZ_GLOBAL, dims[3], coords[3])

s = UniformCubicalComplex3D(nxr_local, nyr_local, nzr_local,
                            FT(nxr_local) * LX / NX_GLOBAL,
                            FT(nyr_local) * LY / NY_GLOBAL,
                            FT(nzr_local) * LZ / NZ_GLOBAL;
                            halo_x=HALO, halo_y=HALO, halo_z=HALO,
                            base_x=FT(x_offset) * LX / NX_GLOBAL,
                            base_y=FT(y_offset) * LY / NY_GLOBAL,
                            base_z=FT(z_offset) * LZ / NZ_GLOBAL,
)

println("Rank $rank | coords=$(coords) | local real mesh: $(nxr_local)×$(nyr_local)×$(nzr_local)")
MPI.Barrier(topo.comm)

# ── Initial condition ─────────────────────────────────────────────────────────
u0 = zeros(FT, nboids(s))

center     = [LX / 2, LY / 2, LZ / 2]
covariance = [0.5, 0.5, 0.5]
dist       = MvNormal(center, covariance)

for rz in 1:nzr(s), ry in 1:nyr(s), rx in 1:nxr(s)
    i  = coord_to_boid(s, rx + hx(s), ry + hy(s), rz + hz(s))
    dp = real_dual_point(s, rx, ry, rz)
    u0[i] = pdf(dist, dp) * 10.0
end

global_integral(u, s, comm) = MPI.Allreduce(sum(interior(Val(3), u, s)), +, comm)

mass_0 = global_integral(u0, s, topo.comm)
rank == 0 && println("Initial mass is $(mass_0)")

# ── Debug: plot initial condition slice for each rank ─────────────────────────
imgdir = "imgs/Heat3D_MPI"
rank == 0 && rm(imgdir, recursive=true, force=true)
rank == 0 && mkpath(imgdir)
MPI.Barrier(topo.comm)

let slice_z = max(1, nzr_local ÷ 2)
    fig = plot_dual_zeroform_slice(
        s, u0, Z_ALIGN, slice_z;
        figure_kwargs=(size=(600, 500),),
        heatmap_kwargs=(colorrange=(0.0, maximum(interior(Val(3), u0, s)) + eps()),)
    )
    fname = joinpath(imgdir, @sprintf("rank%03d_coords%d-%d-%d_IC_slice.png",
                                      rank, coords[1], coords[2], coords[3]))
    save(fname, fig)
    println("Rank $rank | IC slice saved to $fname")
end

MPI.Barrier(topo.comm)

# ── Physics RHS ───────────────────────────────────────────────────────────────
# Mirrors Heat_Dual3D.jl [7] but without flux BC zeroing — periodicity is
# handled entirely by the ghost exchange before each RHS evaluation.

function heat_rhs_mpi!(du, u, p, t)
    s, k, topo = p

    exchange_boids_all!(u, s, topo)

    grad_u     = dual_derivative(Val(0), s, u)
    flux_dual  = -k .* grad_u
    flux_primal = inv_hodge_star(Val(2), s, flux_dual)
    div_flux   = exterior_derivative(Val(2), s, flux_primal)
    laplacian_u = hodge_star(Val(3), s, div_flux)

    du .= laplacian_u
end

# ── Progress callback ─────────────────────────────────────────────────────────

step_counter = Ref(0)

function progress_callback(u, t, integrator)
    step_counter[] += 1
    if step_counter[] % PRINT_EVERY_N_STEPS == 0
        t     = integrator.t
        t_end = integrator.sol.prob.tspan[2]
        pct   = 100.0 * t / t_end
        println("Rank $(rank) | step $(step_counter[]) | t = $(round(t, digits=4)) / $(t_end) ($(round(pct, digits=1))%)")
        flush(stdout)
    end
end

cb = FunctionCallingCallback(progress_callback; func_everystep=true, func_start=false)

# ── ODE solve ────────────────────────────────────────────────────────────────
p    = (s, K_DIFFUSION, topo)
prob = ODEProblem(heat_rhs_mpi!, u0, (T_START, T_END), p)

rank == 0 && println("Solving...")
sol = solve(prob, Tsit5(), saveat=SAVEAT, adaptive=false, dt=DT, callback=cb)
rank == 0 && println("Solve complete.")

mass_f = global_integral(sol[end], s, topo.comm)
rank == 0 && println("Final mass is $(mass_f)")
# rank == 0 && @assert isapprox(mass_0, mass_f, rtol=1e-4) "Mass not conserved: $mass_0 vs $mass_f"

# ── HDF5 output ───────────────────────────────────────────────────────────────
# Each rank writes its own file containing:
#   - the dual 0-form (boid values) at each saved timestep
#   - the global coordinate offset so snapshots can be reassembled
#
# File layout:
#   /metadata/
#       global_dims      [NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL]
#       local_dims       [nxr_local, nyr_local, nzr_local]
#       offsets          [x_offset, y_offset, z_offset]
#       times            [t0, t1, ..., tN]
#   /snapshots/
#       t_000            [nboids(s)] Float64

outdir  = "output_heat3D_mpi"
outfile = joinpath(outdir, "heat3D.h5")
rank == 0 && rm(outdir, recursive=true, force=true)
rank == 0 && mkpath(outdir)
MPI.Barrier(topo.comm)

n_real_local  = nboidsr(s)
n_real_global = NXB_GLOBAL * NYB_GLOBAL * NZB_GLOBAL

x_boid_offset = global_offset(NXB_GLOBAL, dims[1], coords[1])
y_boid_offset = global_offset(NYB_GLOBAL, dims[2], coords[2])
z_boid_offset = global_offset(NZB_GLOBAL, dims[3], coords[3])

times = length(sol.t)

h5open(outfile, "w", topo.comm, MPI.Info()) do fid

    if rank == 0
        meta = create_group(fid, "metadata")
        meta["global_dims"] = [NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL]
        meta["times"]       = collect(sol.t)
        meta["dt"]          = DT
    end

    snaps = create_group(fid, "snapshots")

    for (i, t) in enumerate(sol.t)
        key  = @sprintf("t_%03d", i - 1)
        dset = create_dataset(snaps, key, datatype(Float64),
                              ((NXB_GLOBAL, NYB_GLOBAL, NZB_GLOBAL),
                               (NXB_GLOBAL, NYB_GLOBAL, NZB_GLOBAL)),
                             )

        x_range = (x_boid_offset + 1):(x_boid_offset + nxbr(s))
        y_range = (y_boid_offset + 1):(y_boid_offset + nybr(s))
        z_range = (z_boid_offset + 1):(z_boid_offset + nzbr(s))

        local_data = reshape(interior(Val(3), sol[i], s),
                             nxbr(s), nybr(s), nzbr(s))

        dset[x_range, y_range, z_range] = local_data
    end
end
rank == 0 && println("Output written to $outdir/")

MPI.Barrier(topo.comm)

let slice_z = max(1, nzr_local ÷ 2)
    u_final = sol[end]

    fig = plot_dual_zeroform_slice(
        s, u_final, Z_ALIGN, slice_z;
        figure_kwargs  = (size=(600, 500),),
        heatmap_kwargs = (colorrange=(0.0, maximum(u_final) + eps()),)
    )

    fname = joinpath(imgdir, @sprintf("rank%03d_coords%d-%d-%d_final_slice.png",
                                      rank, coords[1], coords[2], coords[3]))
    save(fname, fig)
    println("Rank $rank | final slice saved to $fname")
end

MPI.Barrier(topo.comm)
MPI.Finalize()