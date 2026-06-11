# Heat_Dual2D_MPI.jl
# 2D heat equation on dual 0-forms (quads), MPI-parallel with HDF5 output.
# Mirrors Heat_Dual3D_MPI.jl for debugging MPI design.

using MPI
using HDF5
using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using DiffEqCallbacks
using Printf
using CairoMakie

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")
include("../../src/CubicalCode/UniformMPI2D.jl")   # to be written

# ── MPI Setup ─────────────────────────────────────────────────────────────────
MPI.Init()

comm  = MPI.COMM_WORLD
rank  = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

# ── Cartesian topology ────────────────────────────────────────────────────────
topo = MPITopology{2}(comm; periodic=(true, true), dims=(0, 0))
coords = MPI.Cart_coords(topo.comm)
dims, _ = MPI.Cart_get(topo.comm)

rank == 0 && println("Cartesian grid: $(dims[1])×$(dims[2])")

# ── Global problem parameters ─────────────────────────────────────────────────
const NXQ_GLOBAL = 79
const NYQ_GLOBAL = 79

const LX = 5.0
const LY = 5.0
const K_DIFFUSION = 0.5
const T_START     = 0.0
const T_END       = 1.0
const DT          = 0.001
const SAVEAT      = 0.025
const HALO        = 5
const PRINT_EVERY_N_STEPS = 250

FT = Float64

# ── Domain decomposition ──────────────────────────────────────────────────────
function decompose(n_global::Int, dim::Int32, coord::Int)
    base = n_global ÷ dim
    rem  = n_global % dim
    return base + (coord < rem ? 1 : 0) + 1
end

function global_offset(n_global::Int, dim::Int32, coord::Int)
    base = n_global ÷ dim
    rem  = n_global % dim
    return coord * base + min(coord, rem)
end

nxr_local = decompose(NXQ_GLOBAL, dims[1], coords[1])
nyr_local = decompose(NYQ_GLOBAL, dims[2], coords[2])

x_offset = global_offset(NXQ_GLOBAL, dims[1], coords[1])
y_offset = global_offset(NYQ_GLOBAL, dims[2], coords[2])

println("My rank is $rank and my offsets are $x_offset and $y_offset .")

local_lx = FT(nxr_local - 1) * LX / (NXQ_GLOBAL)
local_ly = FT(nyr_local - 1) * LY / (NYQ_GLOBAL)

base_x_ = FT(x_offset) * LX / (NXQ_GLOBAL)
base_y_ = FT(y_offset) * LY / (NYQ_GLOBAL)

s = UniformCubicalComplex2D(nxr_local, nyr_local,
                            local_lx, local_ly;
                            halo_x = HALO,
                            halo_y = HALO,
                            base_x = base_x_,
                            base_y = base_y_)


println("My rank is $rank and my base points are $(s.base_x) and $(s.base_y) .")
println("My rank is $rank and my local mesh lengths are $local_lx and $local_ly .")

println()

ghosts = ghost_quads(s)

println("Rank $rank | coords=$(coords) | local real mesh: $(nxr_local)×$(nyr_local)")
MPI.Barrier(topo.comm)

# ── Initial condition ─────────────────────────────────────────────────────────
# u is a dual 0-form living on quad centers
u0 = zeros(FT, nquads(s))

center     = [LX / 2, LY / 2]
covariance = [0.5, 0.1]
dist       = MvNormal(center, covariance)

for ry in 1:nyqr(s), rx in 1:nxqr(s)
    # Map real coords to halo-offset quad index
    q  = coord_to_quad(s, rx + hx(s), ry + hy(s))
    dp = real_dual_point(s, rx, ry)
    u0[q] = pdf(dist, [dp[1], dp[2]]) * 10.0
end

# After computing u0, before plotting
local_max  = maximum(interior(Val(2), u0, s))
global_max = MPI.Allreduce(local_max, max, topo.comm)

println("My rank is $rank and my local mass value is: $(sum(interior(Val(2), u0, s)) * quad_area(s))")

global_integral(u, s, comm) = MPI.Allreduce(sum(interior(Val(2), u, s)) * quad_area(s), +, comm)

mass_0 = global_integral(u0, s, topo.comm)
rank == 0 && println("Initial mass is $(mass_0)")

# ── Plotting setup ────────────────────────────────────────────────────────────
using CairoMakie

imgdir = "imgs/Heat2D_MPI"
rank == 0 && rm(imgdir, recursive=true, force=true)
rank == 0 && mkpath(imgdir)
MPI.Barrier(topo.comm)

function plot_rank_slice(s, u, title_str, fname)
    # Extract interior quad values on a 2D grid for plotting
    local_data = [u[coord_to_quad(s, x + hx(s), y + hy(s))]
                  for y in 1:nyqr(s), x in 1:nxqr(s)]

    fig = Figure(size=(600, 500))
    ax  = Axis(fig[1, 1], title=title_str, xlabel="x", ylabel="y")
    hm  = heatmap!(ax, local_data,
                   colorrange=(0.0, maximum(global_max) + eps()))
    Colorbar(fig[1, 2], hm)
    save(fname, fig)
end

# ── Initial condition plots ───────────────────────────────────────────────────
let title = "Rank $rank | coords=$(coords) | t=0.0"
    fname  = joinpath(imgdir,
                      @sprintf("rank%03d_coords%d-%d_IC.png",
                               rank, coords[1], coords[2]))
    plot_rank_slice(s, u0, title, fname)
    println("Rank $rank | IC plot saved to $fname")
end

MPI.Barrier(topo.comm)

# ── Physics RHS ───────────────────────────────────────────────────────────────
# Mirrors the 3D pipeline but in 2D:
#   u          : dual 0-form on quads
#   grad_u     = dual_derivative(Val(0), s, u)   quad → edge  (dual 0 → dual 1)
#   flux_dual  = -k * grad_u                      dual 1-form
#   flux_primal = inv_hodge_star(Val(1), s, flux_dual)  dual 1 → primal 1
#   div_flux   = exterior_derivative(Val(1), s, flux_primal)  primal 1 → primal 2
#   laplacian_u = hodge_star(Val(2), s, div_flux)  primal 2 → dual 0
#
# Periodicity handled entirely by halo exchange before each RHS evaluation.

# Build operators once after mesh construction, outside the RHS
dd0 = dual_derivative(Val(0), s)        # nquads → ne
ihs1 = inv_hodge_star(Val(1), s)                # ne  → ne       (diagonal)
d1  = exterior_derivative(Val(1), s)            # ne  → nquads
hs2 = hodge_star(Val(2), s)                     # nquads → nquads (diagonal)

function heat_rhs_mpi!(du, u, p, t)
    _, k, _, dd0, ihs1, d1, hs2 = p

    grad_u      = dd0 * u
    flux_dual   = k .* grad_u
    flux_primal = ihs1 * flux_dual
    div_flux    = d1 * flux_primal
    laplacian_u = hs2 * div_flux

    du .= laplacian_u
end

p = (s, K_DIFFUSION, topo, dd0, ihs1, d1, hs2)

# ── Progress callback ─────────────────────────────────────────────────────────
step_counter = Ref(0)

function progress_callback(u, t, integrator)
    step_counter[] += 1
    if step_counter[] % PRINT_EVERY_N_STEPS == 0
        t_    = integrator.t
        t_end = integrator.sol.prob.tspan[2]
        pct   = 100.0 * t_ / t_end
        println("Rank $(rank) | step $(step_counter[]) | t = $(round(t_, digits=4)) / $(t_end) ($(round(pct, digits=1))%)")
        flush(stdout)
    end
end

progress_cb = FunctionCallingCallback(progress_callback; func_everystep=true, func_start=false)

# ── Halo exchange callback ────────────────────────────────────────────────────
exchange_cb = DiscreteCallback(
    (u, t, integrator) -> true,
    integrator -> exchange_quads_all!(integrator.u, ghosts, topo);
    initialize = (c, u, t, integrator) -> exchange_quads_all!(u, ghosts, topo),
    save_positions = (false, false)
)

cb = CallbackSet(exchange_cb, progress_cb)

# ── ODE solve ─────────────────────────────────────────────────────────────────
prob = ODEProblem(heat_rhs_mpi!, u0, (T_START, T_END), p)

rank == 0 && println("Solving...")
sol = solve(prob, Tsit5(), saveat=SAVEAT, adaptive=false, dt=DT, callback=cb)
rank == 0 && println("Solve complete.")

mass_f = global_integral(sol[end], s, topo.comm)
rank == 0 && println("Final mass is $(mass_f)")

# ── Final condition plots ─────────────────────────────────────────────────────
let title = "Rank $rank | coords=$(coords) | t=$(T_END)"
    fname  = joinpath(imgdir,
                      @sprintf("rank%03d_coords%d-%d_final.png",
                               rank, coords[1], coords[2]))
    plot_rank_slice(s, sol[end], title, fname)
    println("Rank $rank | final plot saved to $fname")
end

MPI.Barrier(topo.comm)

# ── HDF5 output ───────────────────────────────────────────────────────────────

outdir  = "output_heat2D_mpi"
outfile = joinpath(outdir, "heat2D.h5")
rank == 0 && rm(outdir, recursive=true, force=true)
rank == 0 && mkpath(outdir)
MPI.Barrier(topo.comm)

n_times = length(sol.t)

h5open(outfile, "w", topo.comm, MPI.Info()) do fid

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = create_group(fid, "metadata")
    meta["global_dims"] = [NXQ_GLOBAL, NYQ_GLOBAL]
    meta["times"]       = collect(sol.t)
    meta["dt"]          = DT

    # ── Single 3D dataset: (x, y, time) ──────────────────────────────────
    # Unlimited time dimension allows appending later runs.
    # Chunk size = one full spatial snapshot per chunk.
    dset = create_dataset(fid, "snapshots",
                      datatype(Float64),
                      dataspace((n_times, NXQ_GLOBAL, NYQ_GLOBAL),
                                max_dims=(-1, NXQ_GLOBAL, NYQ_GLOBAL)),
                      chunk=(1, NXQ_GLOBAL, NYQ_GLOBAL),
                      dxpl_mpio=:collective)

    x_range = (x_offset + 1):(x_offset + nxqr(s))
    y_range = (y_offset + 1):(y_offset + nyqr(s))

    for (i, t) in enumerate(sol.t)
        local_data = reshape(interior(Val(2), sol[i], s), nxqr(s), nyqr(s))
        dset[i, x_range, y_range] = local_data
    end
end

rank == 0 && println("Output written to $outfile")
MPI.Barrier(topo.comm)

# ── Rank 0: read back and visualize global domain ─────────────────────────────
if rank == 0
    h5open(outfile, "r") do fid
        times = read(fid["metadata/times"])

        # ── Final snapshot ────────────────────────────────────────────────────
        snap_data = fid["snapshots"]   # [n_times X NXQ_GLOBAL × NYQ_GLOBAL]

        ic_data = snap_data[1, :, :]
        final_data = snap_data[end, :, :]

        global_max_final = maximum(final_data)
        global_max_ic    = maximum(ic_data)
        cr = (0.0, global_max_ic + eps())

        fig = Figure(size=(700, 600))
        ax  = Axis(fig[1, 1],
                   title  = "Global domain | t = $(round(times[end], digits=4))",
                   xlabel = "x", ylabel = "y")
        hm  = heatmap!(ax, final_data, colorrange=cr)
        Colorbar(fig[1, 2], hm)
        fname = joinpath(imgdir, "global_final.png")
        save(fname, fig)
        println("Rank 0 | global final plot saved to $fname")

        # ── IC snapshot for comparison ────────────────────────────────────────
        
        fig_ic  = Figure(size=(700, 600))
        ax_ic   = Axis(fig_ic[1, 1],
                       title  = "Global domain | t = $(round(times[1], digits=4))",
                       xlabel = "x", ylabel = "y")
        hm_ic   = heatmap!(ax_ic, ic_data, colorrange=cr)
        Colorbar(fig_ic[1, 2], hm_ic)
        fname_ic = joinpath(imgdir, "global_IC.png")
        save(fname_ic, fig_ic)
        println("Rank 0 | global IC plot saved to $fname_ic")

        # ── Animated gif over all snapshots ──────────────────────────────────
        fig_anim = Figure(size=(700, 600))
        ax_anim  = Axis(fig_anim[1, 1], xlabel="x", ylabel="y")
        data_obs = Observable(ic_data)
        hm_anim  = heatmap!(ax_anim, data_obs, colorrange=cr)
        Colorbar(fig_anim[1, 2], hm_anim)

        fname_gif = joinpath(imgdir, "global_evolution.gif")
        record(fig_anim, fname_gif, enumerate(times); framerate=8) do (i, t)
            data_obs[] = snap_data[i, :, :]
            ax_anim.title = "Global domain | t = $(round(t, digits=4))"
        end
        println("Rank 0 | global animation saved to $fname_gif")
    end
end

MPI.Barrier(topo.comm)
MPI.Finalize()