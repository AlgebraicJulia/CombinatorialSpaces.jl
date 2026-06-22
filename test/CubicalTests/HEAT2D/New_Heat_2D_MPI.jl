# Heat_Dual2D_MPI.jl
#
# 2D heat equation on dual 0-forms (quads), MPI-parallel with HDF5 output.
# Uses the new MPITopology worker/output split from UniformMPI.jl and the
# DataHandler pipeline from UniformIO.jl.
#
# Physics pipeline (u = dual 0-form on quads):
#   grad_u      = dual_derivative(Val(0), s, u)     quad → edge
#   flux_dual   = -k * grad_u
#   flux_primal = inv_hodge_star(Val(1), s, flux)   dual 1 → primal 1
#   div_flux    = exterior_derivative(Val(1), s, ..) primal 1 → primal 2
#   laplacian_u = hodge_star(Val(2), s, div_flux)   primal 2 → dual 0
#
# Periodicity handled by halo exchange before each RHS evaluation.

# New_Heat_2D_MPI.jl — reads config from ARGS
# Usage: mpiexecjl -n N julia New_Heat_2D_MPI.jl wy wx oy ox run_tag

length(ARGS) == 6 || error("Usage: mpiexecjl -n N julia New_Heat_2D_MPI.jl wy wx oy ox run_tag")

const w_dims = (parse(Int, ARGS[1]), parse(Int, ARGS[2]))
const o_dims = (parse(Int, ARGS[3]), parse(Int, ARGS[4]))
const RUN_TAG = ARGS[5]

const OUTDIR = joinpath(@__DIR__, "output", RUN_TAG)
const OUTFILE = joinpath(OUTDIR, "heat2D.h5")
const IMGDIR = joinpath(@__DIR__, "imgs", RUN_TAG)

using MPI
using HDF5
using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using DiffEqCallbacks
using Printf
using CairoMakie

include("../../../src/CubicalCode/UniformMesh.jl")
include("../../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../../src/CubicalCode/UniformKernelDEC.jl")
include("../../../src/CubicalCode/UniformMPI.jl")
include("../../../src/CubicalCode/UniformIO.jl")

# ── Global problem parameters ─────────────────────────────────────────────────

const NXQ_GLOBAL = 79
const NYQ_GLOBAL = 79
const NX_GLOBAL = NXQ_GLOBAL + 1
const NY_GLOBAL = NYQ_GLOBAL + 1
const m_dims = (NX_GLOBAL, NY_GLOBAL)

const LX = 5.0
const LY = 5.0
const K_DIFFUSION = 0.5
const T_START = 0.0
const T_END = 1.0
const DT = 0.001
const SAVEAT = 0.025
const HALO = 5
const PRINT_EVERY_N_STEPS = 250

const FT = Float64

# ── DataStream ────────────────────────────────────────────────────────────────

const datum = Datum{Quad,2}("snapshots", "fields", FT)
const stream = DataStream("heat2D", OUTFILE, [datum])

# ── Build topology ────────────────────────────────────────────────────────────
# Worker/output split: workers run the PDE, outputs handle HDF5.

MPI.Init()
topo = MPITopology(m_dims, w_dims, o_dims; periods = (true, true))

world_comm = topo.world_comm
world_rank = MPI.Comm_rank(world_comm)
cart_comm = topo.cart_comm
cart_rank = MPI.Comm_rank(cart_comm)

println("ENTERING FIRST BARRIER AS RANK $world_rank")
MPI.Barrier(world_comm)
println("PAST FIRST BARRIER AS RANK $world_rank")

# ── Output branch ─────────────────────────────────────────────────────────────

if output(topo)
    handler = DataHandler(stream, topo)

    output_leader(topo) && rm(OUTDIR; recursive = true, force = true)
    MPI.Barrier(cart_comm)
    output_leader(topo) && mkpath(OUTDIR)
    MPI.Barrier(cart_comm)

    create_hdf5!(handler, m_dims)
    MPI.Barrier(cart_comm)

    # Busy-wait loop: receive a signal from paired worker leader over intercomm.
    while true
        tag = output_from_worker(topo)
        tag == SIGNAL_DONE && break
        tag == SIGNAL_WRITE && write_output!(handler)
    end
    MPI.Barrier(cart_comm)

    if output_leader(topo)
        mkpath(IMGDIR)

        h5open(OUTFILE, "r") do h5
            dset = h5["fields/snapshots"]
            extent = HDF5.get_extent_dims(HDF5.dataspace(dset))[1]
            ntimes = extent[1]
            ic_data = dset[1, :, :]
            final_data = dset[ntimes, :, :]
            cr = (0.0, maximum(ic_data) + eps())

            for (data, label) in ((ic_data, "IC"), (final_data, "final"))
                fig = Figure(; size = (700, 600))
                ax = Axis(fig[1, 1]; title = "Global domain | $label", xlabel = "x", ylabel = "y")
                hm = heatmap!(ax, data; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                save(joinpath(IMGDIR, "global_$(lowercase(label)).png"), fig)
                println("Output leader | global $label plot saved.")
            end

            # ── GIF ───────────────────────────────────────────────────────────────────
            let
                gif_path = joinpath(IMGDIR, "global_heat2D.gif")
                frame_obs = Observable(dset[1, :, :])

                fig = Figure(; size = (700, 600))
                ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
                hm = heatmap!(ax, frame_obs; colorrange = cr)
                Colorbar(fig[1, 2], hm)

                record(fig, gif_path, 1:ntimes; framerate = 10) do i
                    frame_obs[] = dset[i, :, :]
                    return ax.title[] = "Global domain | t = $(round((i-1) * SAVEAT, digits=3))"
                end
                println("Output leader | global gif saved.")
            end
        end
    end

    # ── Worker branch ─────────────────────────────────────────────────────────────

else
    cache = topo.cache
    cart_coords = MPI.Cart_coords(cart_comm)
    cart_dims, _, _ = MPI.Cart_get(cart_comm)

    s, ghosts = worker_mesh(topo, (LX, LY); halo = (HALO, HALO))

    cart_rank == 0 && println("Worker Cartesian grid: $(cart_dims[1])×$(cart_dims[2])")
    println(
        "Worker rank $cart_rank | coords=$(cart_coords) | local real mesh: $(cache.lm_dims[1])×$(cache.lm_dims[2])",
    )
    MPI.Barrier(cart_comm)

    # ── DEC operators ─────────────────────────────────────────────────────────

    dd0 = dual_derivative(Val(0), s)
    ihs1 = inv_hodge_star(Val(1), s)
    d1 = exterior_derivative(Val(1), s)
    hs2 = hodge_star(Val(2), s)

    # ── Initial condition ─────────────────────────────────────────────────────

    u0 = zeros(FT, nquads(s))

    center = [LX / 2, LY / 2]
    covariance = [0.5, 0.1]
    dist = MvNormal(center, covariance)

    for ry in 1:nyqr(s), rx in 1:nxqr(s)
        q = coord_to_quad(s, rx + hx(s), ry + hy(s))
        dp = real_dual_point(s, rx, ry)
        u0[q] = pdf(dist, [dp[1], dp[2]]) * 10.0
    end

    global_integral(u, s, comm) = MPI.Allreduce(sum(interior(Val(2), u, s)) * quad_area(s), +, comm)

    mass_0 = global_integral(u0, s, cart_comm)
    cart_rank == 0 && println("Initial mass: $(mass_0)")

    # ── Plotting helpers ──────────────────────────────────────────────────────

    cart_rank == 0 && rm(IMGDIR; recursive = true, force = true)
    MPI.Barrier(cart_comm)
    cart_rank == 0 && mkpath(IMGDIR)
    MPI.Barrier(cart_comm)

    local_max = maximum(interior(Val(2), u0, s))
    global_max = MPI.Allreduce(local_max, max, cart_comm)

    function plot_rank_slice(s, u, title_str, fname)
        local_data = [u[coord_to_quad(s, x + hx(s), y + hy(s))] for x in 1:nxqr(s), y in 1:nyqr(s)]
        fig = Figure(; size = (600, 500))
        ax = Axis(fig[1, 1]; title = title_str, xlabel = "x", ylabel = "y")
        hm = heatmap!(ax, local_data; colorrange = (0.0, global_max + eps()))
        Colorbar(fig[1, 2], hm)
        return save(fname, fig)
    end

    let title = "Rank $cart_rank | coords=$(cart_coords) | t=0.0"
        fname = joinpath(
            IMGDIR,
            @sprintf("rank%03d_coords%d-%d_IC.png", cart_rank, cart_coords[1], cart_coords[2])
        )
        plot_rank_slice(s, u0, title, fname)
    end

    MPI.Barrier(cart_comm)

    # ── RHS and callbacks ─────────────────────────────────────────────────────

    function heat_rhs_mpi!(du, u, p, t)
        _, k, _, dd0, ihs1, d1, hs2 = p
        grad_u = dd0 * u
        flux_dual = k .* grad_u
        flux_primal = ihs1 * flux_dual
        div_flux = d1 * flux_primal
        laplacian_u = hs2 * div_flux
        return du .= laplacian_u
    end

    p = (s, K_DIFFUSION, topo, dd0, ihs1, d1, hs2)

    progress_cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            step = integrator.stats.naccept
            if step % PRINT_EVERY_N_STEPS == 0
                pct = 100.0 * t / integrator.sol.prob.tspan[2]
                println(
                    "Worker $cart_rank | step $step | t = $(round(t, digits=4)) ($(round(pct, digits=1))%)",
                )
                flush(stdout)
            end
        end;
        func_everystep = true,
        func_start = false,
    )

    exchange_cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> exchange_quads_all!(integrator.u, ghosts, topo);
        initialize = (c, u, t, integrator) -> exchange_quads_all!(u, ghosts, topo),
        save_positions = (false, false),
    )

    save_cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            data = reshape(interior(Val(2), u, s), nxqr(s), nyqr(s))[:]
            send_output!([data], stream, topo)
        end;
        funcat = collect(T_START:SAVEAT:T_END),
    )

    cb = CallbackSet(exchange_cb, save_cb, progress_cb)

    # ── ODE solve ─────────────────────────────────────────────────────────────

    prob = ODEProblem(heat_rhs_mpi!, u0, (T_START, T_END), p)
    cart_rank == 0 && println("Solving...")
    sol = solve(prob, Tsit5(); saveat = SAVEAT, adaptive = false, dt = DT, callback = cb)
    cart_rank == 0 && println("Solve complete.")

    mass_f = global_integral(sol[end], s, cart_comm)
    cart_rank == 0 && println("Final mass: $(mass_f)")
    cart_rank == 0 && println("Mass drift: $(round(100.0 * (mass_f - mass_0) / mass_0, digits=6))%")

    worker_to_output(SIGNAL_DONE, topo)
    MPI.Barrier(cart_comm)

    # ── Per-rank final plots ───────────────────────────────────────────────────

    let title = "Rank $cart_rank | coords=$(cart_coords) | t=$(T_END)"
        fname = joinpath(
            IMGDIR,
            @sprintf("rank%03d_coords%d-%d_final.png", cart_rank, cart_coords[1], cart_coords[2])
        )
        plot_rank_slice(s, sol[end], title, fname)
    end

    println("LEAVING WORKER RANK $world_rank")
    MPI.Barrier(cart_comm)
end

println("PROCESS RANK $world_rank WAITING TO LEAVE")
MPI.Barrier(world_comm)
println("CROSSED THE BARRIER, RANK $world_rank")
MPI.Finalize()