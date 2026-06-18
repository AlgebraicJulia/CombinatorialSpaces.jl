# Advection_Dual2D_MPI.jl
#
# 2D constant-velocity advection on dual 0-forms (quads), MPI-parallel with HDF5 output.
# Uses the MPITopology worker/output split from UniformMPI.jl and the
# DataHandler pipeline from UniformIO.jl.
#
# Physics pipeline (u = dual 0-form on quads, v = dual 1-form on edges):
#   w   = wedge_product_dd(Val(0), Val(1), s, u, v)   dual 0 ∧ dual 1 → dual 1
#   du/dt = -dual_codifferential(Val(1), s, w)
#         = -(hs2 * d1 * ihs1) * w                    dual 1 → dual 0
#
# Velocity: uniform x-direction, v_x = V_X * dual_edge_length_x, v_y = 0
# Periodicity handled by halo exchange via DiscreteCallback before each RHS evaluation.

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
include("../../src/CubicalCode/UniformMPI.jl")
include("../../src/CubicalCode/UniformIO.jl")

# ── Global problem parameters ─────────────────────────────────────────────────

const NXQ_GLOBAL = 79
const NYQ_GLOBAL = 79
const NX_GLOBAL = NXQ_GLOBAL + 1
const NY_GLOBAL = NYQ_GLOBAL + 1
const m_dims = (NX_GLOBAL, NY_GLOBAL)

const LX = 5.0
const LY = 5.0
const V_X = 1.0      # advection speed in x
const V_Y = 1.0
const T_START = 0.0
const T_END = 5.0    # one full period: LX / V_X = 5.0
const DT = 0.001
const SAVEAT = 0.1
const HALO = 5
const PRINT_EVERY_N_STEPS = 250

const OUTDIR = "output_advection2D_mpi"
const OUTFILE = joinpath(OUTDIR, "advection2D.h5")
const IMGDIR = "imgs/Advection2D_MPI"

FT = Float64

# ── DataStream ────────────────────────────────────────────────────────────────

const datum = Datum{Quad,2}("snapshots", "fields", FT)
const stream = DataStream("advection2D", OUTFILE, [datum])

# ── Build topology ────────────────────────────────────────────────────────────

const w_dims = (4, 4)
const o_dims = (2, 2)

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

            # ── GIF ───────────────────────────────────────────────────────────
            let
                gif_path = joinpath(IMGDIR, "global_advection2D.gif")
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

    ihs1 = inv_hodge_star(Val(1), s)
    d1 = exterior_derivative(Val(1), s)
    hs2 = hodge_star(Val(2), s)

    # ── Constant velocity dual 1-form ─────────────────────────────────────────
    # v is a dual 1-form on edges: x-edges carry V_X * dx_dual, y-edges are zero.
    # dx_dual = lx(s) / nxq(s) is the dual edge length in x on a uniform mesh.
    # Edges are laid out as [x-family..., y-family...] [3].

    v = zeros(FT, ne(s))

    # x-aligned dual edges live in the y-aligned primal edge index range [3]
    for y in 1:nye(s), x in 1:nx(s)
        e = coord_to_edge(s, x, y, Y_ALIGN)
        v[e] = V_X * dual_edge_len(s, x, y, X_ALIGN)
    end

    for y in 1:ny(s), x in 1:nxe(s)
        e = coord_to_edge(s, x, y, X_ALIGN)
        v[e] = V_Y * dual_edge_len(s, x, y, Y_ALIGN)
    end

    # ── Initial condition ─────────────────────────────────────────────────────

    u0 = zeros(FT, nquads(s))

    center = [LX / 2, LY / 2]
    covariance = [0.5, 0.5]
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

    # ── RHS ───────────────────────────────────────────────────────────────────

    function advection_rhs_mpi!(du, u, p, t)
        _, v, ihs1, d1, hs2 = p
        w = wedge_product_dd(Val(0), Val(1), s, u, v)
        return du .= -(hs2 * d1 * ihs1) * w
    end

    p = (s, v, ihs1, d1, hs2)

    # ── Callbacks ─────────────────────────────────────────────────────────────

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

    prob = ODEProblem(advection_rhs_mpi!, u0, (T_START, T_END), p)
    cart_rank == 0 && println("Solving...")
    sol = solve(prob, Tsit5(); saveat = SAVEAT, adaptive = false, dt = DT, callback = cb)
    cart_rank == 0 && println("Solve complete.")

    mass_f = global_integral(sol[end], s, cart_comm)
    cart_rank == 0 && println("Final mass: $(mass_f)")

    worker_to_output(SIGNAL_DONE, topo)
    MPI.Barrier(cart_comm)

    # ── Per-rank final plots ───────────────────────────────────────────────────

    let title = "Rank $cart_rank | coords=$(cart_coords) | t=$(T_END / 2)"
        fname = joinpath(
            IMGDIR,
            @sprintf("rank%03d_coords%d-%d_half.png", cart_rank, cart_coords[1], cart_coords[2])
        )
        half_idx = length(sol.t) ÷ 2
        plot_rank_slice(s, sol[half_idx], title, fname)
    end

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