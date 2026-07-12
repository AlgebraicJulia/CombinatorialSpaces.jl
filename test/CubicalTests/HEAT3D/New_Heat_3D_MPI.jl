# Heat_Dual3D_MPI.jl
#
# 3D heat equation on dual 0-forms (boids), MPI-parallel with HDF5 output.
# Uses the MPITopology worker/output split from UniformMPI.jl and the
# DataHandler pipeline from UniformIO.jl.
#
# Physics pipeline (u = dual 0-form on boids):
#   grad_u      = dual_derivative(Val(0), s, u)       boid → quad
#   flux_dual   = -k * grad_u
#   flux_primal = inv_hodge_star(Val(2), s, flux_dual) dual 2 → primal 2
#   div_flux    = exterior_derivative(Val(2), s, ...)  primal 2 → primal 3
#   laplacian_u = hodge_star(Val(3), s, div_flux)      primal 3 → dual 0
#
# Periodicity handled by halo exchange via DiscreteCallback before each RHS evaluation.

length(ARGS) == 7 ||
    error("Usage: mpiexecjl -n N julia New_Heat_3D_MPI.jl wy wx wz oy ox oz run_tag")

const w_dims = (parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
const o_dims = (parse(Int, ARGS[4]), parse(Int, ARGS[5]), parse(Int, ARGS[6]))
const RUN_TAG = ARGS[7]

const OUTDIR = joinpath(@__DIR__, "output", RUN_TAG)
const OUTFILE = joinpath(OUTDIR, "heat3D.h5")
const IMGDIR = joinpath(@__DIR__, "imgs", RUN_TAG)

using MPI
using HDF5
using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using DiffEqCallbacks
using Printf

include("../../../src/CubicalCode/UniformMesh.jl")
include("../../../src/CubicalCode/UniformMesh3D.jl")
include("../../../src/CubicalCode/UniformKernelDEC3D.jl")
include("../../../src/CubicalCode/UniformMPI.jl")
include("../../../src/CubicalCode/UniformIO.jl")
include("../../../src/CubicalCode/UniformPlotting.jl")

# ── Global problem parameters ─────────────────────────────────────────────────

const NXB_GLOBAL = 79
const NYB_GLOBAL = 79
const NZB_GLOBAL = 79
const NX_GLOBAL = NXB_GLOBAL + 1
const NY_GLOBAL = NYB_GLOBAL + 1
const NZ_GLOBAL = NZB_GLOBAL + 1
const m_dims = (NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL)

const LX = 5.0
const LY = 5.0
const LZ = 5.0
const K_DIFFUSION = 0.5
const T_START = 0.0
const T_END = 0.1
const DT = 0.001
const SAVEAT = 0.025
const HALO = 5
const PRINT_EVERY_N_STEPS = 50

const FT = Float64

# ── DataStream ────────────────────────────────────────────────────────────────

const datum = Datum{Boid,3}("snapshots", "fields", FT)
const stream = DataStream("heat3D", OUTFILE, [datum])

# ── Build topology ────────────────────────────────────────────────────────────

MPI.Init()
topo = MPITopology(m_dims, w_dims, o_dims; periods = (true, true, true))

world_comm = topo.world_comm
world_rank = MPI.Comm_rank(world_comm)
cart_comm = topo.cart_comm
cart_rank = MPI.Comm_rank(cart_comm)

println("ENTERING FIRST BARRIER AS RANK $world_rank")
MPI.Barrier(world_comm)
println("PAST FIRST BARRIER AS RANK $world_rank")

# if output(topo) # Output branch
#     handler = DataHandler(stream, topo)

#     output_leader(topo) && rm(OUTDIR; recursive = true, force = true)
#     MPI.Barrier(cart_comm)
#     output_leader(topo) && mkpath(OUTDIR)
#     MPI.Barrier(cart_comm)

#     create_hdf5!(handler, m_dims)
#     MPI.Barrier(cart_comm)

#     if output_leader(topo)
#         wc = out_cache(handler).worker_caches[1]
#         println("wc.mesh type:   ", typeof(wc.mesh))
#         println("wc.mesh dims:   ", (nx(wc.mesh), ny(wc.mesh), nz(wc.mesh)))
#         println("datum_dims:     ", datum_dims(datum, wc.mesh))
#         println("lm_dims:        ", wc.lm_dims)
#     end

#     while true
#         tag = output_from_worker(topo)
#         tag == SIGNAL_DONE && break
#         tag == SIGNAL_WRITE && write_output!(handler)
#     end
#     MPI.Barrier(cart_comm)

#     # plot global IC and final z-slices from HDF5
#     if output_leader(topo)
#         mkpath(IMGDIR)
#         z_mid = NZB_GLOBAL ÷ 2

#         h5open(OUTFILE, "r") do h5
#             dset = h5["fields/snapshots"]
#             ic_data = dset[1, :, :, z_mid]      # (NXB_GLOBAL, NYB_GLOBAL)
#             final_data = dset[end, :, :, z_mid]
#             cr = (0.0, maximum(ic_data) + eps())

#             for (data, label) in ((ic_data, "IC"), (final_data, "final"))
#                 fig = Figure(; size = (700, 600))
#                 ax = Axis(
#                     fig[1, 1];
#                     title = "Global z=$(z_mid) slice | $label",
#                     xlabel = "x",
#                     ylabel = "y",
#                 )
#                 hm = heatmap!(ax, data; colorrange = cr)
#                 Colorbar(fig[1, 2], hm)
#                 save(joinpath(IMGDIR, "global_$(lowercase(label))_zslice.png"), fig)
#                 println("Output leader | global $label slice saved.")
#             end

#             let
#                 gif_path = joinpath(IMGDIR, "global_heat3D_zslice.gif")
#                 extent, _ = HDF5.get_extent_dims(HDF5.dataspace(dset))
#                 ntimes = extent[1]
#                 frame_obs = Observable(dset[1, :, :, z_mid])

#                 fig = Figure(; size = (700, 600))
#                 ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
#                 hm = heatmap!(ax, frame_obs; colorrange = cr)
#                 Colorbar(fig[1, 2], hm)

#                 record(fig, gif_path, 1:ntimes; framerate = 10) do i
#                     frame_obs[] = dset[i, :, :, z_mid]
#                     return ax.title[] = "Global z=$(z_mid) slice | t = $(round((i-1) * SAVEAT, digits=3))"
#                 end
#                 println("Output leader | global gif saved.")
#             end
#         end
#     end

if output(topo)
    handler = DataHandler(stream, topo)

    output_leader(topo) && rm(OUTDIR; recursive = true, force = true)
    MPI.Barrier(cart_comm)
    output_leader(topo) && mkpath(OUTDIR)
    MPI.Barrier(cart_comm)

    create_hdf5!(handler, m_dims)
    MPI.Barrier(cart_comm)

    save_count = Ref(0)

    while true
        tag = output_from_worker(topo)
        tag == SIGNAL_DONE && break
        if tag == SIGNAL_WRITE
            write_output!(handler)
            save_count[] += 1

            if output_leader(topo)
                tbuf = handler.tile_buffers[1][1]

                nx_t, ny_t, nz_t = size(tbuf)
                x_mid = nx_t ÷ 2
                y_mid = ny_t ÷ 2
                z_mid = nz_t ÷ 2

                println("tbuf shape    : $(size(tbuf))")
                println("tbuf min/max  : $(minimum(tbuf)) / $(maximum(tbuf))")
                println("tbuf nonzeros : $(count(!iszero, tbuf))")
                println("tbuf sum      : $(sum(tbuf))")
                println("  x-slice nonzeros (x=$(x_mid)): $(count(!iszero, tbuf[x_mid, :, :]))")
                println("  y-slice nonzeros (y=$(y_mid)): $(count(!iszero, tbuf[:, y_mid, :]))")
                println("  z-slice nonzeros (z=$(z_mid)): $(count(!iszero, tbuf[:, :, z_mid]))")
                flush(stdout)

                mkpath(IMGDIR)
                cr = (0.0, maximum(tbuf) + eps())

                slices = (
                    (tbuf[x_mid, :, :], "x=$(x_mid)", "y", "z"),
                    (tbuf[:, y_mid, :], "y=$(y_mid)", "x", "z"),
                    (tbuf[:, :, z_mid], "z=$(z_mid)", "x", "y"),
                )

                fig = Figure(; size = (600, 1800))
                for (row, (slice_data, plane_label, xlabel, ylabel)) in enumerate(slices)
                    ax = Axis(
                        fig[row, 1];
                        title = "tbuf $(plane_label) slice | save $(save_count[])",
                        xlabel = xlabel,
                        ylabel = ylabel,
                    )
                    hm = heatmap!(ax, slice_data; colorrange = cr)
                    Colorbar(fig[row, 2], hm)  # colorbar to the right of each heatmap
                end

                save(joinpath(IMGDIR, @sprintf("tbuf_save%03d_allslices.png", save_count[])), fig)
                println("Output leader | combined slice figure saved (save $(save_count[])).")
            end
        end
    end
    MPI.Barrier(cart_comm)

else # Worker branch
    cache = topo.cache
    cart_coords = MPI.Cart_coords(cart_comm)
    cart_dims, _, _ = MPI.Cart_get(cart_comm)

    s, ghosts = worker_mesh(topo, (LX, LY, LZ); halo = (HALO, HALO, HALO))

    println("ghost send west size: ", length(ghosts.west.send))
    println("ghost recv west size: ", length(ghosts.west.recv))

    cart_rank == 0 &&
        println("Worker Cartesian grid: $(cart_dims[1])×$(cart_dims[2])×$(cart_dims[3])")
    println(
        "Worker rank $cart_rank | coords=$(cart_coords) | local real mesh: $(cache.lm_dims[1])×$(cache.lm_dims[2])×$(cache.lm_dims[3])",
    )
    MPI.Barrier(cart_comm)

    # ── Initial condition ─────────────────────────────────────────────────────

    u0 = zeros(FT, nboids(s))

    center = [LX / 2, LY / 2, LZ / 2]
    covariance = [0.1, 0.3, 0.5]
    dist = MvNormal(center, covariance)

    for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
        i = coord_to_boid(s, rx + hx(s), ry + hy(s), rz + hz(s))
        dp = real_dual_point(s, rx, ry, rz)
        u0[i] = pdf(dist, dp) * 10.0
    end
    global_integral(u, s, comm) = MPI.Allreduce(sum(interior(Val(3), u, s)), +, comm)

    mass_0 = global_integral(u0, s, cart_comm)
    cart_rank == 0 && println("Initial mass: $(mass_0)")

    # ── Plotting helpers ──────────────────────────────────────────────────────

    cart_rank == 0 && rm(IMGDIR; recursive = true, force = true)
    MPI.Barrier(cart_comm)
    cart_rank == 0 && mkpath(IMGDIR)
    MPI.Barrier(cart_comm)

    local_max = maximum(interior(Val(3), u0, s))
    global_max = MPI.Allreduce(local_max, max, cart_comm)

    # function plot_rank_slice(s, u, title_str, fname)
    #     slice_z = max(1, nzbr(s) ÷ 2)
    #     fig = plot_dual_zeroform_slice(
    #         s,
    #         u,
    #         Z_ALIGN,
    #         slice_z;
    #         figure_kwargs = (size = (600, 500),),
    #         heatmap_kwargs = (colorrange = (0.0, global_max + eps()),),
    #     )
    #     return save(fname, fig)
    # end

    # let title = "Rank $cart_rank | coords=$(cart_coords) | t=0.0"
    #     fname = joinpath(
    #         IMGDIR,
    #         @sprintf(
    #             "rank%03d_coords%d-%d-%d_IC_slice.png",
    #             cart_rank,
    #             cart_coords[1],
    #             cart_coords[2],
    #             cart_coords[3]
    #         )
    #     )
    #     plot_rank_slice(s, u0, title, fname)
    # end

    MPI.Barrier(cart_comm)

    # ── RHS ───────────────────────────────────────────────────────────────────

    function heat_rhs_mpi!(du, u, p, t)
        s, k = p
        grad_u = dual_derivative(Val(0), s, u)
        flux_dual = k .* grad_u
        flux_primal = inv_hodge_star(Val(2), s, flux_dual)
        div_flux = exterior_derivative(Val(2), s, flux_primal)
        laplacian_u = hodge_star(Val(3), s, div_flux)
        return du .= laplacian_u
    end

    p = (s, K_DIFFUSION)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    exchange_cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> exchange_boids_all!(integrator.u, ghosts, topo);
        initialize = (c, u, t, integrator) -> exchange_boids_all!(u, ghosts, topo),
        save_positions = (false, false),
    )

    save_cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            data = interior(Val(3), u, s)
            send_output!([data], stream, topo)
        end;
        funcat = collect(T_START:SAVEAT:T_END),
    )

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

    # ── Per-rank final plots ──────────────────────────────────────────────────

    # let title = "Rank $cart_rank | coords=$(cart_coords) | t=$(T_END)"
    #     fname = joinpath(
    #         IMGDIR,
    #         @sprintf(
    #             "rank%03d_coords%d-%d-%d_final_slice.png",
    #             cart_rank,
    #             cart_coords[1],
    #             cart_coords[2],
    #             cart_coords[3]
    #         )
    #     )
    #     plot_rank_slice(s, sol[end], title, fname)
    # end

    println("LEAVING WORKER RANK $world_rank")
    MPI.Barrier(cart_comm)
end

println("PROCESS RANK $world_rank WAITING TO LEAVE")
MPI.Barrier(world_comm)
println("CROSSED THE BARRIER, RANK $world_rank")
MPI.Finalize()