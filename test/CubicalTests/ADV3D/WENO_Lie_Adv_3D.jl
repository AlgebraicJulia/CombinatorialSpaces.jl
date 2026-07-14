# New_Adv_3D_MPI.jl
#
# 3D constant-velocity advection on dual 0-forms (boids), MPI-parallel with HDF5 output.
# Uses the MPITopology worker/output split from UniformMPI.jl and the
# DataHandler pipeline from UniformIO.jl.
#
# Physics pipeline (u = dual 0-form on boids, v = dual 1-form on quads):
#   w     = wedge_product_dd(Val(0), Val(1), s, u, v)   dual 0 ∧ dual 1 → dual 1 (quads)
#   du/dt = -dual_codifferential(Val(1), s, w)
#         = -(hs3 * d2 * ihs2) * w                      quads → boids
#
# Velocity: uniform x-direction, v_x = V_X * dual_quad_area(s, x, y, z, X_ALIGN)
# Periodicity handled by halo exchange via DiscreteCallback before each RHS evaluation.
#
# Usage:
#   mpiexecjl -n N julia --project=PATH New_Adv_3D_MPI.jl wy wx wz run_tag

length(ARGS) == 4 || error("Usage: mpiexecjl -n N julia New_Adv_3D_MPI.jl wy wx wz run_tag")

const w_dims = (parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
const o_dims = (1, 1, 1)
const RUN_TAG = ARGS[4]

const OUTDIR = joinpath(@__DIR__, "output", RUN_TAG)
const OUTFILE = joinpath(OUTDIR, "advection3D.h5")
const IMGDIR = joinpath(@__DIR__, "imgs", RUN_TAG)

using MPI
using HDF5
using OrdinaryDiffEqSSPRK
using Distributions
using KernelAbstractions
using DiffEqCallbacks
using Printf
using CairoMakie

const SRC_PATH = "../../../src/CubicalCode/"

include(joinpath(SRC_PATH, "UniformMesh.jl"))
include(joinpath(SRC_PATH, "UniformMesh3D.jl"))
include(joinpath(SRC_PATH, "UniformKernelDEC.jl"))
include(joinpath(SRC_PATH, "UniformKernelDEC3D.jl"))
include(joinpath(SRC_PATH, "UniformMPI.jl"))
include(joinpath(SRC_PATH, "UniformIO.jl"))
include(joinpath(SRC_PATH, "UniformUpwinding.jl"))
include(joinpath(SRC_PATH, "WENO.jl"))

# ── Problem parameters ────────────────────────────────────────────────────────

const NXB_GLOBAL = 100
const NYB_GLOBAL = 100
const NZB_GLOBAL = 100
const NX_GLOBAL = NXB_GLOBAL + 1
const NY_GLOBAL = NYB_GLOBAL + 1
const NZ_GLOBAL = NZB_GLOBAL + 1
const m_dims = (NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL)

const LX = 5.0
const LY = 5.0
const LZ = 5.0
const V_X = -1.0
const V_Y = -1.0
const V_Z = -1.0
const T_START = 0.0
const T_END = 5.0    # one full period: LX / V_X = 5.0
const DT = 0.01
const SAVEAT = 0.1
const HALO = 15
const PRINT_EVERY_N_STEPS = 250

const FT = Float64

# ── DataStream ────────────────────────────────────────────────────────────────

const datum = Datum{Boid,3}("snapshots", "fields", FT)
const stream = DataStream("advection3D", OUTFILE, [datum])

# ── MPI init and topology ─────────────────────────────────────────────────────

MPI.Init()
topo = MPITopology(m_dims, w_dims, o_dims; periods = (true, true, true))

world_comm = topo.world_comm
world_rank = MPI.Comm_rank(world_comm)
cart_comm = topo.cart_comm
cart_rank = MPI.Comm_rank(cart_comm)

# println("ENTERING FIRST BARRIER AS RANK $world_rank")
MPI.Barrier(world_comm)
# println("PAST FIRST BARRIER AS RANK $world_rank")

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
            extent, _ = HDF5.get_extent_dims(HDF5.dataspace(dset))
            ntimes = extent[1]

            x_mid = NXB_GLOBAL ÷ 2
            y_mid = NYB_GLOBAL ÷ 2
            z_mid = NZB_GLOBAL ÷ 2

            ic_x = dset[1, x_mid, :, :]
            ic_y = dset[1, :, y_mid, :]
            ic_z = dset[1, :, :, z_mid]
            cr = (0.0, maximum(ic_x) + eps())

            # ── Static IC and final plots for all three slices ────────────────────

            for (slice_fn, axis_label, idx, ax1_label, ax2_label) in
                ((i -> dset[i, x_mid, :, :], "x", x_mid, "y", "z"), (i -> dset[i, :, y_mid, :], "y", y_mid, "x", "z"), (i -> dset[i, :, :, z_mid], "z", z_mid, "x", "y"))
                ic_data = slice_fn(1)
                fn_data = slice_fn(ntimes)

                for (data, label) in ((ic_data, "IC"), (fn_data, "final"))
                    fig = Figure(; size = (700, 600))
                    ax = Axis(fig[1, 1]; title = "Global $(axis_label)=$(idx) slice | $label", xlabel = ax1_label, ylabel = ax2_label)
                    hm = heatmap!(ax, data; colorrange = cr)
                    Colorbar(fig[1, 2], hm)
                    save(joinpath(IMGDIR, "global_$(lowercase(label))_$(axis_label)slice.png"), fig)
                    println("Output leader | global $label $(axis_label)-slice saved.")
                end

                # ── GIF ───────────────────────────────────────────────────────────

                gif_path = joinpath(IMGDIR, "global_advection3D_$(axis_label)slice.gif")
                frame_obs = Observable(slice_fn(1))
                fig = Figure(; size = (700, 600))
                ax = Axis(fig[1, 1]; xlabel = ax1_label, ylabel = ax2_label)
                hm = heatmap!(ax, frame_obs; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                record(fig, gif_path, 1:ntimes; framerate = 10) do i
                    frame_obs[] = slice_fn(i)
                    return ax.title[] = "Global $(axis_label)=$(idx) slice | t = $(round((i-1)*SAVEAT, digits=3))"
                end
                println("Output leader | global $(axis_label)-slice gif saved.")
            end
        end
    end

    # ── Worker branch ─────────────────────────────────────────────────────────────

else
    cache = topo.cache
    cart_coords = MPI.Cart_coords(cart_comm)
    cart_dims, _, _ = MPI.Cart_get(cart_comm)

    s = worker_mesh(topo, (LX, LY, LZ); halo = HALO)

    ex_stream = DataStream(Datum[Datum{Boid,3}("u", "fields", FT)])
    ex_handler = ExchangeHandler(ex_stream, topo, s)

    cart_rank == 0 && println("Worker Cartesian grid: $(cart_dims[1])×$(cart_dims[2])×$(cart_dims[3])")
    # println("Worker rank $cart_rank | coords=$(cart_coords) | " * "local real mesh: $(cache.lm_dims[1])×$(cache.lm_dims[2])×$(cache.lm_dims[3])")
    MPI.Barrier(cart_comm)

    # ── Constant velocity dual 1-form on quads ────────────────────────────────
    # v is a dual 1-form on quads. For uniform x-advection, only X-ALIGN (YZ) quads
    # carry flux: V_X * dual_quad_area(s, x, y, z, X_ALIGN).
    # Similarly for Y and Z components.

    v = zeros(FT, ne(s))
    xedges(s, v) .= V_X * dx(s)
    yedges(s, v) .= V_Y * dy(s)
    zedges(s, v) .= V_Z * dz(s)
    
    # ── Initial condition ─────────────────────────────────────────────────────

    # u0 = zeros(FT, nboids(s))
    # dist = MvNormal([LX / 2, LY / 2, LZ / 2], [0.3, 0.3, 0.3])

    # for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
    #     i = coord_to_boid(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s))
    #     dp = real_dual_point(s, rx, ry, rz)
    #     u0[i] = pdf(dist, [dp[1], dp[2], dp[3]]) * 10.0
    # end

    # ── Initial condition: solid cube ─────────────────────────────────────────
    u0 = zeros(FT, nboids(s))

    const CUBE_VAL = FT(1.0)
    const CUBE_X1 = FT(1.75)
    const CUBE_X2 = FT(3.25)
    const CUBE_Y1 = FT(1.75)
    const CUBE_Y2 = FT(3.25)
    const CUBE_Z1 = FT(1.75)
    const CUBE_Z2 = FT(3.25)

    for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
        i = coord_to_boid(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s))
        dp = real_dual_point(s, rx, ry, rz)
        x, y, z = dp[1], dp[2], dp[3]

        u0[i] = ifelse(
            (CUBE_X1 <= x <= CUBE_X2) &&
            (CUBE_Y1 <= y <= CUBE_Y2) &&
            (CUBE_Z1 <= z <= CUBE_Z2),
            CUBE_VAL,
            zero(FT),
        )
    end

    function global_integral(u, s, comm)
        return MPI.Allreduce(sum(interior(Val(3), u, s)), +, comm) * boid_volume(s)
    end

    mass_0 = global_integral(u0, s, cart_comm)
    cart_rank == 0 && println("Initial mass: $(mass_0)")

    # ── Plotting helpers ──────────────────────────────────────────────────────

    cart_rank == 0 && rm(IMGDIR; recursive = true, force = true)
    MPI.Barrier(cart_comm)
    cart_rank == 0 && mkpath(IMGDIR)
    MPI.Barrier(cart_comm)

    local_max = maximum(u0)
    global_max = MPI.Allreduce(local_max, max, cart_comm)
    cr = (0.0, global_max + eps())

    MPI.Barrier(cart_comm)

    # ── RHS ───────────────────────────────────────────────────────────────────

    weno_cache = AdvectionCache(WENO5(), s);

    function advection_rhs_mpi!(du, u, p, t)
        s, weno_cache, v, dd0, ihs2, hs3 = p
        q = dd0(u)                 # dual 1-form on quads
        p2 = ihs2(q)               # primal 2-form on quads
        w = wedge_product(Val(1), Val(2), WENO5(), weno_cache, v, p2)  # primal 3-form on boids
        return du .= -hs3(w)       # dual 0-form on boids
    end

    dd0  = f -> dual_derivative(Val(0), s, f)
    ihs2 = f -> inv_hodge_star(Val(2), s, f)
    hs3  = f -> hodge_star(Val(3), s, f)
    p = (s, weno_cache, v, dd0, ihs2, hs3)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    progress_cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            step = integrator.stats.naccept
            if step % PRINT_EVERY_N_STEPS == 0 && world_rank == 0
                pct = 100.0 * t / integrator.sol.prob.tspan[2]
                println("Worker $cart_rank | step $step | t = $(round(t, digits=4)) ($(round(pct, digits=1))%)")
                flush(stdout)
            end
        end;
        func_everystep = true,
        func_start = false,
    )

    exchange_cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> exchange!(ex_handler, (u = integrator.u,));
        initialize = (c, u, t, integrator) -> exchange!(ex_handler, (u = u,)),
        save_positions = (false, false),
    )

    save_cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            data = interior(Val(3), u, s)[:]
            send_output!([data], stream, topo)
        end;
        funcat = T_START:SAVEAT:T_END,
    )

    cb = CallbackSet(exchange_cb, save_cb, progress_cb)

    # ── Solve ─────────────────────────────────────────────────────────────────

    prob = ODEProblem(advection_rhs_mpi!, u0, (T_START, T_END), p)
    cart_rank == 0 && println("Solving...")
    sol = solve(prob, SSPRK33(); saveat = SAVEAT, adaptive = false, dt = DT, callback = cb)
    cart_rank == 0 && println("Solve complete.")

    mass_f = global_integral(sol[end], s, cart_comm)
    cart_rank == 0 && println("Final mass:  $(mass_f)")
    cart_rank == 0 && println("Mass drift:  $(round(100.0 * (mass_f - mass_0) / mass_0, digits=6))%")

    worker_to_output(SIGNAL_DONE, topo)
    MPI.Barrier(cart_comm)

    close!(ex_handler)
    MPI.Barrier(cart_comm)
end

MPI.Barrier(world_comm)
MPI.Finalize()